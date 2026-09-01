"""Core drift-diffusion physics: Bernoulli function, nonlinear Poisson (Newton),
and Scharfetter-Gummel continuity solves.

Equations solved (steady state, 1D), all in physical units (cm, s, V, C):

Poisson:      eps * d2(psi)/dx2 = q*(n - p - Cdop)
              n = ni*exp((psi-phin)/Vt),  p = ni*exp((phip-psi)/Vt)

Continuity:   dJn/dx =  q*R          dJp/dx = -q*R
              R = SRH recombination rate (net recombination - generation)

Currents (Scharfetter-Gummel, exponentially fitted, exact for piecewise-linear psi):
  Jn_{i+1/2} = (q*Dn/h_i) * [ n_{i+1}*B(u_{i+1}-u_i) - n_i*B(u_i-u_{i+1}) ]
  Jp_{i+1/2} = (q*Dp/h_i) * [ p_i*B(u_{i+1}-u_i) - p_{i+1}*B(u_i-u_{i+1}) ]
  where u = psi/Vt, B(x) = x/(exp(x)-1) is the Bernoulli function.
  (This reduces to zero net current for the equilibrium Boltzmann profile,
  the defining sanity check for the SG discretization.)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from params import Q, Material


def bernoulli(x: np.ndarray) -> np.ndarray:
    """Numerically safe Bernoulli function B(x) = x/(exp(x)-1)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    small = np.abs(x) < 1e-8
    out[small] = 1.0 - x[small] / 2.0  # Taylor series near 0

    big_pos = x > 40.0
    out[big_pos] = x[big_pos] * np.exp(-x[big_pos])  # avoid overflow, ->0

    big_neg = x < -40.0
    out[big_neg] = -x[big_neg]  # exp(x)->0, B(x) -> -x

    mid = ~(small | big_pos | big_neg)
    out[mid] = x[mid] / np.expm1(x[mid])
    return out


def equilibrium_bulk_potential(mat: Material, Cdop: float) -> float:
    """Exact charge-neutral bulk potential (psi with n=p=ni at psi=0) for a
    given net doping Cdop = Nd-Na, solving n0-p0=Cdop, n0*p0=ni^2 exactly
    (valid even when |Cdop| is not >> ni)."""
    ni = mat.ni
    n0 = (Cdop + np.sqrt(Cdop ** 2 + 4 * ni ** 2)) / 2.0
    return mat.Vt * np.log(n0 / ni)


def _control_volumes(x: np.ndarray) -> np.ndarray:
    """Finite-volume cell width around each node (half the sum of neighboring
    spacings; half-width at the boundary nodes)."""
    h = np.diff(x)
    W = np.zeros_like(x)
    W[1:-1] = (h[:-1] + h[1:]) / 2.0
    W[0] = h[0] / 2.0
    W[-1] = h[-1] / 2.0
    return W


def solve_poisson(x, Cdop, mat: Material, phin, phip, psi_guess,
                   tol=1e-10, max_iter=100, damping_cap=None):
    """Newton solve of the nonlinear Poisson equation for psi(x), given fixed
    quasi-Fermi levels phin(x), phip(x) (both zero at equilibrium).

    Dirichlet BC: psi[0] and psi[-1] are held fixed at psi_guess[0], psi_guess[-1].
    """
    N = len(x)
    h = np.diff(x)
    psi = psi_guess.copy()
    Vt = mat.Vt
    eps = mat.eps

    hm = h[:-1]   # h_{i-1}, for interior i=1..N-2
    hp = h[1:]    # h_i
    cvol = (hm + hp) / 2.0
    lap_coeff_m = eps / hm / cvol
    lap_coeff_p = eps / hp / cvol

    for it in range(max_iter):
        n = mat.ni * np.exp((psi - phin) / Vt)
        p = mat.ni * np.exp((phip - psi) / Vt)

        F = np.zeros(N)
        lower = np.zeros(N)
        diag = np.zeros(N)
        upper = np.zeros(N)

        # Dirichlet rows
        diag[0] = 1.0
        diag[-1] = 1.0

        F[1:-1] = (lap_coeff_p * (psi[2:] - psi[1:-1]) - lap_coeff_m * (psi[1:-1] - psi[:-2])) \
            - Q * (n[1:-1] - p[1:-1] - Cdop[1:-1])
        lower[1:-1] = lap_coeff_m
        upper[1:-1] = lap_coeff_p
        diag[1:-1] = -(lap_coeff_m + lap_coeff_p) - Q * (n[1:-1] + p[1:-1]) / Vt

        J = sp.diags([lower[1:], diag, upper[:-1]], offsets=[-1, 0, 1], format="csc")
        delta = spla.spsolve(J, F)

        if damping_cap is not None:
            step = np.clip(delta, -damping_cap, damping_cap)
        else:
            step = delta

        psi = psi - step

        if np.max(np.abs(delta)) < tol:
            break

    n = mat.ni * np.exp((psi - phin) / Vt)
    p = mat.ni * np.exp((phip - psi) / Vt)
    return psi, n, p, it + 1


def srh_recombination(mat: Material, n, p):
    ni = mat.ni
    return (n * p - ni ** 2) / (mat.tau_p * (n + ni) + mat.tau_n * (p + ni))


def solve_continuity_n(x, psi, mat: Material, R, n_bc0, n_bcL):
    """Linear tridiagonal solve for electron density n(x) given psi(x) and a
    (lagged) recombination source R(x), with Dirichlet BC at both contacts."""
    N = len(x)
    h = np.diff(x)
    u = psi / mat.Vt
    W = _control_volumes(x)

    Bp = bernoulli(u[1:] - u[:-1])   # B(u_{i+1} - u_i), for edge i (i -> i+1)
    Bm = bernoulli(u[:-1] - u[1:])   # B(u_i - u_{i+1})

    coef = Q * mat.Dn / h  # per-edge conductance-like coefficient, edge i between node i,i+1

    lower = np.zeros(N)
    diag = np.zeros(N)
    upper = np.zeros(N)
    rhs = np.zeros(N)

    diag[0] = 1.0
    rhs[0] = n_bc0
    diag[-1] = 1.0
    rhs[-1] = n_bcL

    # Jn_{i+1/2} = coef_i * (Bp_i * n_{i+1} - Bm_i * n_i); interior i=1..N-2
    c_m = coef[:-1]   # edge (i-1,i), for i=1..N-2
    c_p = coef[1:]    # edge (i,i+1)
    lower[1:-1] = c_m * Bm[:-1]
    diag[1:-1] = -c_p * Bm[1:] - c_m * Bp[:-1]
    upper[1:-1] = c_p * Bp[1:]
    rhs[1:-1] = Q * R[1:-1] * W[1:-1]

    A = sp.diags([lower[1:], diag, upper[:-1]], offsets=[-1, 0, 1], format="csc")
    n = spla.spsolve(A, rhs)
    return n


def solve_continuity_p(x, psi, mat: Material, R, p_bc0, p_bcL):
    """Linear tridiagonal solve for hole density p(x)."""
    N = len(x)
    h = np.diff(x)
    u = psi / mat.Vt
    W = _control_volumes(x)

    Bp = bernoulli(u[1:] - u[:-1])
    Bm = bernoulli(u[:-1] - u[1:])

    coef = Q * mat.Dp / h

    lower = np.zeros(N)
    diag = np.zeros(N)
    upper = np.zeros(N)
    rhs = np.zeros(N)

    diag[0] = 1.0
    rhs[0] = p_bc0
    diag[-1] = 1.0
    rhs[-1] = p_bcL

    # Jp_{i+1/2} = coef_i * (Bp_i * p_i - Bm_i * p_{i+1}); interior i=1..N-2
    c_m = coef[:-1]
    c_p = coef[1:]
    lower[1:-1] = -c_m * Bp[:-1]
    diag[1:-1] = c_p * Bp[1:] + c_m * Bm[:-1]
    upper[1:-1] = -c_p * Bm[1:]
    rhs[1:-1] = -Q * R[1:-1] * W[1:-1]

    A = sp.diags([lower[1:], diag, upper[:-1]], offsets=[-1, 0, 1], format="csc")
    p = spla.spsolve(A, rhs)
    return p


def edge_currents(x, psi, n, p, mat: Material):
    """Electron, hole, and total current density (A/cm^2) at each cell edge."""
    h = np.diff(x)
    u = psi / mat.Vt
    Bp = bernoulli(u[1:] - u[:-1])
    Bm = bernoulli(u[:-1] - u[1:])

    Jn = (Q * mat.Dn / h) * (n[1:] * Bp - n[:-1] * Bm)
    Jp = (Q * mat.Dp / h) * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp, Jn + Jp
