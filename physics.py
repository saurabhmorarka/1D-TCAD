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


def bernoulli_deriv(x: np.ndarray) -> np.ndarray:
    """dB/dx for the Bernoulli function B(x) = x/(exp(x)-1), needed for the
    analytic Newton Jacobian of the Scharfetter-Gummel flux."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    small = np.abs(x) < 1e-8
    out[small] = -0.5 + x[small] / 6.0  # Taylor series of B'(x) near 0

    big_pos = x > 40.0
    out[big_pos] = (1.0 - x[big_pos]) * np.exp(-x[big_pos])  # B(x)~x*e^-x here

    big_neg = x < -40.0
    out[big_neg] = -1.0  # B(x) ~ -x here

    mid = ~(small | big_pos | big_neg)
    xm = x[mid]
    denom = np.expm1(xm)
    out[mid] = (denom - xm * np.exp(xm)) / denom ** 2
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
                   tol=1e-10, max_iter=100, damping_cap=None,
                   eps=None, ni=None, n_frozen=None, p_frozen=None):
    """Newton solve of the nonlinear Poisson equation for psi(x), given fixed
    quasi-Fermi levels phin(x), phip(x) (both zero at equilibrium).

    Dirichlet BC: psi[0] and psi[-1] are held fixed at psi_guess[0], psi_guess[-1].

    eps: permittivity, either the (scalar) mat.eps default, or an array of
        length len(x)-1 giving a distinct value per mesh EDGE - needed for a
        layered structure (e.g. oxide/semiconductor in a MOS capacitor)
        where permittivity is discontinuous at an interface. Using a
        per-edge (not per-node) value is what makes the finite-volume flux
        automatically enforce D-field continuity across that interface.
    ni: intrinsic concentration, either the (scalar) mat.ni default, or an
        array of length len(x) giving a distinct value per node - e.g. 0 in
        an oxide region (no mobile carriers there at all: n=p=0 identically,
        independent of psi, which is exactly what ni=0 in n=ni*exp(...)
        gives, including a correctly-zeroed Jacobian contribution).
    n_frozen, p_frozen: if given (array of length len(x), NaN where not
        frozen), overrides that carrier's density at those nodes to a fixed
        value instead of the Boltzmann relation - used for the
        quasi-small-signal "high-frequency C-V" trick, where the inversion
        (minority-carrier) charge is held fixed while the majority carrier
        and potential respond to a small gate-voltage perturbation. Use
        n_frozen for a p-type substrate (electrons are the minority/
        inversion carrier - pMOS-cap) or p_frozen for an n-type substrate
        (holes are the minority carrier - nMOS-cap); at most one is normally
        given at a time, but both are accepted for generality.
    """
    N = len(x)
    h = np.diff(x)
    psi = psi_guess.copy()
    Vt = mat.Vt
    eps_edge = np.broadcast_to(mat.eps if eps is None else eps, N - 1)
    ni_arr = np.broadcast_to(mat.ni if ni is None else ni, N)
    n_frozen_mask = np.zeros(N, dtype=bool) if n_frozen is None else ~np.isnan(n_frozen)
    p_frozen_mask = np.zeros(N, dtype=bool) if p_frozen is None else ~np.isnan(p_frozen)

    hm = h[:-1]   # h_{i-1}, for interior i=1..N-2
    hp = h[1:]    # h_i
    cvol = (hm + hp) / 2.0
    lap_coeff_m = eps_edge[:-1] / hm / cvol
    lap_coeff_p = eps_edge[1:] / hp / cvol

    for it in range(max_iter):
        n = ni_arr * np.exp((psi - phin) / Vt)
        p = ni_arr * np.exp((phip - psi) / Vt)
        if n_frozen is not None:
            n = np.where(n_frozen_mask, n_frozen, n)
        if p_frozen is not None:
            p = np.where(p_frozen_mask, p_frozen, p)

        F = np.zeros(N)
        lower = np.zeros(N)
        diag = np.zeros(N)
        upper = np.zeros(N)

        dn_dpsi = np.where(n_frozen_mask, 0.0, n / Vt)    # frozen -> no psi-dependence
        dp_dpsi = np.where(p_frozen_mask, 0.0, -p / Vt)   # frozen -> no psi-dependence
        F[1:-1] = (lap_coeff_p * (psi[2:] - psi[1:-1]) - lap_coeff_m * (psi[1:-1] - psi[:-2])) \
            - Q * (n[1:-1] - p[1:-1] - Cdop[1:-1])
        lower[1:-1] = lap_coeff_m
        upper[1:-1] = lap_coeff_p
        diag[1:-1] = -(lap_coeff_m + lap_coeff_p) - Q * (dn_dpsi[1:-1] - dp_dpsi[1:-1])

        # Dirichlet rows: F=0 here always forces the boundary's Newton update
        # to exactly 0 (psi[0]/psi[-1] stay pinned at their initial-guess
        # value) *only* if the row's own diagonal actually stays the pivot
        # spsolve's partial pivoting selects for that column. diag=1 is fine
        # numerically when nearby coefficients are O(1)-ish, but a short
        # Debye length (fine mesh -> huge lap_coeff) combined with a bad
        # early Newton iterate (huge exp-driven charge term) can make the
        # neighboring interior row's coupling entry into this column
        # (lower[1] resp. upper[-2], both pure lap_coeff, unaffected by the
        # charge blowup) outweigh a bare 1.0, so the solver pivots onto that
        # row instead and silently leaks a nonzero, wrong value into what
        # must be an exact zero. Scaling the Dirichlet diagonal to the local
        # lap_coeff magnitude guarantees it stays the largest entry in its
        # column, so pivoting can never dilute the boundary condition -
        # caught via the poly-gate MOS-cap case (mesh.build_mos_grid with
        # Cdop_gate), where the poly's short Debye length makes this a real
        # (not just theoretical) failure mode.
        diag[0] = max(1.0, lap_coeff_m[0] if len(lap_coeff_m) else 1.0)
        diag[-1] = max(1.0, lap_coeff_p[-1] if len(lap_coeff_p) else 1.0)

        J = sp.diags([lower[1:], diag, upper[:-1]], offsets=[-1, 0, 1], format="csc")
        delta = spla.spsolve(J, F)

        if damping_cap is not None:
            step = np.clip(delta, -damping_cap, damping_cap)
        else:
            step = delta

        psi = psi - step

        if np.max(np.abs(delta)) < tol:
            break

    n = ni_arr * np.exp((psi - phin) / Vt)
    p = ni_arr * np.exp((phip - psi) / Vt)
    if n_frozen is not None:
        n = np.where(n_frozen_mask, n_frozen, n)
    if p_frozen is not None:
        p = np.where(p_frozen_mask, p_frozen, p)
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
