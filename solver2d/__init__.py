"""2D drift-diffusion solver: the point-cloud/box-FV generalization of
core/newton_solver_qf.py, plus post-processing (terminal current
extraction). Kept separate from mesh2d/ (which only builds the point-cloud
mesh/FV geometry, never solves anything) the same way core/mesh.py's mesh
builders are separate from core/newton_solver_qf.py's solver - mesh
construction and PDE solving are different concerns even when one
consumes the other.
"""
