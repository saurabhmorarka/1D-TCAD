"""Interactive 2D structure/field viewer - the 2D sibling to core/plot.py,
built as new, separate code rather than a dim==2 branch inside it (see
ARCHITECTURE.md: 2D/3D visualization needs a genuinely different tool).
Phase 1 only renders the structure (regions + point-cloud mesh + boundary
condition tags); field rendering (psi/n/p/current density via tripcolor)
is added once the 2D solver exists.
"""
