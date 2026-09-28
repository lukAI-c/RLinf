"""Legacy self-contained FMM compatibility entry point.

The production ``map_backend=source`` path directly loads LHX's vendored
``models/fmm_planner.py`` and ``models/Policy.py`` through ``LaviraSourceCore``.
Do not use this forwarding module to reproduce new LHX behavior. It is kept
only for the diagnostic ``sparse_ab`` backend and older Genesis callers.
"""

from ..lavira_map import FMMPlanner, OccupancyMap

__all__ = ["FMMPlanner", "OccupancyMap"]
