"""Load the pinned LHX source tree without modifying its algorithm files."""

from __future__ import annotations

import collections
import collections.abc
import importlib
import sys
import types
from pathlib import Path

import numpy as np


SOURCE_COMMIT = "b3d6c35067ad731a16ec096b698ddfc2f5c91af6"
SOURCE_ROOT = Path(__file__).resolve().parent / "source"


def _install_compatibility_modules() -> None:
    # LHX targets Python 3.8, where these aliases still lived in collections.
    if not hasattr(collections, "Sequence"):
        collections.Sequence = collections.abc.Sequence

    if "habitat" not in sys.modules:
        habitat = types.ModuleType("habitat")
        habitat.Config = object
        sys.modules["habitat"] = habitat

    if "habitat_extensions" not in sys.modules:
        package = types.ModuleType("habitat_extensions")
        package.__path__ = [str(SOURCE_ROOT / "habitat_extensions")]
        sys.modules["habitat_extensions"] = package

    # Semantic_Mapping only uses threshold_poses from this module. Loading the
    # complete Habitat helper would require numpy-quaternion in the host policy
    # environment, although simulator pose ownership remains in the RPC server.
    pose_utils = types.ModuleType("habitat_extensions.pose_utils")

    def threshold_poses(coords, shape):
        coords[0] = min(max(0, coords[0]), shape[0] - 1)
        coords[1] = min(max(0, coords[1]), shape[1] - 1)
        return coords

    pose_utils.threshold_poses = threshold_poses
    sys.modules.setdefault("habitat_extensions.pose_utils", pose_utils)

    # RLinf owns user-facing map visualization. Stub the source visualization
    # import boundary so untouched map/FMM modules can load without Habitat-Lab.
    visualization = types.ModuleType("vlnce_baselines.utils.visualization")
    visualization.init_vis_image = lambda *args, **kwargs: np.zeros((1, 1, 3), dtype=np.uint8)
    visualization.add_class = lambda image, *args, **kwargs: image
    visualization.draw_line = lambda _a, _b, image, *args, **kwargs: image
    visualization.get_contour_points = lambda *args, **kwargs: np.empty((0, 2), dtype=np.int32)
    sys.modules.setdefault("vlnce_baselines.utils.visualization", visualization)

    fast_slic = types.ModuleType("fast_slic")

    class _UnavailableSlic:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("fast_slic is outside the vendored mapping/FMM path")

    fast_slic.Slic = _UnavailableSlic
    sys.modules.setdefault("fast_slic", fast_slic)

    pyinstrument = types.ModuleType("pyinstrument")
    pyinstrument.Profiler = object
    sys.modules.setdefault("pyinstrument", pyinstrument)


def _install_torch_dtype_compatibility() -> None:
    """Preserve the source math while fixing a Python/NumPy dtype edge.

    The pinned source returns float64 ``np.eye`` only for zero-angle rotations
    and float32 matrices otherwise. Newer PyTorch rejects the resulting
    float32/float64 matmul. LHX intends all mapper rotations to be float32, so
    normalize only this import boundary and leave the vendored source intact.
    """
    rotation_utils = importlib.import_module("vlnce_baselines.utils.rotation_utils")
    if getattr(rotation_utils.get_r_matrix, "_rlinf_dtype_compat", False):
        return
    source_get_r_matrix = rotation_utils.get_r_matrix

    def get_r_matrix_float32(*args, **kwargs):
        return np.asarray(source_get_r_matrix(*args, **kwargs), dtype=np.float32)

    get_r_matrix_float32._rlinf_dtype_compat = True
    rotation_utils.get_r_matrix = get_r_matrix_float32


def load_source_core():
    """Return untouched source classes/functions used by the RLinf adapter."""
    root = str(SOURCE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    loaded = sys.modules.get("vlnce_baselines")
    if loaded is None:
        loaded = types.ModuleType("vlnce_baselines")
        loaded.__path__ = [str(SOURCE_ROOT / "vlnce_baselines")]
        sys.modules["vlnce_baselines"] = loaded
    else:
        search_paths = [Path(path).resolve() for path in getattr(loaded, "__path__", [])]
        expected = (SOURCE_ROOT / "vlnce_baselines").resolve()
        if expected not in search_paths:
            raise RuntimeError(f"foreign vlnce_baselines already loaded from {search_paths}")
    _install_compatibility_modules()
    _install_torch_dtype_compatibility()
    mapping = importlib.import_module("vlnce_baselines.map.mapping")
    policy = importlib.import_module("vlnce_baselines.models.Policy")
    map_utils = importlib.import_module("vlnce_baselines.utils.map_utils")
    data_utils = importlib.import_module("vlnce_baselines.utils.data_utils")
    return types.SimpleNamespace(
        Semantic_Mapping=mapping.Semantic_Mapping,
        FusionMapPolicy=policy.FusionMapPolicy,
        FMMPlanner=policy.FMMPlanner,
        map_utils=map_utils,
        OrderedSet=data_utils.OrderedSet,
    )
