"""Deterministic target vocabulary and execution-side grounding adapter."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


DINO_TARGET_VOCAB = (
    "archway", "doorway", "door", "hallway", "corridor", "stairs", "rug",
    "chair", "couch", "table", "bed", "toilet", "sink", "refrigerator",
    "oven", "television", "plant", "cabinet", "counter", "island",
    "treadmill", "exercise equipment", "pillar",
)

TARGET_REGIONS = ("left", "center", "right", "any")
# Only bounded, locally groundable openings use portal waypoint geometry.
# ``hallway`` and ``corridor`` are scene regions; GroundingDINO commonly returns
# a near-full-frame box for them, which is not a meaningful portal boundary.
PORTAL_CLASSES = frozenset({"archway", "doorway", "door"})
AREA_CLASSES = frozenset({"hallway", "corridor", "floor", "path"})
_AREA_TOKEN_MAP = {
    "hallway": "hallway",
    "hall": "hallway",
    "corridor": "corridor",
    "passage": "path",
    "walkway": "path",
    "path": "path",
    "floor": "floor",
}
_PORTAL_TOKEN_MAP = {
    "archway": "archway",
    "archways": "archway",
    "doorway": "doorway",
    "doorways": "doorway",
    "door": "door",
    "doors": "door",
    "opening": "doorway",
    "openings": "doorway",
}

# Exact aliases observed in or added from GroundedSAM audit review. No fuzzy
# matching is intentional; unknown phrases retain the original DINO fallback.
TARGET_ALIASES = {
    "archway exit": "archway",
    "archway opening": "archway",
    "archway entrance": "archway",
    "archway to hallway": "archway",
    "arched doorway": "archway",
    "stone archway": "archway",
    "doorway exit": "doorway",
    "doorway opening": "doorway",
    "doorway entrance": "doorway",
    "doorway to hallway": "doorway",
    "doorway to bedroom": "doorway",
    "bathroom exit doorway": "doorway",
    "open doorway": "doorway",
    "open doorway to hallway": "doorway",
    "open doorway leading to hallway": "doorway",
    "hallway doorway": "doorway",
    "doorway in behind view": "doorway",
    # Episode-609 fixed-input DINO A/B: ``hallway`` produces a 99% scene box,
    # while ``archway`` localizes the visible bounded passage. Keep this exact
    # alias narrow; not every generic entrance is arched.
    "hallway entrance": "archway",
    "hallway opening": "doorway",
    "corridor entrance": "doorway",
    "hallway floor": "hallway",
    "hallway passage": "hallway",
    "hallway entrance floor": "hallway",
    "corridor floor": "corridor",
    "floor passage": "path",
    "floor passage between pillars": "path",
    "floor path": "path",
    "white rug": "rug",
    "white floor rug": "rug",
    "white rug on floor": "rug",
    "white rug in hallway": "rug",
    "hallway floor and rug": "rug",
    "tv": "television",
}

_OBJECT_CLASSES = tuple(
    label for label in DINO_TARGET_VOCAB
    if label not in PORTAL_CLASSES and label not in AREA_CLASSES
)


@dataclass(frozen=True)
class GroundingTargetMapping:
    """Execution-only mapping; it never changes the model's parsed target."""

    raw_target: str
    normalized_target: str
    canonical_target: Optional[str]
    dino_query: str
    geometry_kind: str
    status: str


def normalize_target(value: object) -> tuple[Optional[str], str]:
    """Return ``(canonical, status)`` using exact matching only."""
    if not isinstance(value, str):
        return None, "invalid_type"
    normalized = " ".join(value.strip().lower().split())
    if normalized in DINO_TARGET_VOCAB:
        return normalized, "canonical"
    if normalized in TARGET_ALIASES:
        return TARGET_ALIASES[normalized], "alias"
    if not normalized:
        return None, "empty"
    return None, "unknown"


def map_target_for_grounding(value: object) -> GroundingTargetMapping:
    """Map a free-form target to a stable DINO query and geometry route.

    Matching is deliberately deterministic: exact reviewed aliases first,
    then visible portal nouns, concrete object nouns, and finally area nouns.
    Unknown phrases remain usable by the original DINO/LHX fallback path.
    """
    if not isinstance(value, str):
        return GroundingTargetMapping("", "", None, "", "unknown", "invalid_type")

    raw_target = value.strip()
    normalized = " ".join(re.findall(r"[a-z0-9]+", raw_target.lower()))
    if not normalized:
        return GroundingTargetMapping(raw_target, "", None, "", "unknown", "empty")

    canonical, status = normalize_target(normalized)
    if canonical is not None:
        geometry = (
            "portal" if canonical in PORTAL_CLASSES
            else "area" if canonical in AREA_CLASSES
            else "object"
        )
        return GroundingTargetMapping(
            raw_target, normalized, canonical, canonical, geometry, status
        )

    tokens = normalized.split()
    token_set = set(tokens)

    # A bounded physical opening is more specific than its destination, e.g.
    # "archway to hallway". Prefer the most visually specific portal noun.
    for token in ("archway", "archways", "doorway", "doorways",
                  "door", "doors", "opening", "openings"):
        if token in token_set:
            canonical = _PORTAL_TOKEN_MAP[token]
            return GroundingTargetMapping(
                raw_target, normalized, canonical, canonical, "portal", "head_noun"
            )

    # Objects take precedence over area words so "white floor rug" remains a
    # rug even when a previously unseen modifier prevents exact alias matching.
    padded = f" {normalized} "
    for label in sorted(_OBJECT_CLASSES, key=lambda item: (-len(item.split()), item)):
        if f" {label} " in padded:
            return GroundingTargetMapping(
                raw_target, normalized, label, label, "object", "head_noun"
            )

    for token in tokens:
        if token in _AREA_TOKEN_MAP:
            canonical = _AREA_TOKEN_MAP[token]
            return GroundingTargetMapping(
                raw_target, normalized, canonical, canonical, "area", "area_noun"
            )

    # Do not reject or invent a class for an unknown phrase. Sending the
    # normalized phrase preserves the pre-adapter DINO -> LHX behaviour.
    return GroundingTargetMapping(
        raw_target, normalized, None, normalized, "unknown", "unknown"
    )


def target_geometry_kind(value: object) -> str:
    """Route a free-form target without requiring a different model schema.

    Concrete portals take precedence over destination words, so phrases such
    as doorway-to-hallway keep their source-style portal grounding. Area
    geometry is used only when the phrase does not identify a bounded portal
    or another canonical object.
    """
    return map_target_for_grounding(value).geometry_kind


def region_for_box(box_2d: list[float]) -> str:
    """Return the image-third containing a normalized 0-1000 bbox center."""
    center_x = (float(box_2d[0]) + float(box_2d[2])) * 0.5
    if center_x < 333.333:
        return "left"
    if center_x < 666.667:
        return "center"
    return "right"
