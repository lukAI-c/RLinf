"""
Lavira-JSON action parser for QwenNavPolicy.

Parses model output text into:
  - action token sequence (mapped to GenArk's 4-way discrete action space)
  - bbox (raw, no validation against image size)
  - stop flag
  - parse status (for format_reward / debugging)

GenArk action codes:
  0 = stop
  1 = move forward 0.25m
  2 = turn left 30°
  3 = turn right 30°

Lavira action → GenArk action sequence (semantic macro):
  "navigate to forward" → [1]
  "navigate to left"    → [2, 1]            (turn left, then forward)
  "navigate to right"   → [3, 1]
  "navigate to behind"  → [2, 2, 2, 2, 2, 2, 1]   (six 30° turns = 180°, then forward)
  stop=True             → [0]   (overrides direction)
"""

from __future__ import annotations
import json
import re
from typing import Optional

from .canonical_targets import TARGET_REGIONS, normalize_target


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_STOP, ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT = 0, 1, 2, 3
# Sentinel returned to env when JSON parse fails (so env can distinguish
# "model chose stop" from "parse failed → fallback stop" for format_reward).
ACTION_PARSE_FAIL = 4
# Schema parsed successfully, but the waypoint geometry/action is not executable.
# GenArk treats this as format/schema OK for curriculum reward, then falls back
# to the same MOVE_FORWARD behavior as ACTION_PARSE_FAIL.
ACTION_SCHEMA_OK_PARSE_FAIL = 5
# RLinf adapter sentinel for LaViRA's "no action this loop" branch.  It is
# consumed by GenarkVecEnv without physics, elapsed-step, path, or nav reward.
ACTION_NOOP = 6
# Habitat-eval-only sentinel: execute LaViRA's physical 12x TURN_LEFT
# panorama acquisition, then return the collected F/L/B/R views.
ACTION_PANORAMA_SCAN = 7
# Sentinel base for parse_ok + has_valid_bbox_2d: sent = 10 + true_action.
# 10=stop+bbox, 11=fwd+bbox, 12=left+bbox, 13=right+bbox
# Env decodes: action >= 10 → has_bbox=True, true_action = action - 10.
ACTION_PARSE_OK_HAS_BBOX_BASE = 10
# Sentinel base for non-executable structured-output diagnostics:
# action = 20 + bitmask, decoded by GenArk as fallback MOVE_FORWARD.
ACTION_STRUCTURED_PARSE_FAIL_BASE = 20

# RL-Struct style reward bits carried through the action sentinel.
REWARD_JSON_VALID = 1 << 0
REWARD_REQUIRED_FIELDS = 1 << 1
REWARD_FIELD_FORMAT = 1 << 2
REWARD_GEOMETRY_VALID = 1 << 3
REWARD_LENGTH_OK = 1 << 4
REWARD_SCHEMA_BITS = REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS | REWARD_FIELD_FORMAT
REWARD_ALL_FORMAT_BITS = REWARD_SCHEMA_BITS | REWARD_GEOMETRY_VALID | REWARD_LENGTH_OK

DIRECTION_TO_ACTIONS: dict[str, list[int]] = {
    "navigate to forward": [ACTION_FORWARD],
    "navigate to left":    [ACTION_TURN_LEFT, ACTION_FORWARD],
    "navigate to right":   [ACTION_TURN_RIGHT, ACTION_FORWARD],
    "navigate to behind":  [ACTION_TURN_LEFT] * 6 + [ACTION_FORWARD],
}

VALID_DIRECTIONS = set(DIRECTION_TO_ACTIONS.keys())
VALID_STAIRS = {"up", "down", False, "false", "False"}

# JSON extraction regexes (priority: fenced → bare object)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.+?\})\s*```", re.S)
_BARE_JSON_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.S)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ParsedAction:
    """
    Structured output of parse_lavira_json / parse_lavira_merged_json.

    Core fields (all parsers):
      actions     : list[int]      — GenArk action sequence (1+ elements when ok)
      bbox        : list[float]|None — pixel bbox (default schema); None in merged schema
      stop        : bool
      stair       : str|False      — "up"/"down"/False
      progress    : str            — progress_analysis text (for logging)
      reasoning   : str            — reasoning text (default) / reasoning_action (merged)
      raw_dir     : str|None       — original "navigate to ..." string
      ok          : bool           — True if JSON parsed and required fields valid
      err         : str|None       — error code if not ok
      schema_ok   : bool           — True when JSON + required schema fields are valid
      geometry_ok : bool           — True when bbox/point geometry is executable
      reward_bits : int            — structured-output reward components

    Merged-schema fields (parse_lavira_merged_json only; defaults for default parser):
      action_type     : str             — "NAVIGATE" / "BACKTRACK" / "STOP" / "PARSE_FAIL"
      waypoint_id     : int|None        — backtrack target waypoint id
      bbox_2d         : list[float]|None — 0-1000 normalized bbox in chosen-direction view
      target          : str             — short bbox content description
      planning        : str             — planning field text
      reasoning_action: str             — reasoning_action field text
      reasoning_bbox  : str             — reasoning_bbox field text
      point_2d        : list[float]|None — 0-1000 normalized point in bbox view
      reasoning_plan_action: str        — lavira_waypoint field text
      reasoning_bbox_point: str         — lavira_waypoint field text
      backtrack_valid : bool            — policy-set validity flag for backtrack
    """

    __slots__ = (
        "actions", "bbox", "stop", "stair",
        "progress", "reasoning", "raw_dir", "ok", "err",
        "schema_ok", "geometry_ok", "reward_bits",
        # merged-schema extended fields
        "action_type", "waypoint_id", "bbox_2d",
        "target", "planning", "reasoning_action", "reasoning_bbox",
        "point_2d", "reasoning_plan_action", "reasoning_bbox_point",
        "backtrack_valid", "raw_target", "target_region",
    )

    def __init__(
        self,
        actions: list[int],
        bbox: Optional[list[float]] = None,
        stop: bool = False,
        stair=False,
        progress: str = "",
        reasoning: str = "",
        raw_dir: Optional[str] = None,
        ok: bool = True,
        err: Optional[str] = None,
        schema_ok: Optional[bool] = None,
        geometry_ok: Optional[bool] = None,
        reward_bits: int = 0,
        # merged-schema extended fields
        action_type: str = "",
        waypoint_id: Optional[int] = None,
        bbox_2d: Optional[list[float]] = None,
        target: str = "",
        planning: str = "",
        reasoning_action: str = "",
        reasoning_bbox: str = "",
        point_2d: Optional[list[float]] = None,
        reasoning_plan_action: str = "",
        reasoning_bbox_point: str = "",
        backtrack_valid: bool = False,
        raw_target: Optional[str] = None,
        target_region: str = "any",
    ):
        self.actions = actions
        self.bbox = bbox
        self.stop = stop
        self.stair = stair
        self.progress = progress
        self.reasoning = reasoning
        self.raw_dir = raw_dir
        self.ok = ok
        self.err = err
        self.schema_ok = ok if schema_ok is None else bool(schema_ok)
        self.geometry_ok = ok if geometry_ok is None else bool(geometry_ok)
        if reward_bits:
            self.reward_bits = int(reward_bits)
        elif self.geometry_ok:
            self.reward_bits = REWARD_SCHEMA_BITS | REWARD_GEOMETRY_VALID
        elif self.schema_ok:
            self.reward_bits = REWARD_SCHEMA_BITS
        else:
            self.reward_bits = 0
        self.action_type = action_type
        self.waypoint_id = waypoint_id
        self.bbox_2d = bbox_2d
        self.target = target
        self.planning = planning
        self.reasoning_action = reasoning_action
        self.reasoning_bbox = reasoning_bbox
        self.point_2d = point_2d
        self.reasoning_plan_action = reasoning_plan_action
        self.reasoning_bbox_point = reasoning_bbox_point
        self.backtrack_valid = backtrack_valid
        self.raw_target = raw_target if raw_target is not None else target
        self.target_region = str(target_region or "any")


def parse_lavira_json(text: str) -> ParsedAction:
    """
    Parse a single model response into a ParsedAction.

    On any failure (no JSON / invalid JSON / missing field / unknown action),
    returns a ParsedAction with `ok=False`, `err=<code>`, `actions=[STOP]`
    (safe fallback that ends the episode harmlessly).
    """
    if not text:
        return _fallback("empty_text")

    json_str = _extract_json_string(text)
    if json_str is None:
        return _fallback("no_json_found")

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        return _fallback(f"json_invalid:{e.msg}")

    if not isinstance(obj, dict):
        return _fallback("not_object")

    # Required: action and stop
    raw_dir = obj.get("action")
    stop = bool(obj.get("stop", False))

    # If stop=True, we don't care about action validity
    if stop:
        return ParsedAction(
            actions=[ACTION_STOP],
            bbox=_clean_bbox(obj.get("bbox")),
            stop=True,
            stair=_clean_stair(obj.get("stair", False)),
            progress=str(obj.get("progress_analysis", ""))[:512],
            reasoning=str(obj.get("reasoning", ""))[:512],
            raw_dir=raw_dir if isinstance(raw_dir, str) else None,
            ok=True,
            err=None,
        )

    # stop=False: must have valid direction
    if not isinstance(raw_dir, str) or raw_dir not in VALID_DIRECTIONS:
        return _fallback(f"unknown_direction:{raw_dir}")

    return ParsedAction(
        actions=list(DIRECTION_TO_ACTIONS[raw_dir]),
        bbox=_clean_bbox(obj.get("bbox")),
        stop=False,
        stair=_clean_stair(obj.get("stair", False)),
        progress=str(obj.get("progress_analysis", ""))[:512],
        reasoning=str(obj.get("reasoning", ""))[:512],
        raw_dir=raw_dir,
        ok=True,
        err=None,
    )


# Regex to extract waypoint id from "backtrack to <N>"
_BACKTRACK_RE = re.compile(r"^backtrack\s+to\s+(\d+)$", re.I)


def parse_lavira_merged_json(text: str) -> ParsedAction:
    """
    Parse a model response that uses the LaViRA merged LA+VA JSON schema.

    Expected schema (9 fields in order):
      progress_analysis, planning, reasoning_action, action,
      stop, stair, reasoning_bbox, bbox_2d, target

    Action handling:
      - "navigate to {forward|left|right|behind}" → NAVIGATE, 30° macros (Phase A)
      - "backtrack to <N>"                        → BACKTRACK, degraded to forward (Phase A)
      - stop=true                                 → STOP, actions=[ACTION_STOP]
      - anything else                             → fallback PARSE_FAIL
    """
    if not text:
        return _fallback_merged("empty_text")

    json_str = _extract_json_string(text)
    if json_str is None:
        return _fallback_merged("no_json_found")

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        return _fallback_merged(f"json_invalid:{e.msg}")

    if not isinstance(obj, dict):
        return _fallback_merged("not_object")

    raw_action = obj.get("action")
    stop = bool(obj.get("stop", False))

    # Shared text fields (all truncated for logging)
    progress = str(obj.get("progress_analysis", ""))[:512]
    planning = str(obj.get("planning", ""))[:512]
    reasoning_action = str(obj.get("reasoning_action", ""))[:512]
    reasoning_bbox = str(obj.get("reasoning_bbox", ""))[:256]
    target = str(obj.get("target", ""))[:256]
    stair = _clean_stair(obj.get("stair", False))
    bbox_2d = _clean_bbox(obj.get("bbox_2d"))

    if stop:
        return ParsedAction(
            actions=[ACTION_STOP],
            bbox=None,
            stop=True,
            stair=stair,
            progress=progress,
            reasoning=reasoning_action,
            raw_dir=raw_action if isinstance(raw_action, str) else None,
            ok=True,
            err=None,
            schema_ok=True,
            geometry_ok=True,
            action_type="STOP",
            waypoint_id=None,
            bbox_2d=bbox_2d,
            target=target,
            planning=planning,
            reasoning_action=reasoning_action,
            reasoning_bbox=reasoning_bbox,
        )

    if not isinstance(raw_action, str):
        return _fallback_merged(f"unknown_direction:{raw_action}")

    # Check for backtrack (Phase A: degrade to forward)
    bm = _BACKTRACK_RE.match(raw_action.strip())
    if bm:
        waypoint_id = int(bm.group(1))
        return ParsedAction(
            actions=list(DIRECTION_TO_ACTIONS["navigate to forward"]),
            bbox=None,
            stop=False,
            stair=stair,
            progress=progress,
            reasoning=f"[backtrack→forward wp={waypoint_id}] " + reasoning_action,
            raw_dir="navigate to forward",
            ok=True,
            err=None,
            action_type="BACKTRACK",
            waypoint_id=waypoint_id,
            bbox_2d=None,  # bbox unused for backtrack
            target="",
            planning=planning,
            reasoning_action=reasoning_action,
            reasoning_bbox=reasoning_bbox,
        )

    # Regular navigate action
    if raw_action not in VALID_DIRECTIONS:
        return _fallback_merged(f"unknown_direction:{raw_action}")

    return ParsedAction(
        actions=list(DIRECTION_TO_ACTIONS[raw_action]),
        bbox=None,
        stop=False,
        stair=stair,
        progress=progress,
        reasoning=reasoning_action,
        raw_dir=raw_action,
        ok=True,
        err=None,
        action_type="NAVIGATE",
        waypoint_id=None,
        bbox_2d=bbox_2d,
        target=target,
        planning=planning,
        reasoning_action=reasoning_action,
        reasoning_bbox=reasoning_bbox,
    )


_WAYPOINT_NAV_DIRECTIONS = {
    "navigate to forward",
    "navigate to left",
    "navigate to right",
    "navigate to behind",
}


def parse_lavira_waypoint_json(text: str) -> ParsedAction:
    """
    Parse the LaViRA waypoint prompt schema.

    Expected schema (7 fields):
      progress_analysis, reasoning_plan_action, planning, action, stop, stair,
      target
    """
    if not text:
        return _fallback_waypoint("empty_text")

    # The waypoint template is intentionally a strict contract: one bare JSON
    # object, no markdown fence or commentary.  Do not reuse the permissive
    # extractor used by legacy prompt styles here, otherwise non-template
    # completions silently enter the controller and training batch.
    json_str = text.strip()
    if not (json_str.startswith("{") and json_str.endswith("}")):
        return _fallback_waypoint("no_json_found")

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        return _fallback_waypoint(f"json_invalid:{e.msg}")

    if not isinstance(obj, dict):
        return _fallback_waypoint("not_object", reward_bits=REWARD_JSON_VALID)

    required = (
        "progress_analysis", "reasoning_plan_action", "planning", "action",
        "stop", "stair", "target",
    )
    missing = [k for k in required if k not in obj]
    if missing:
        return _fallback_waypoint(
            "missing_fields:" + ",".join(missing),
            reward_bits=REWARD_JSON_VALID,
        )
    actual_fields = tuple(obj.keys())
    if set(actual_fields) != set(required):
        unexpected = [k for k in actual_fields if k not in required]
        return _fallback_waypoint(
            "unexpected_fields:" + ",".join(unexpected),
            reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
        )
    if actual_fields != required:
        return _fallback_waypoint(
            "field_order",
            reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
        )

    # Keep the runtime schema as strict as the prompt.  Coercing e.g.
    # target=false into the string "False" makes malformed outputs look valid
    # and sends unusable phrases into GroundingDINO/SAM.
    text_fields = (
        "progress_analysis",
        "reasoning_plan_action",
        "planning",
        "action",
        "target",
    )
    for field_name in text_fields:
        if not isinstance(obj[field_name], str):
            return _fallback_waypoint(
                f"invalid_type:{field_name}",
                reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
            )
    if not isinstance(obj["stop"], bool):
        return _fallback_waypoint(
            "invalid_type:stop",
            reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
        )
    if obj["stair"] is not False and obj["stair"] not in ("up", "down"):
        return _fallback_waypoint(
            "invalid_type:stair",
            reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
        )

    raw_action = obj["action"]
    stop = obj["stop"]
    progress = obj["progress_analysis"][:512]
    reasoning_plan_action = obj["reasoning_plan_action"][:512]
    planning = obj["planning"][:512]
    reasoning_bbox_point = ""
    target = obj["target"][:256]
    stair = obj["stair"]
    # Geometry is produced by GroundedSAM from ``target`` downstream.  The
    # language model contract deliberately contains no bbox/point fields.
    bbox_2d = None
    point_2d = None

    if stop:
        return ParsedAction(
            actions=[ACTION_STOP],
            bbox=None,
            stop=True,
            stair=stair,
            progress=progress,
            reasoning=reasoning_plan_action,
            raw_dir=raw_action if isinstance(raw_action, str) else None,
            ok=True,
            err=None,
            schema_ok=True,
            geometry_ok=True,
            reward_bits=REWARD_SCHEMA_BITS,
            action_type="STOP",
            bbox_2d=bbox_2d,
            point_2d=point_2d,
            target=target,
            planning=planning,
            reasoning_action=reasoning_plan_action,
            reasoning_bbox=reasoning_bbox_point,
            reasoning_plan_action=reasoning_plan_action,
            reasoning_bbox_point=reasoning_bbox_point,
        )

    if not isinstance(raw_action, str):
        return _fallback_waypoint(
            f"unknown_direction:{raw_action}",
            schema_ok=True,
            reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
        )

    raw_action = raw_action.strip()
    bm = _BACKTRACK_RE.match(raw_action)
    if bm:
        # Backtracking is fully specified by the waypoint id; no visual
        # geometry belongs in the model output contract.
        waypoint_id = int(bm.group(1))
        return ParsedAction(
            actions=[],
            bbox=None,
            stop=False,
            stair=stair,
            progress=progress,
            reasoning=reasoning_plan_action,
            raw_dir=raw_action,
            ok=True,
            err=None,
            schema_ok=True,
            geometry_ok=True,
            reward_bits=REWARD_SCHEMA_BITS,
            action_type="BACKTRACK",
            waypoint_id=waypoint_id,
            bbox_2d=bbox_2d,
            point_2d=point_2d,
            target=target,
            planning=planning,
            reasoning_action=reasoning_plan_action,
            reasoning_bbox=reasoning_bbox_point,
            reasoning_plan_action=reasoning_plan_action,
            reasoning_bbox_point=reasoning_bbox_point,
        )

    raw_action = raw_action.strip()
    if raw_action not in _WAYPOINT_NAV_DIRECTIONS:
        return _fallback_waypoint(
            f"unknown_direction:{raw_action}",
            schema_ok=True,
            reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
        )
    if not target.strip():
        return _fallback_waypoint(
            "empty_navigate_target",
            reward_bits=REWARD_SCHEMA_BITS,
        )
    return ParsedAction(
        actions=list(DIRECTION_TO_ACTIONS[raw_action]),
        bbox=None,
        stop=False,
        stair=stair,
        progress=progress,
        reasoning=reasoning_plan_action,
        raw_dir=raw_action,
        ok=True,
        err=None,
        schema_ok=True,
        geometry_ok=False,
        reward_bits=REWARD_SCHEMA_BITS,
        action_type="NAVIGATE",
        waypoint_id=None,
        bbox_2d=bbox_2d,
        point_2d=point_2d,
        target=target,
        planning=planning,
        reasoning_action=reasoning_plan_action,
        reasoning_bbox=reasoning_bbox_point,
        reasoning_plan_action=reasoning_plan_action,
        reasoning_bbox_point=reasoning_bbox_point,
    )


def parse_lavira_canonical_waypoint_json(text: str) -> ParsedAction:
    """Parse the compact canonical target contract used by ``canonical_v1``.

    The navigation/reasoning fields intentionally remain identical to the
    source waypoint contract.  Only the free-form target is replaced by a
    canonical class and an image-third selector.
    """
    required = (
        "progress_analysis", "reasoning_plan_action", "planning", "action",
        "stop", "stair", "target_class", "target_region",
    )
    if not text:
        return _fallback_waypoint("empty_text")
    raw = text.strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        return _fallback_waypoint("no_json_found")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _fallback_waypoint(f"json_invalid:{exc.msg}")
    if not isinstance(obj, dict):
        return _fallback_waypoint("not_object", reward_bits=REWARD_JSON_VALID)
    missing = [key for key in required if key not in obj]
    if missing:
        return _fallback_waypoint(
            "missing_fields:" + ",".join(missing), reward_bits=REWARD_JSON_VALID
        )
    if tuple(obj.keys()) != required:
        return _fallback_waypoint(
            "field_order" if set(obj) == set(required)
            else "unexpected_fields:" + ",".join(k for k in obj if k not in required),
            reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
        )
    for key in ("progress_analysis", "reasoning_plan_action", "planning", "action",
                "target_class", "target_region"):
        if not isinstance(obj[key], str):
            return _fallback_waypoint(
                f"invalid_type:{key}",
                reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
            )
    if not isinstance(obj["stop"], bool):
        return _fallback_waypoint(
            "invalid_type:stop",
            reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
        )
    if obj["stair"] is not False and obj["stair"] not in ("up", "down"):
        return _fallback_waypoint(
            "invalid_type:stair",
            reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
        )

    progress = obj["progress_analysis"][:512]
    reasoning = obj["reasoning_plan_action"][:512]
    planning = obj["planning"][:512]
    action = obj["action"].strip()
    stop = obj["stop"]
    stair = obj["stair"]
    raw_target = obj["target_class"]
    target_region = obj["target_region"].strip().lower()
    if target_region not in TARGET_REGIONS:
        return _fallback_waypoint(
            f"invalid_target_region:{target_region}",
            reward_bits=REWARD_SCHEMA_BITS,
        )

    if stop:
        if raw_target.strip() or target_region != "any":
            return _fallback_waypoint(
                "stop_target_not_empty", reward_bits=REWARD_SCHEMA_BITS
            )
        return ParsedAction(
            actions=[ACTION_STOP], stop=True, stair=stair, progress=progress,
            reasoning=reasoning, raw_dir=action if action else None, ok=True,
            schema_ok=True, geometry_ok=True, reward_bits=REWARD_SCHEMA_BITS,
            action_type="STOP", target="", raw_target=raw_target,
            target_region=target_region, planning=planning,
            reasoning_action=reasoning, reasoning_plan_action=reasoning,
        )

    backtrack = _BACKTRACK_RE.match(action)
    if backtrack:
        if raw_target.strip() or target_region != "any":
            return _fallback_waypoint(
                "backtrack_target_not_empty", reward_bits=REWARD_SCHEMA_BITS
            )
        return ParsedAction(
            actions=[], stop=False, stair=stair, progress=progress,
            reasoning=reasoning, raw_dir=action, ok=True, schema_ok=True,
            geometry_ok=True, reward_bits=REWARD_SCHEMA_BITS,
            action_type="BACKTRACK", waypoint_id=int(backtrack.group(1)),
            target="", raw_target=raw_target, target_region=target_region,
            planning=planning, reasoning_action=reasoning,
            reasoning_plan_action=reasoning,
        )

    if action not in _WAYPOINT_NAV_DIRECTIONS:
        return _fallback_waypoint(
            f"unknown_direction:{action}", schema_ok=True,
            reward_bits=REWARD_JSON_VALID | REWARD_REQUIRED_FIELDS,
        )
    canonical, status = normalize_target(raw_target)
    if canonical is None:
        return _fallback_waypoint(
            f"unknown_target:{raw_target}", schema_ok=True,
            reward_bits=REWARD_SCHEMA_BITS,
        )
    return ParsedAction(
        actions=list(DIRECTION_TO_ACTIONS[action]), stop=False, stair=stair,
        progress=progress, reasoning=reasoning, raw_dir=action, ok=True,
        schema_ok=True, geometry_ok=False, reward_bits=REWARD_SCHEMA_BITS,
        action_type="NAVIGATE", target=canonical, raw_target=raw_target,
        target_region=target_region, planning=planning,
        reasoning_action=reasoning, reasoning_plan_action=reasoning,
    )


def _fallback_merged(err: str) -> ParsedAction:
    return ParsedAction(
        actions=[ACTION_STOP],
        bbox=None,
        stop=True,
        stair=False,
        progress="",
        reasoning="",
        raw_dir=None,
        ok=False,
        err=err,
        action_type="PARSE_FAIL",
    )


def _fallback_waypoint(
    err: str,
    schema_ok: bool = False,
    reward_bits: int = 0,
) -> ParsedAction:
    return ParsedAction(
        actions=[ACTION_PARSE_FAIL],
        bbox=None,
        stop=False,
        stair=False,
        progress="",
        reasoning="",
        raw_dir=None,
        ok=False,
        err=err,
        schema_ok=schema_ok,
        geometry_ok=False,
        reward_bits=reward_bits,
        action_type="PARSE_FAIL",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_string(text: str) -> Optional[str]:
    """Try fenced ```json ... ``` first, then bare {...}."""
    m = _FENCED_JSON_RE.search(text)
    if m:
        return m.group(1)
    m = _BARE_JSON_RE.search(text)
    if m:
        return m.group(0)
    return None


def _clean_bbox(v) -> Optional[list[float]]:
    """Validate bbox is a 4-element numeric list. Returns None otherwise."""
    if v is None:
        return None
    if isinstance(v, list) and len(v) == 4:
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None
    return None


def _clean_point(v) -> Optional[list[float]]:
    """Validate point is a 2-element numeric list. Returns None otherwise."""
    if v is None:
        return None
    if isinstance(v, list) and len(v) == 2:
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None
    return None


def _bbox_in_range(b: list[float]) -> bool:
    if len(b) != 4:
        return False
    x1, y1, x2, y2 = b
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def _point_in_range(p: list[float]) -> bool:
    if len(p) != 2:
        return False
    x, y = p
    return 0 <= x <= 1000 and 0 <= y <= 1000


def _point_inside_bbox(p: list[float], b: list[float]) -> bool:
    x, y = p
    x1, y1, x2, y2 = b
    return x1 <= x <= x2 and y1 <= y <= y2


def _clean_stair(v):
    if v in (True, "true", "True"):
        # 'true' without direction is meaningless — treat as no stair
        return False
    if v in ("up", "down"):
        return v
    return False


def _fallback(err: str) -> ParsedAction:
    return ParsedAction(
        actions=[ACTION_STOP],
        bbox=None,
        stop=True,
        stair=False,
        progress="",
        reasoning="",
        raw_dir=None,
        ok=False,
        err=err,
    )


# ---------------------------------------------------------------------------
# Stop double-check parser (Lavira-style second LLM call)
# ---------------------------------------------------------------------------

class ParsedStopCheck:
    """Result of the stop double-check LLM call."""
    __slots__ = ("analysis", "decision", "ok")

    def __init__(self, analysis: str, decision: str, ok: bool):
        self.analysis = analysis
        self.decision = decision   # "STOP" or "CONTINUE"
        self.ok = ok


def parse_stop_check_json(text: str) -> ParsedStopCheck:
    """
    Parse the stop-check model response.
    Expected: {"analysis": "...", "decision": "STOP" or "CONTINUE"}
    Falls back to CONTINUE on any parse failure (conservative: don't stop if unsure).
    """
    if not text:
        return ParsedStopCheck(analysis="", decision="CONTINUE", ok=False)

    json_str = _extract_json_string(text)
    if json_str is None:
        return ParsedStopCheck(analysis="", decision="CONTINUE", ok=False)

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return ParsedStopCheck(analysis="", decision="CONTINUE", ok=False)

    if not isinstance(obj, dict):
        return ParsedStopCheck(analysis="", decision="CONTINUE", ok=False)

    decision = str(obj.get("decision", "CONTINUE")).strip().upper()
    if decision not in ("STOP", "CONTINUE"):
        decision = "CONTINUE"

    return ParsedStopCheck(
        analysis=str(obj.get("analysis", ""))[:512],
        decision=decision,
        ok=True,
    )
