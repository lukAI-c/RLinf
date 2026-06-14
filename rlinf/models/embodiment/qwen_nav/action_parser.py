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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_STOP, ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT = 0, 1, 2, 3
# Sentinel returned to env when JSON parse fails (so env can distinguish
# "model chose stop" from "parse failed → fallback stop" for format_reward).
ACTION_PARSE_FAIL = 4
# Sentinel base for parse_ok + has_valid_bbox_2d: sent = 10 + true_action.
# 10=stop+bbox, 11=fwd+bbox, 12=left+bbox, 13=right+bbox
# Env decodes: action >= 10 → has_bbox=True, true_action = action - 10.
ACTION_PARSE_OK_HAS_BBOX_BASE = 10

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

    Merged-schema fields (parse_lavira_merged_json only; defaults for default parser):
      action_type     : str             — "NAVIGATE" / "BACKTRACK" / "STOP" / "PARSE_FAIL"
      waypoint_id     : int|None        — backtrack target waypoint id
      bbox_2d         : list[float]|None — 0-1000 normalized bbox in chosen-direction view
      target          : str             — short bbox content description
      planning        : str             — planning field text
      reasoning_action: str             — reasoning_action field text
      reasoning_bbox  : str             — reasoning_bbox field text
    """

    __slots__ = (
        "actions", "bbox", "stop", "stair",
        "progress", "reasoning", "raw_dir", "ok", "err",
        # merged-schema extended fields
        "action_type", "waypoint_id", "bbox_2d",
        "target", "planning", "reasoning_action", "reasoning_bbox",
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
        # merged-schema extended fields
        action_type: str = "",
        waypoint_id: Optional[int] = None,
        bbox_2d: Optional[list[float]] = None,
        target: str = "",
        planning: str = "",
        reasoning_action: str = "",
        reasoning_bbox: str = "",
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
        self.action_type = action_type
        self.waypoint_id = waypoint_id
        self.bbox_2d = bbox_2d
        self.target = target
        self.planning = planning
        self.reasoning_action = reasoning_action
        self.reasoning_bbox = reasoning_bbox


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
