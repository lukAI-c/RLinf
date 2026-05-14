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
    Structured output of parse_lavira_json.

    Fields:
      actions     : list[int]      — GenArk action sequence (1+ elements when ok)
      bbox        : list[float]|None
      stop        : bool
      stair       : str|False      — "up"/"down"/False
      progress    : str            — progress_analysis text (for logging)
      reasoning   : str            — reasoning text
      raw_dir     : str|None       — original "navigate to ..." string
      ok          : bool           — True if JSON parsed and required fields valid
      err         : str|None       — error code if not ok
    """

    __slots__ = (
        "actions", "bbox", "stop", "stair",
        "progress", "reasoning", "raw_dir", "ok", "err",
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
