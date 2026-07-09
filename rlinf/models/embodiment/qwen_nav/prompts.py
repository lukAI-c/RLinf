"""
Lavira-style JSON prompt builder for GenArk navigation (single-turn rebuild mode).

Each step:
  - System prompt: role + action space + JSON schema
  - User content (rebuilt every step):
      [system text]
      Instruction: "..."
      Navigation History: <hist_img_0> ... <hist_img_K>
      Current 4-directional views: <front> <left> <right> <behind>
      Task description + JSON output instruction

Output JSON schema:
  {
      "progress_analysis": "<...>",
      "reasoning": "<...>",
      "action": "navigate to forward|left|right|behind",
      "bbox": [x1, y1, x2, y2] or null,
      "stop": true|false,
      "stair": "up"|"down"|false
  }
"""

from __future__ import annotations
from typing import Optional, Any


# ---------------------------------------------------------------------------
# System prompt (lavira-style, single-turn ready)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """/no_think
You are a navigation agent in an indoor environment. You receive 4-directional views (front, left, right, behind) of the current location and a sequence of historical observation images. Based on the text instruction, history, and current views, you decide the next navigation action.

Action Space (each action corresponds to the labeled view image):
  - "navigate to forward" → [Front view]  go straight ahead
  - "navigate to left"    → [Left view]   turn left and advance
  - "navigate to right"   → [Right view]  turn right and advance
  - "navigate to behind"  → [Behind view] turn around and advance

You MUST respond with a valid JSON object with ALL of these fields in order:
{
    "progress_analysis": "<brief analysis: where am I now, which part of the instruction is done, what remains>",
    "reasoning": "<why this action is the best next step given current views and progress>",
    "action": "navigate to forward|navigate to left|navigate to right|navigate to behind",
    "bbox": [x1, y1, x2, y2] or null,
    "stop": true|false,
    "stair": "up"|"down"|false
}

Guidelines:
- "progress_analysis": 1-2 sentences summarizing current position and task progress.
- "reasoning": 1-2 sentences explaining your directional choice based on visible cues.
- "action": choose one from the four options above.
- "bbox": pixel bounding box of the navigation target in the chosen direction's view. Use null if no clear target.
- "stop": set true ONLY when you are within arm's reach of the final destination AND you have navigated significantly. Do NOT stop in the first few steps — explore first.
- "stair": "up" or "down" if next action involves stairs; otherwise false.
- Do NOT try to open doors.
- Output ONLY the JSON object — no markdown fence, no extra text.
"""


# ---------------------------------------------------------------------------
# User-content templates (per-step rebuild)
# ---------------------------------------------------------------------------

USER_HEADER = """Navigation Task: "{instruction}"

Navigation History (chronological, oldest → newest):"""

USER_HISTORY_LABEL = "Step {step_idx}: <image>"

USER_NO_HISTORY = "(no history yet — this is the first step)"

USER_CURRENT_HEADER = "\n\nCurrent 4-directional views (relative to your current facing direction):"

USER_CURRENT_VIEWS = """[navigate to forward]  Front view:  <image>
[navigate to left]     Left view:   <image>
[navigate to right]    Right view:  <image>
[navigate to behind]   Behind view: <image>"""

USER_TASK = """\n\nAnalyze your progress, reason about the best next action, then respond with the JSON object (progress_analysis → reasoning → action → bbox → stop → stair). Output ONLY the JSON, nothing else."""

USER_STOP_REJECTED = """\n\n[System: Your previous STOP decision was rejected (rejection count: {count}). The target was not clearly visible or not close enough. Continue navigating toward the destination.]"""

# ---------------------------------------------------------------------------
# Stop double-check prompt (Lavira-style second LLM call)
# ---------------------------------------------------------------------------

STOP_CHECK_SYSTEM_PROMPT = """/no_think
You are a navigation verification agent. Your only job is to confirm or reject a STOP decision."""

STOP_CHECK_USER_TEMPLATE = """Task: "{instruction}"

The navigation agent has decided to STOP, believing it has reached the destination.
Please verify this decision based on the current views.

Current 4-directional views:
[navigate to forward]  Front view:  <image>
[navigate to left]     Left view:   <image>
[navigate to right]    Right view:  <image>
[navigate to behind]   Behind view: <image>

Decision Rules:
- CONFIRM STOP: the destination target is clearly visible in one of the views AND you appear to be within ~2 meters of it.
- CONTINUE: the target is NOT visible, OR you are clearly more than 2 meters away.

Respond with JSON only (no markdown, no extra text):
{{"analysis": "<brief analysis of target visibility and distance>", "decision": "STOP" or "CONTINUE"}}"""


def build_stop_check_text(instruction: str) -> str:
    """Build the user text for the stop double-check call (no history, just current views)."""
    return STOP_CHECK_USER_TEMPLATE.format(instruction=instruction or "navigate to the goal")


def build_user_content_text(
    instruction: str,
    history_step_indices: list[int],
) -> str:
    """
    Build the text portion of the user message.

    The actual <image> tokens map (in order) to:
        history images (chronological) + 4 current views (F, L, R, B).

    Caller is responsible for passing the matching list of PIL/np images
    in the same order to the VLM processor.
    """
    parts = [USER_HEADER.format(instruction=instruction or "navigate to the goal")]

    if history_step_indices:
        for s_idx in history_step_indices:
            parts.append(USER_HISTORY_LABEL.format(step_idx=s_idx))
    else:
        parts.append(USER_NO_HISTORY)

    parts.append(USER_CURRENT_HEADER)
    parts.append(USER_CURRENT_VIEWS)
    parts.append(USER_TASK)

    return "\n".join(parts)


def expected_image_count(num_history: int, has_4dir: bool = True) -> int:
    """Number of <image> tokens the prompt requires for given history length."""
    return num_history + (4 if has_4dir else 1)


# ---------------------------------------------------------------------------
# LaViRA merged LA+VA prompt (Phase A schema alignment)
# ---------------------------------------------------------------------------

LAVIRA_MERGED_SYSTEM_PROMPT = """/no_think
You are a navigation agent in an indoor environment. You receive 4-directional views (front, left, right, behind) of the current location and a sequence of historical observation images. Based on the text instruction, history, and current views, you decide the next navigation action AND select the bounding box of the next target IN THE VIEW OF THE CHOSEN DIRECTION.

Action Space (each action corresponds to the labeled view image):
  - "navigate to forward" → [Front view]  go straight ahead
  - "navigate to left"    → [Left view]   turn left and advance
  - "navigate to right"   → [Right view]  turn right and advance
  - "navigate to behind"  → [Behind view] turn around and advance

You MUST respond with a valid JSON object with ALL of these fields in order:
{
    "progress_analysis": "<brief assessment: where am I now, what have I done, what remains>",
    "planning": "<remaining sub-goals / next-step intent>",
    "reasoning_action": "<reasoning for the chosen action given the plan and observations>",
    "action": "navigate to forward|navigate to left|navigate to right|navigate to behind",
    "stop": true|false,
    "stair": "up"|"down"|false,
    "reasoning_bbox": "<which view you are reading + reasoning for the bbox target>",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<short description of the target object or area>"
}

Guidelines:
- "progress_analysis": 1-2 sentences summarizing current position and task progress.
- "planning": short bullet-style plan for the next few sub-goals.
- "reasoning_action": justify the chosen action given the plan and observations.
- "action": choose one from the four options above.
- "stop": set true ONLY when you are within arm's reach of the final destination AND you have navigated significantly. Do NOT stop in the first few steps — explore first.
- "stair": "up" or "down" if next action involves stairs; otherwise false.
- "reasoning_bbox": cite which view you are looking at (e.g. "FORWARD view shows...") and justify the chosen target. Keep it under 50 words.
- "bbox_2d": bounding box [x1, y1, x2, y2] of the navigation target IN THE VIEW OF THE CHOSEN DIRECTION, using 0-1000 normalized coordinates (forward→front view, left→left view, right→right view, behind→behind view). The bbox MUST be in the chosen direction's view.
- "target": short description of the bbox content (e.g. "wooden door", "hallway entrance").
- Do NOT try to open doors.
- The target should be visible-but-not-too-close — at least ~1m away.
- If choosing stairs, the bbox MUST be on the ENTRY of the stairs and "stair" set to "up" or "down".
- Set stop=true whenever you have reached the target described in the instruction.
- Focus on following the text instruction.
- Output ONLY the JSON object — no markdown fence, no extra text.
"""

LAVIRA_MERGED_USER_TASK = """\n\nAnalyze your progress, plan the next sub-goals, reason about the best action and target bbox, then respond with the JSON object (all 9 fields in order: progress_analysis → planning → reasoning_action → action → stop → stair → reasoning_bbox → bbox_2d → target). Output ONLY the JSON, nothing else."""


def build_merged_user_content_text(
    instruction: str,
    history_step_indices: list[int],
    stop_rejection_feedback: str = "",
) -> str:
    """
    Build the text portion of the user message for the lavira_merged prompt style.

    Drop-in replacement for build_user_content_text; same image ordering
    (history + 4 current views), only the task description changes.
    """
    parts = [USER_HEADER.format(instruction=instruction or "navigate to the goal")]

    if history_step_indices:
        for s_idx in history_step_indices:
            parts.append(USER_HISTORY_LABEL.format(step_idx=s_idx))
    else:
        parts.append(USER_NO_HISTORY)

    parts.append(USER_CURRENT_HEADER)
    parts.append(USER_CURRENT_VIEWS)
    parts.append(LAVIRA_MERGED_USER_TASK)

    if stop_rejection_feedback:
        parts.append(stop_rejection_feedback)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LaViRA waypoint prompt (online macro-action policy)
# ---------------------------------------------------------------------------

LAVIRA_WAYPOINT_SYSTEM_PROMPT = """/no_think
You are an embodied navigation agent. Using the instruction, the navigation history, and the current views, pick the next action and ground the next target in the chosen direction's view.
"""

LAVIRA_WAYPOINT_HEADER = """# Instruction
"{instruction}"

# Inputs

Navigation history:"""

LAVIRA_WAYPOINT_NO_HISTORY = "(no waypoint history yet)"
LAVIRA_WAYPOINT_HISTORY_PREAMBLE = """
On each after-turn view below, the green box and red dot mark the target chosen at that waypoint at that time."""
LAVIRA_WAYPOINT_HISTORY_LABEL = """
Waypoint {waypoint_id} arrival view (image labeled WP{waypoint_id}):
<image>

At Waypoint {waypoint_id} you turned {turned_direction}. After-turn view (image labeled WP{waypoint_id}); target then: {target}.
<image>"""
LAVIRA_WAYPOINT_CONTINUOUS_HEADER = """
Waypoint {waypoint_id} -> Current Position (continuous frames):"""
LAVIRA_WAYPOINT_PREVIOUS_PROGRESS = """
Your previous progress_analysis: "{progress_analysis}\""""

LAVIRA_WAYPOINT_CURRENT_VIEWS = """
Current 4-directional views:

Current FORWARD view:
<image>

Current LEFT view:
<image>

Current BEHIND view:
<image>

Current RIGHT view:
<image>"""

LAVIRA_WAYPOINT_OUTPUT = """
# Available actions
{available_actions}

# Output
Return exactly ONE JSON object with these fields in this order, nothing before or after it, and stop right after the closing brace:
{{
    "progress_analysis": "<cumulative episode-progress summary>",
    "reasoning_plan_action": "<one-sentence justification of the planning/action>",
    "planning": "<next sub-goals, terse>",
    "action": "<one of the actions above>",
    "stop": <true or false>,
    "stair": <"up" or "down" or false>,
    "target": "<short visual target phrase>"
}}

# Field rules
- `progress_analysis` (<=50 words): cumulative summary of the whole episode — areas already visited and sub-goals already completed. Extend your previous progress_analysis: keep what is still true and append the latest progress. Do NOT describe the current views.
- `reasoning_plan_action` (<=30 words): one sentence stating why the chosen action follows the instruction.
- `planning` (<=25 words): the next sub-goals, terse.
- `action`: exactly one of the actions above.
- `stop`: Set stop=true whenever you have reached the target described in the instruction.
- `stair`: "up"/"down" only when taking stairs; otherwise false.
- `target` (<=8 words): short visible object/area phrase for GroundingDINO/SAM in the chosen direction view. Use an empty string for STOP or backtracking."""


def _action_to_turn_label(action: str) -> str:
    if action == "navigate to left":
        return "left"
    if action == "navigate to right":
        return "right"
    if action == "navigate to forward":
        return "forward"
    if action:
        return action.replace("navigate to ", "")
    return "unknown"


def build_waypoint_user_content_text(
    instruction: str,
    waypoint_ids: Optional[list[int]] = None,
    waypoint_infos: Optional[list[dict[str, Any]]] = None,
    stop_rejection_feedback: str = "",
) -> str:
    """Build user text for lavira_waypoint style.

    Image order: waypoint overlay images (chronological) + 4 current views.
    """
    parts = [
        LAVIRA_WAYPOINT_HEADER.format(
            instruction=instruction or "navigate to the goal"
        )
    ]
    if waypoint_infos is None:
        waypoint_infos = [
            {
                "id": wid,
                "action": "",
                "target": "",
                "progress_analysis": "",
                "continuous_count": 0,
            }
            for wid in (waypoint_ids or [])
        ]

    real_infos = [info for info in waypoint_infos if int(info.get("id", -1)) >= 0]
    if waypoint_infos:
        parts.append(LAVIRA_WAYPOINT_HISTORY_PREAMBLE)
        last_progress = ""
        for info in waypoint_infos:
            wid = int(info.get("id", -1))
            if wid >= 0:
                action = str(info.get("action", ""))
                target = str(info.get("target", "") or "unknown target")
                progress = str(info.get("progress_analysis", ""))
                continuous_count = int(info.get("continuous_count", 0))
                if progress:
                    last_progress = progress
                parts.append(
                    LAVIRA_WAYPOINT_HISTORY_LABEL.format(
                        waypoint_id=wid,
                        turned_direction=_action_to_turn_label(action),
                        target=target,
                    )
                )
                if continuous_count > 0:
                    parts.append(
                        LAVIRA_WAYPOINT_CONTINUOUS_HEADER.format(
                            waypoint_id=wid
                        )
                    )
                    parts.extend("<image>" for _ in range(continuous_count))
            else:
                continuous_count = int(info.get("continuous_count", 0))
                parts.append("Padding waypoint arrival view: <image>")
                parts.append("Padding waypoint after-turn view: <image>")
                if continuous_count > 0:
                    parts.append("Padding waypoint continuous frames:")
                    parts.extend("<image>" for _ in range(continuous_count))
        if last_progress:
            parts.append(
                LAVIRA_WAYPOINT_PREVIOUS_PROGRESS.format(
                    progress_analysis=last_progress
                )
            )
    else:
        parts.append(LAVIRA_WAYPOINT_NO_HISTORY)

    parts.append(LAVIRA_WAYPOINT_CURRENT_VIEWS)

    actions = [
        "   - navigate to forward - continue straight ahead",
        "   - navigate to left - turn left and go forward",
        "   - navigate to right - turn right and go forward",
    ]
    available_ids = [str(info["id"]) for info in real_infos]
    if available_ids:
        actions.append(
            "   - backtrack to <waypoint_id> - return to a previous waypoint "
            f"(Available IDs: {', '.join(available_ids)})"
        )
    parts.append(
        LAVIRA_WAYPOINT_OUTPUT.format(available_actions="\n".join(actions))
    )
    if stop_rejection_feedback:
        parts.append(stop_rejection_feedback)
    return "\n".join(parts)
