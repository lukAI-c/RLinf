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
from typing import Optional


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
