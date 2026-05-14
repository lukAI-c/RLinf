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

Action Space:
  - "navigate to forward" - go straight ahead
  - "navigate to left"    - turn left and advance
  - "navigate to right"   - turn right and advance
  - "navigate to behind"  - turn around and advance

You MUST respond with a valid JSON object. Required fields:
{
    "action": "navigate to forward|navigate to left|navigate to right|navigate to behind",
    "bbox": [x1, y1, x2, y2] or null,
    "stop": true|false,
    "stair": "up"|"down"|false
}

Guidelines:
- Choose one action from the four options above.
- "bbox": pixel bounding box of the navigation target in the chosen direction's view. Use null if no clear target.
- Set "stop": true only when you have reached the destination.
- Set "stair": "up" or "down" if next action involves stairs; otherwise false.
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

USER_CURRENT_HEADER = "\n\nCurrent 4-directional views:"

USER_CURRENT_VIEWS = """Front view:  <image>
Left view:   <image>
Right view:  <image>
Behind view: <image>"""

USER_TASK = """\n\nDecide your next action and respond with the JSON object as instructed in the system prompt. Output ONLY the JSON, nothing else."""


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
