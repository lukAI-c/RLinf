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

from .canonical_targets import DINO_TARGET_VOCAB


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
# Stop double-check prompt (copied from LHX prompts_vln.py)
# ---------------------------------------------------------------------------

STOP_CHECK_SYSTEM_PROMPT = """/no_think
You are a navigation verification agent. Your only job is to confirm or reject a STOP decision."""

# DITA-style termination judge.  Unlike the source STOP double-check prompt,
# this is deliberately neutral: it is evaluated at every sampled navigation
# decision and must not assume that the policy has already chosen STOP.
TERMINATION_SHADOW_SYSTEM_PROMPT = """/no_think
You are a navigation termination-state judge. Decide only whether the agent is
currently at the final destination described by the original instruction."""

TERMINATION_SHADOW_USER_TEMPLATE = """Original navigation instruction:
"{instruction}"

Inspect the current four directional views. Determine whether the final goal
condition in the instruction is satisfied at the agent's current location.
Ignore any navigation action proposed by another policy.

Return true only when the agent is currently at the final destination and
should terminate. Otherwise return false.

Answer with exactly one JSON boolean literal: true or false."""

STOP_CHECK_USER_TEMPLATE = """You are an intelligent navigation agent.
Task: "{instruction}"

You have decided to STOP, believing you have reached the target.
Now, please double-check your decision based on the current 4-directional views and the final target.

Final Target:
{target}

Current Views:
{current_views}

Requirements:
1. Check if the target object is clearly visible in any of the views.
2. Estimate if the distance to the target is less than {distance_threshold:g} meter(s).
3. Compare previous views with current views to confirm you have approached the target.

Decision Rules:
- If target is visible AND distance < {distance_threshold:g}m: CONFIRM STOP.
- If target is NOT visible OR distance >= {distance_threshold:g}m: CONTINUE NAVIGATION.

Response format (JSON):
{{
    "analysis": "<analysis of visibility and distance, and comparison with previous views>",
    "decision": "STOP" or "CONTINUE"
}}
"""


def build_stop_check_text(
    instruction: str,
    distance_threshold: float = 1.0,
    target: str = "",
) -> str:
    """Build LHX's stop-check text with textual image placeholders."""
    return STOP_CHECK_USER_TEMPLATE.format(
        instruction=instruction or "navigate to the goal",
        distance_threshold=float(distance_threshold),
        target=target or "the destination described by the instruction",
        current_views=(
            "Current FORWARD view: <image>\n"
            "View after turning LEFT: <image>\n"
            "View after turning BEHIND: <image>\n"
            "View after turning RIGHT: <image>"
        ),
    )


def build_termination_shadow_text(instruction: str) -> str:
    """Build the neutral prompt used by the read-only termination shadow judge."""
    return TERMINATION_SHADOW_USER_TEMPLATE.format(
        instruction=instruction or "navigate to the goal",
    )


def build_source_stop_check_parts(
    instruction: str,
    target: str,
    distance_threshold: float = 1.0,
) -> tuple[str, str]:
    """Return the two text blocks surrounding LHX's interleaved views."""
    rendered = STOP_CHECK_USER_TEMPLATE.format(
        instruction=instruction or "navigate to the goal",
        target=target or "the destination described by the instruction",
        distance_threshold=float(distance_threshold),
        current_views="{current_views}",
    )
    prefix, suffix = rendered.split("{current_views}", 1)
    return prefix, suffix


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

LAVIRA_WAYPOINT_SYSTEM_PROMPT = """You are an embodied navigation agent. Using the instruction, the navigation history, and the current views, pick the next action and ground the next target in the chosen direction's view.
"""

LAVIRA_CANONICAL_SYSTEM_PROMPT = f"""You are an embodied navigation agent. Pick the next action and name one visible target in the chosen direction's view.

For target_class use exactly one visual noun from this list:
{', '.join(DINO_TARGET_VOCAB)}
For target_region use exactly one of: left, center, right, any. Do not include
direction, destination, function, material, or relational words in target_class.
For STOP or BACKTRACK, target_class must be \"\" and target_region must be \"any\".
Output only the JSON object requested by the user.
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
- `target` (<=8 words): name the visible object or area GroundingDINO should detect in the CHOSEN direction's view. For backtracking, leave this empty."""


# Copied from lavira-rft ``LA_PROMPT_BACKTRACK_REPLAN``.  The caller inserts
# the image tokens at its three source section boundaries.
LAVIRA_BACKTRACK_REPLAN_HEADER = """You are a navigation agent. You have backtracked to a previous waypoint to have a second chance to choose an action and name the next visible target.

Instruction: "{instruction}"
{old_progress_block}
Navigation History (up to this waypoint):
"""

LAVIRA_CANONICAL_OUTPUT = """
# Output
Return exactly ONE JSON object with these fields in this order, nothing before or after it:
{
    "progress_analysis": "<cumulative episode-progress summary>",
    "reasoning_plan_action": "<one-sentence justification of the planning/action>",
    "planning": "<next sub-goals, terse>",
    "action": "<one of the actions above>",
    "stop": <true or false>,
    "stair": <"up" or "down" or false>,
    "target_class": "<one canonical visual noun or empty for STOP/BACKTRACK>",
    "target_region": "<left|center|right|any>"
}

# target_class examples: "archway" and "rug" are valid; "archway exit" and
# "doorway to hallway" are invalid because they contain non-visual relations.
# target_region is the image third containing the target, not the turn action.
"""

LAVIRA_BACKTRACK_REPLAN_FAILED = """
Previous Trajectory (the path taken from here that you are reconsidering):
Previous Action: **navigate to {previous_action}** from here.
"""

LAVIRA_BACKTRACK_REPLAN_CURRENT = """
Current 4-directional views at this waypoint:

Current FORWARD view:
<image>

Current LEFT view:
<image>

Current BEHIND view:
<image>

Current RIGHT view:
<image>
"""

LAVIRA_BACKTRACK_REPLAN_OUTPUT = """
Task:
1. FIRST, write `progress_analysis`: one paragraph describing what you've observed and done so far, including the outcome of the Previous Trajectory.
2. Write `reasoning_plan_action`: justify the upcoming plan and the chosen action given the observations and failed trajectory.
3. Write `planning`: short bullet-style plan for the next few sub-goals.
4. Pick `action`. Choose ONE of:
{available_actions}
5. Decide `stop` (true when target reached) and `stair` ("up"|"down"|false).
6. Output `target`: a short visible object/area phrase for GroundingDINO localization.

# Output
Return exactly ONE JSON object with these fields in this order, nothing before or after it:
{{
    "progress_analysis": "<assessment including failed trajectory>",
    "reasoning_plan_action": "<second-chance reasoning>",
    "planning": "<remaining sub-goals>",
    "action": "<one of the actions above>",
    "stop": <true or false>,
    "stair": <"up" or "down" or false>,
    "target": "<short visual target phrase>"
}}
"""


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
    available_backtrack_ids: Optional[list[int]] = None,
    stop_rejection_feedback: str = "",
    allow_move_behind: Optional[bool] = None,
    blocked_directions: Optional[set[str]] = None,
    canonical: bool = False,
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

    # Only expose waypoints that the controller still accepts as backtrack
    # targets. Failed branches remain visible as history for explanation, but
    # must not be copied into the action enum as live candidates.
    real_infos = [
        info for info in waypoint_infos
        if int(info.get("id", -1)) >= 0
        and not bool(info.get("failed", False))
        and not bool(info.get("failed_dir", False))
    ]
    if allow_move_behind is None:
        # Match LaViRA-RFT: behind is a navigation action only at the initial
        # waypoint. Later prompts still show the BEHIND view for reasoning.
        allow_move_behind = not real_infos
    if waypoint_infos:
        zone3_note = next((str(info.get("zone3_note", "")) for info in waypoint_infos
                           if info.get("zone3_note")), "")
        if zone3_note:
            parts.append(zone3_note)
        parts.append(LAVIRA_WAYPOINT_HISTORY_PREAMBLE)
        last_progress = ""
        for info in waypoint_infos:
            wid = int(info.get("id", -1))
            if info.get("kind") == "initial":
                continuous_count = int(info.get("continuous_count", 0))
                padding_count = int(info.get("padding_count", 0))
                parts.append("Initial views (continuous):")
                parts.extend("<image>" for _ in range(continuous_count))
                if padding_count:
                    parts.append("Fixed history padding (not trajectory evidence):")
                    parts.extend("<image>" for _ in range(padding_count))
                continue
            if wid >= 0:
                action = str(info.get("action", ""))
                target = str(info.get("target", "") or "unknown target")
                progress = str(info.get("progress_analysis", ""))
                continuous_count = int(info.get("continuous_count", 0))
                if progress:
                    last_progress = progress
                if bool(info.get("failed", False)):
                    parts.append(
                        f"Waypoint {wid} belongs to an abandoned failed branch; "
                        "its images are intentionally omitted from navigation history."
                    )
                    parts.extend(
                        "<image>" for _ in range(
                            2 + continuous_count + int(info.get("padding_count", 0))
                        )
                    )
                    continue
                if bool(info.get("failed_dir", False)):
                    parts.append(
                        f"Waypoint {wid} arrival view (image labeled WP{wid}):\n<image>\n\n"
                        f"At Waypoint {wid}, the prior turn started an abandoned branch; "
                        "its after-turn image is intentionally blank.\n<image>"
                    )
                else:
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
                padding_count = int(info.get("padding_count", 0))
                if padding_count:
                    parts.append("Fixed history padding (not trajectory evidence):")
                    parts.extend("<image>" for _ in range(padding_count))
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
    candidate_directions = ["forward", "left", "right"]
    if allow_move_behind:
        actions.append("   - navigate to behind - turn around and go forward")
        candidate_directions.append("behind")
    # Exact source policy: remove depth-blocked navigation directions before
    # presenting the action contract, with the same behind/forward fallback.
    blocked = set(blocked_directions or ())
    allowed_navigation = [
        direction for direction in candidate_directions if direction not in blocked
    ]
    if not allowed_navigation:
        if "behind" in candidate_directions:
            allowed_navigation = ["behind"]
        elif "forward" in candidate_directions:
            allowed_navigation = ["forward"]
    actions = [
        line for line in actions
        if not line.startswith("   - navigate to ")
        or line.removeprefix("   - navigate to ").split(" - ", 1)[0] in allowed_navigation
    ]
    # In runtime mode this list is the controller's current reachability
    # decision.  Do not merge it with prompt-cache history: the two structures
    # advance at different points in a replay and a stale cached node can no
    # longer be a legal FMM backtrack target.
    if available_backtrack_ids is None:
        available_ids = [str(info["id"]) for info in real_infos]
    else:
        available_ids = [str(waypoint_id) for waypoint_id in available_backtrack_ids]
    if available_ids:
        actions.append(
            "   - backtrack to <waypoint_id> - return to a previous waypoint "
            f"(Available IDs: {', '.join(available_ids)})"
        )
    output_template = LAVIRA_CANONICAL_OUTPUT if canonical else LAVIRA_WAYPOINT_OUTPUT
    if canonical:
        parts.append("# Available actions\n" + "\n".join(actions))
        parts.append(output_template)
    else:
        parts.append(output_template.format(available_actions="\n".join(actions)))
    if stop_rejection_feedback:
        parts.append(stop_rejection_feedback)
    return "\n".join(parts)


def build_backtrack_replan_user_content_text(
    instruction: str,
    history_entries: list[dict[str, Any]],
    failed_entries: list[dict[str, Any]],
    previous_action: str,
    blocked_directions: Optional[set[str]] = None,
    padding_image_count: int = 0,
) -> str:
    """Genesis image-token adapter for source ``replan_at_backtrack``.

    The source function interleaves image URLs with the three prompt sections.
    Qwen's processor instead consumes ordered PIL images plus ``<image>``
    markers, so this retains the source section ordering while leaving image
    transport to the caller.
    """
    parts = [LAVIRA_BACKTRACK_REPLAN_HEADER.format(
        instruction=instruction or "navigate to the goal",
        old_progress_block="",
    )]
    for entry in history_entries:
        waypoint_id = entry["id"]
        parts.append(f"Waypoint {waypoint_id} retained history:")
        parts.extend("<image>" for _ in range(int(entry["image_count"])))

    parts.append(LAVIRA_BACKTRACK_REPLAN_FAILED.format(
        previous_action=_action_to_turn_label(previous_action),
    ))
    if failed_entries:
        # Keep the source evaluator's explicit delimiter: these observations
        # describe the abandoned branch and are not positive route evidence.
        parts.append("Trajectory after Backtrack Point (Failed Path):")
        for entry in failed_entries:
            if entry["id"] != "path":
                parts.append(f"Abandoned waypoint {entry['id']} (failed branch):")
            parts.extend("<image>" for _ in range(int(entry["image_count"])))
    else:
        parts.append("None (Immediate failure).")
    if padding_image_count:
        parts.append("Padding visual context (not part of the trajectory):")
        parts.extend("<image>" for _ in range(padding_image_count))

    parts.append(LAVIRA_BACKTRACK_REPLAN_CURRENT)
    blocked = set(blocked_directions or ())
    directions = ["forward", "left", "right", "behind"]
    allowed = [direction for direction in directions if direction not in blocked]
    if not allowed:
        allowed = ["behind"]
    actions = {
        "forward": "navigate to forward - continue straight ahead",
        "left": "navigate to left - turn left and go forward",
        "right": "navigate to right - turn right and go forward",
        "behind": "navigate to behind - turn around and go forward",
    }
    parts.append(LAVIRA_BACKTRACK_REPLAN_OUTPUT.format(
        available_actions="\n".join(f"   - {actions[d]}" for d in allowed),
    ))
    return "\n".join(parts)
