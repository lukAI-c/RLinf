from types import SimpleNamespace

from rlinf.models.embodiment.qwen_nav.qwen_nav_policy import QwenNavPolicy


class _CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def _policy_for_mask_test() -> QwenNavPolicy:
    policy = object.__new__(QwenNavPolicy)
    policy.max_new_tokens = 512
    policy.processor = SimpleNamespace(tokenizer=_CharacterTokenizer())
    return policy


def test_action_target_mask_adds_target_without_unmasking_reasoning():
    text = (
        '{"reasoning_plan_action":"long explanation",'
        '"action":"navigate to right","stop":false,"stair":false,'
        '"target":"archway exit"}'
    )
    policy = _policy_for_mask_test()
    action_mask = policy._compute_action_loss_mask(
        text, len(text), mask_mode="action"
    )
    target_mask = policy._compute_action_loss_mask(
        text, len(text), mask_mode="action_target"
    )

    target_start = text.index("archway exit")
    reasoning_start = text.index("long explanation")
    assert not action_mask[target_start]
    assert target_mask[target_start]
    assert not target_mask[reasoning_start]
    assert target_mask.sum() > action_mask.sum()
