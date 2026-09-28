from rlinf.envs.genark.genark_env import is_clean_stop_success


def test_clean_stop_is_the_only_maxrl_success():
    assert is_clean_stop_success("stop", 2.0, 3.0)
    assert not is_clean_stop_success("timeout", 2.0, 3.0)
    assert not is_clean_stop_success("stop", 3.0, 3.0)
    assert not is_clean_stop_success("stop", float("nan"), 3.0)
