import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import REWARDS, SafeBuildInspectorEnv


def test_gymnasium_api():
    check_env(SafeBuildInspectorEnv(), skip_render_check=True)


def test_spaces_and_start_state():
    env = SafeBuildInspectorEnv()
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert env.action_space.n == 7
    assert info["agent_pos"] == (0, 0)          # starts at depot
    assert info["battery"] == 1.0
    assert info["remaining"] == env.n_buildings


def test_truncates_when_battery_runs_out():
    env = SafeBuildInspectorEnv(max_steps=20)
    env.reset(seed=2)
    for _ in range(20):
        _, _, term, trunc, _ = env.step(4)       # useless inspections
    assert trunc and not term


def test_correct_decisions_complete_mission():
    env = SafeBuildInspectorEnv()
    env.reset(seed=3)
    total, term = 0.0, False
    while not term:
        ay, ax = env.agent
        y, x = min(env._unclassified(), key=lambda c: abs(c[0] - ay) + abs(c[1] - ax))
        if (y, x) != (ay, ax):
            a = 0 if y < ay else 1 if y > ay else 2 if x > ax else 3
        else:
            b = env.buildings[(y, x)]
            a = 4 if not b["inspected"] else (5 if b["unsafe"] else 6)
        _, r, term, trunc, info = env.step(a)
        total += r
        assert not trunc
    assert info["missed_unsafe"] == 0 and info["false_alarms"] == 0
    assert total > 50


def test_reward_signs():
    assert REWARDS["clear_wrong"] < REWARDS["flag_wrong"] < 0 < REWARDS["clear_correct"] < REWARDS["flag_correct"]


def test_software_render():
    env = SafeBuildInspectorEnv(render_mode="rgb_array")
    env.reset(seed=4)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.mean() > 10
