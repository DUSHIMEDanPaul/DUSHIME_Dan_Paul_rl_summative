"""SafeBuild Inspector environment package."""
from gymnasium.envs.registration import register

from environment.custom_env import SafeBuildInspectorEnv, ACTION_NAMES, REWARDS

register(id="SafeBuildInspector-v0", entry_point="environment.custom_env:SafeBuildInspectorEnv")

__all__ = ["SafeBuildInspectorEnv", "ACTION_NAMES", "REWARDS"]
