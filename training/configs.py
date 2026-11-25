"""
Hyperparameter sweeps: 10 configurations per algorithm (as required).

Each config changes one or two settings relative to a baseline (config 0) so the
effect of every hyperparameter can be discussed in the report.
"""

DQN_BASE = dict(
    lr=1e-3, gamma=0.99, buffer_size=50_000, batch_size=128, hidden=[128, 128],
    activation="relu", eps_init=1.0, eps_end=0.05, eps_anneal_frac=0.5,
    tau=0.01, double_dqn=False, updates_per_batch=8, frames_per_batch=200,
    init_random_frames=2_000, grad_clip=10.0,
)
DQN_CONFIGS = [
    {},                                              # 0 baseline
    {"lr": 3e-4},                                     # 1 lower learning rate
    {"lr": 3e-3},                                     # 2 higher learning rate
    {"gamma": 0.95},                                  # 3 shorter horizon
    {"buffer_size": 10_000},                          # 4 small replay buffer
    {"batch_size": 64},                               # 5 smaller batch
    {"eps_anneal_frac": 0.2, "eps_end": 0.02},        # 6 fast exploration decay
    {"eps_anneal_frac": 0.8, "eps_end": 0.1},         # 7 slow exploration decay
    {"double_dqn": True},                             # 8 Double DQN
    {"hidden": [256, 256], "tau": 0.005},             # 9 bigger net, slower target
]

REINFORCE_BASE = dict(
    lr=1e-3, gamma=0.99, hidden=[128, 128], activation="tanh",
    frames_per_batch=1_500, entropy_coef=0.01, use_baseline=True, normalize_advantage=True,
    critic_coef=0.5, grad_clip=1.0,
)
REINFORCE_CONFIGS = [
    {},                                              # 0 baseline
    {"lr": 3e-4},                                     # 1
    {"lr": 3e-3},                                     # 2
    {"gamma": 0.95},                                  # 3
    {"frames_per_batch": 3_000},                      # 4 more episodes per update
    {"frames_per_batch": 750},                        # 5 fewer episodes per update
    {"entropy_coef": 0.0},                            # 6 no entropy bonus
    {"entropy_coef": 0.05},                           # 7 strong entropy bonus
    {"use_baseline": False},                          # 8 vanilla REINFORCE (no baseline)
    {"hidden": [256, 256]},                           # 9 bigger net
]

PPO_BASE = dict(
    lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, hidden=[128, 128],
    activation="tanh", frames_per_batch=2_000, num_epochs=8, minibatch_size=256,
    entropy_coef=0.01, critic_coef=0.5, grad_clip=0.5,
)
PPO_CONFIGS = [
    {},                                              # 0 baseline
    {"lr": 1e-4},                                     # 1
    {"lr": 1e-3},                                     # 2
    {"gamma": 0.95},                                  # 3
    {"gae_lambda": 0.8},                              # 4
    {"clip_epsilon": 0.1},                            # 5 tighter clipping
    {"clip_epsilon": 0.3},                            # 6 looser clipping
    {"num_epochs": 4, "minibatch_size": 512},         # 7 fewer, larger updates
    {"entropy_coef": 0.05},                           # 8 more exploration
    {"frames_per_batch": 4_000, "hidden": [256, 256]},  # 9 bigger batch + net
]

A2C_BASE = dict(
    lr=7e-4, gamma=0.99, gae_lambda=1.0, hidden=[128, 128], activation="tanh",
    frames_per_batch=200, entropy_coef=0.01, critic_coef=0.5, grad_clip=0.5,
)
A2C_CONFIGS = [
    {},                                              # 0 baseline (n-step returns)
    {"lr": 3e-4},                                     # 1
    {"lr": 2e-3},                                     # 2
    {"gamma": 0.95},                                  # 3
    {"gae_lambda": 0.95},                             # 4 GAE instead of n-step
    {"frames_per_batch": 50},                         # 5 shorter rollouts
    {"frames_per_batch": 800},                        # 6 longer rollouts
    {"entropy_coef": 0.0},                            # 7 no entropy bonus
    {"entropy_coef": 0.05},                           # 8 strong entropy bonus
    {"critic_coef": 1.0, "hidden": [256, 256]},       # 9 stronger critic, bigger net
]

ALL = {
    "dqn": (DQN_BASE, DQN_CONFIGS),
    "reinforce": (REINFORCE_BASE, REINFORCE_CONFIGS),
    "ppo": (PPO_BASE, PPO_CONFIGS),
    "a2c": (A2C_BASE, A2C_CONFIGS),
}

# Default training budget per run (environment frames). Increase for better agents.
DEFAULT_FRAMES = {"dqn": 200_000, "reinforce": 300_000, "ppo": 300_000, "a2c": 300_000}


def get_config(algo: str, idx: int) -> dict:
    base, variants = ALL[algo]
    cfg = dict(base)
    cfg.update(variants[idx])
    cfg["config_id"] = idx
    return cfg
