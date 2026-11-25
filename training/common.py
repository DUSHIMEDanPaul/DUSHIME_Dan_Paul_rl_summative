"""Shared helpers for all TorchRL training scripts."""

from __future__ import annotations

import csv
import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")

from torchrl.envs import Compose, GymWrapper, RewardSum, StepCounter, TransformedEnv  # noqa: E402
from torchrl.modules import MLP  # noqa: E402

try:  # TorchRL >= 0.15 renamed SyncDataCollector -> Collector
    from torchrl.collectors import SyncDataCollector as Collector  # type: ignore
except ImportError:  # pragma: no cover
    from torchrl.collectors import Collector  # noqa: F401

from environment.custom_env import SafeBuildInspectorEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"

ENV_KWARGS = dict(grid_size=10, n_buildings=8, max_steps=150)
EVAL_SEEDS = list(range(10_000, 10_020))  # unseen maps used for evaluation / generalisation


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int = 0):
    """TorchRL-wrapped SafeBuild environment (identical for every algorithm)."""
    base = SafeBuildInspectorEnv(**ENV_KWARGS)
    env = TransformedEnv(GymWrapper(base), Compose(StepCounter(), RewardSum()))
    env.set_seed(seed)
    return env


def obs_dim_and_actions():
    e = SafeBuildInspectorEnv(**ENV_KWARGS)
    return e.observation_space.shape[0], e.action_space.n


def build_mlp(in_dim: int, out_dim: int, hidden: list[int], activation=nn.Tanh) -> nn.Module:
    """Plain MLP shared by every algorithm so saved weights load in main.py."""
    return MLP(in_features=in_dim, out_features=out_dim, num_cells=list(hidden),
               activation_class=activation)


ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


def _act(net, obs, mode, kind, rng):
    out = net(torch.as_tensor(obs).unsqueeze(0))[0]
    if mode == "greedy":
        return int(out.argmax().item())
    if kind == "q":  # DQN stochastic evaluation = epsilon-greedy with eps 0.05
        return int(rng.integers(out.numel())) if rng.random() < 0.05 else int(out.argmax().item())
    return int(torch.distributions.Categorical(logits=out).sample().item())  # sample pi(a|s)


@torch.no_grad()
def evaluate(net: nn.Module, kind: str = "policy", seeds=EVAL_SEEDS) -> dict:
    """Evaluate on 20 unseen maps in two modes.

    kind: "q" for DQN (net outputs Q-values) or "policy" (net outputs logits).
    greedy     -> always the argmax action
    stochastic -> DQN: epsilon-greedy (0.05); policy gradient: sample from pi(a|s)
    eval_* keys report the better of the two modes (eval_mode says which).
    """
    env = SafeBuildInspectorEnv(**ENV_KWARGS)
    res = {}
    for mode in ("greedy", "stochastic"):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        returns, lengths, success, accuracy = [], [], [], []
        for s in seeds:
            obs, info = env.reset(seed=s)
            done, total, steps = False, 0.0, 0
            while not done:
                obs, r, term, trunc, info = env.step(_act(net, obs, mode, kind, rng))
                total += r
                steps += 1
                done = term or trunc
            returns.append(total)
            lengths.append(steps)
            success.append(float(info["remaining"] == 0))
            accuracy.append((info["correct_flags"] + info["correct_clears"]) / env.n_buildings)
        res[mode] = {
            "return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
            "length_mean": float(np.mean(lengths)), "success_rate": float(np.mean(success)),
            "accuracy": float(np.mean(accuracy)),
        }
    best = max(res, key=lambda m: res[m]["return_mean"])
    out = {f"eval_{k}": v for k, v in res[best].items()}
    out["eval_mode"] = best
    for mode in res:
        out[f"{mode}_return_mean"] = res[mode]["return_mean"]
        out[f"{mode}_success_rate"] = res[mode]["success_rate"]
        out[f"{mode}_accuracy"] = res[mode]["accuracy"]
    return out


class RunLogger:
    """Writes per-episode and per-update CSV logs plus a JSON summary."""

    def __init__(self, algo: str, run_name: str, config: dict):
        self.dir = LOG_DIR / algo / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.algo, self.run_name, self.config = algo, run_name, config
        self.t0 = time.time()
        self.episodes = 0
        self.ep_file = open(self.dir / "episodes.csv", "w", newline="")
        self.ep_writer = csv.writer(self.ep_file)
        self.ep_writer.writerow(["episode", "frames", "return", "length"])
        self.up_file = open(self.dir / "updates.csv", "w", newline="")
        self.up_writer = None
        with open(self.dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def log_batch_episodes(self, data, frames: int):
        """Record every episode that finished inside a collected batch."""
        done = data["next", "done"].reshape(-1)
        rets = data["next", "episode_reward"].reshape(-1)[done]
        lens = data["next", "step_count"].reshape(-1)[done]
        for r, l in zip(rets.tolist(), lens.tolist()):
            self.episodes += 1
            self.ep_writer.writerow([self.episodes, frames, round(r, 3), l])
        self.ep_file.flush()
        return rets

    def log_update(self, **metrics):
        if self.up_writer is None:
            self.up_writer = csv.DictWriter(self.up_file, fieldnames=list(metrics))
            self.up_writer.writeheader()
        self.up_writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()})
        self.up_file.flush()

    def finish(self, summary: dict):
        summary = {"algo": self.algo, "run": self.run_name, "episodes": self.episodes,
                   "train_minutes": round((time.time() - self.t0) / 60, 2), **summary, **self.config}
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        self.ep_file.close()
        self.up_file.close()
        append_sweep_row(self.algo, summary)
        return summary


def append_sweep_row(algo: str, row: dict):
    path = LOG_DIR / f"{algo}_sweep_results.csv"
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def save_model(net: nn.Module, algo: str, run_name: str, config: dict, metrics: dict):
    """Save weights + metadata; keep a copy as best_<algo>.pt if it beats the previous best."""
    sub = "dqn" if algo == "dqn" else "pg"
    out = MODEL_DIR / sub
    out.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": net.state_dict(), "algo": algo, "config": config, "metrics": metrics,
               "obs_dim": obs_dim_and_actions()[0], "n_actions": obs_dim_and_actions()[1]}
    torch.save(payload, out / f"{algo}_{run_name}.pt")
    best_path = out / f"best_{algo}.pt"
    if best_path.exists():
        prev = torch.load(best_path, weights_only=False)["metrics"].get("eval_return_mean", -1e9)
        if metrics["eval_return_mean"] <= prev:
            return False
    torch.save(payload, best_path)
    return True


def load_policy_net(path: str | os.PathLike) -> tuple[nn.Module, dict]:
    """Rebuild a saved network (used by main.py)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    net = build_mlp(ckpt["obs_dim"], ckpt["n_actions"], cfg["hidden"], ACTIVATIONS[cfg.get("activation", "tanh")])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
