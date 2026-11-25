"""
DQN (value-based) training with TorchRL.

Usage (from the project root):
    uv run python -m training.dqn_training --config 0            # one run
    uv run python -m training.dqn_training --sweep               # all 10 configs
    uv run python -m training.dqn_training --sweep --frames 50000 # quicker sweep
"""

from __future__ import annotations

import argparse

import torch
from tensordict.nn import TensorDictSequential
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate

from training.common import (ACTIVATIONS, Collector, RunLogger, build_mlp, evaluate, make_env,
                             obs_dim_and_actions, save_model, set_seed)
from training.configs import DEFAULT_FRAMES, get_config


def train_dqn(cfg: dict, total_frames: int, seed: int = 0, verbose: bool = True) -> dict:
    set_seed(seed)
    run_name = f"cfg{cfg['config_id']}_seed{seed}"
    logger = RunLogger("dqn", run_name, {**cfg, "total_frames": total_frames, "seed": seed})

    env = make_env(seed)
    obs_dim, n_actions = obs_dim_and_actions()
    q_net = build_mlp(obs_dim, n_actions, cfg["hidden"], ACTIVATIONS[cfg["activation"]])
    actor = QValueActor(q_net, in_keys=["observation"], spec=env.action_spec)
    egreedy = EGreedyModule(spec=env.action_spec, eps_init=cfg["eps_init"], eps_end=cfg["eps_end"],
                            annealing_num_steps=int(cfg["eps_anneal_frac"] * total_frames))
    policy = TensorDictSequential(actor, egreedy)

    collector = Collector(lambda: make_env(seed), policy, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=total_frames, init_random_frames=cfg["init_random_frames"])
    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]), batch_size=cfg["batch_size"])

    loss_module = DQNLoss(actor, action_space=env.action_spec, delay_value=True,
                          double_dqn=cfg["double_dqn"], loss_function="smooth_l1")
    loss_module.make_value_estimator(gamma=cfg["gamma"])
    target_updater = SoftUpdate(loss_module, tau=cfg["tau"])
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])

    frames, recent = 0, []
    for i, data in enumerate(collector):
        frames += data.numel()
        buffer.extend(data.reshape(-1))
        egreedy.step(data.numel())
        rets = logger.log_batch_episodes(data, frames)
        recent = (recent + rets.tolist())[-50:]

        if frames < cfg["init_random_frames"]:
            continue
        losses = []
        for _ in range(cfg["updates_per_batch"]):
            sample = buffer.sample()
            loss = loss_module(sample)["loss"]
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["grad_clip"])
            optim.step()
            target_updater.step()
            losses.append(loss.item())
        mean_ret = sum(recent) / max(1, len(recent))
        logger.log_update(frames=frames, loss=sum(losses) / len(losses), epsilon=float(egreedy.eps),
                          mean_return_50=mean_ret)
        if verbose and i % 50 == 0:
            print(f"[dqn cfg{cfg['config_id']}] frames {frames:>7} | loss {losses[-1]:.3f} | "
                  f"eps {float(egreedy.eps):.2f} | mean return(50) {mean_ret:7.2f}")
    collector.shutdown()

    metrics = evaluate(q_net, kind="q")
    is_best = save_model(q_net, "dqn", run_name, cfg, metrics)
    summary = logger.finish({**metrics, "final_train_return_50": sum(recent) / max(1, len(recent))})
    print(f"[dqn cfg{cfg['config_id']}] EVAL on 20 unseen maps: {metrics}" + ("  <- new best" if is_best else ""))
    return summary


def main():
    p = argparse.ArgumentParser(description="Train DQN on SafeBuild Inspector")
    p.add_argument("--config", type=int, default=0, help="config index 0-9")
    p.add_argument("--sweep", action="store_true", help="run all 10 configs")
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES["dqn"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ids = range(10) if args.sweep else [args.config]
    for idx in ids:
        train_dqn(get_config("dqn", idx), args.frames, args.seed)


if __name__ == "__main__":
    main()
