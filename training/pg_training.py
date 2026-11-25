"""
Policy-gradient training with TorchRL: REINFORCE, PPO and A2C (Actor-Critic).

Usage (from the project root):
    uv run python -m training.pg_training --algo ppo --config 0
    uv run python -m training.pg_training --algo reinforce --sweep
    uv run python -m training.pg_training --algo all --sweep --frames 100000
"""

from __future__ import annotations

import argparse

import torch
from tensordict.nn import TensorDictModule
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE
from torchrl.objectives.value.functional import reward2go

from training.common import (ACTIVATIONS, Collector, RunLogger, build_mlp, evaluate, make_env,
                             obs_dim_and_actions, save_model, set_seed)
from training.configs import DEFAULT_FRAMES, get_config


def build_actor_critic(cfg, env, with_critic=True):
    obs_dim, n_actions = obs_dim_and_actions()
    act = ACTIVATIONS[cfg["activation"]]
    policy_net = build_mlp(obs_dim, n_actions, cfg["hidden"], act)
    actor = ProbabilisticActor(
        TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"]),
        spec=env.action_spec, in_keys=["logits"], distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    critic = None
    if with_critic:
        critic = ValueOperator(build_mlp(obs_dim, 1, cfg["hidden"], act), in_keys=["observation"])
    return policy_net, actor, critic


def _entropy(actor, td):
    with torch.no_grad():
        return float(actor.get_dist(td).entropy().mean())


# ---------------------------------------------------------------------------- REINFORCE
def train_reinforce(cfg, total_frames, seed=0, verbose=True):
    set_seed(seed)
    run_name = f"cfg{cfg['config_id']}_seed{seed}"
    logger = RunLogger("reinforce", run_name, {**cfg, "total_frames": total_frames, "seed": seed})
    env = make_env(seed)
    policy_net, actor, critic = build_actor_critic(cfg, env, with_critic=True)
    # TorchRL's ReinforceLoss: loss_actor = -log pi(a|s) * advantage; loss_value trains the
    # critic towards the Monte-Carlo return. With use_baseline=False the critic is ignored
    # and the advantage is the raw return G_t (vanilla REINFORCE).
    loss_module = ReinforceLoss(actor, critic)
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])
    # reset_at_each_iter=True -> each batch starts with fresh episodes (Monte-Carlo returns)
    collector = Collector(lambda: make_env(seed), actor, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=total_frames, reset_at_each_iter=True)
    frames, recent = 0, []
    for i, data in enumerate(collector):
        frames += data.numel()
        rets = logger.log_batch_episodes(data, frames)
        recent = (recent + rets.tolist())[-50:]
        # Monte-Carlo reward-to-go G_t (episodes cut at the batch end count as ended)
        done = data["next", "done"].clone()
        done[-1] = True
        g = reward2go(data["next", "reward"], done, cfg["gamma"], time_dim=-2)
        data.set("value_target", g)
        with torch.no_grad():
            v = critic(data.select("observation").clone())["state_value"]
        adv = g - v if cfg["use_baseline"] else g
        if cfg["normalize_advantage"]:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data.set("advantage", adv)
        out = loss_module(data)
        ent = actor.get_dist(data).entropy().mean()
        loss = out["loss_actor"] - cfg["entropy_coef"] * ent
        if cfg["use_baseline"]:
            loss = loss + cfg["critic_coef"] * out["loss_value"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["grad_clip"])
        optim.step()
        mean_ret = sum(recent) / max(1, len(recent))
        logger.log_update(frames=frames, loss=float(loss), entropy=float(ent), mean_return_50=mean_ret)
        if verbose and i % 20 == 0:
            print(f"[reinforce cfg{cfg['config_id']}] frames {frames:>7} | loss {float(loss):.3f} | "
                  f"entropy {float(ent):.3f} | mean return(50) {mean_ret:7.2f}")
    collector.shutdown()
    return _finish("reinforce", policy_net, cfg, run_name, logger, recent)


# ---------------------------------------------------------------------------- PPO
def train_ppo(cfg, total_frames, seed=0, verbose=True):
    set_seed(seed)
    run_name = f"cfg{cfg['config_id']}_seed{seed}"
    logger = RunLogger("ppo", run_name, {**cfg, "total_frames": total_frames, "seed": seed})
    env = make_env(seed)
    policy_net, actor, critic = build_actor_critic(cfg, env)
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=True)
    loss_module = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip_epsilon"], entropy_bonus=True,
                              entropy_coeff=cfg["entropy_coef"], critic_coeff=cfg["critic_coef"],
                              loss_critic_type="smooth_l1")
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])
    collector = Collector(lambda: make_env(seed), actor, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=total_frames)
    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg["frames_per_batch"]),
                          sampler=SamplerWithoutReplacement(), batch_size=cfg["minibatch_size"])
    frames, recent = 0, []
    n_mb = max(1, cfg["frames_per_batch"] // cfg["minibatch_size"])
    for i, data in enumerate(collector):
        frames += data.numel()
        rets = logger.log_batch_episodes(data, frames)
        recent = (recent + rets.tolist())[-50:]
        stats = {"loss_objective": 0.0, "loss_critic": 0.0, "entropy": 0.0}
        for _ in range(cfg["num_epochs"]):
            with torch.no_grad():
                gae(data)
            buffer.extend(data.reshape(-1))
            for _ in range(n_mb):
                sub = buffer.sample()
                out = loss_module(sub)
                loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["grad_clip"])
                optim.step()
                stats["loss_objective"] += float(out["loss_objective"])
                stats["loss_critic"] += float(out["loss_critic"])
                stats["entropy"] += float(out["entropy"])
        k = cfg["num_epochs"] * n_mb
        mean_ret = sum(recent) / max(1, len(recent))
        logger.log_update(frames=frames, loss_objective=stats["loss_objective"] / k,
                          loss_critic=stats["loss_critic"] / k, entropy=stats["entropy"] / k,
                          mean_return_50=mean_ret)
        if verbose and i % 10 == 0:
            print(f"[ppo cfg{cfg['config_id']}] frames {frames:>7} | entropy {stats['entropy']/k:.3f} | "
                  f"critic {stats['loss_critic']/k:.3f} | mean return(50) {mean_ret:7.2f}")
    collector.shutdown()
    return _finish("ppo", policy_net, cfg, run_name, logger, recent)


# ---------------------------------------------------------------------------- A2C
def train_a2c(cfg, total_frames, seed=0, verbose=True):
    set_seed(seed)
    run_name = f"cfg{cfg['config_id']}_seed{seed}"
    logger = RunLogger("a2c", run_name, {**cfg, "total_frames": total_frames, "seed": seed})
    env = make_env(seed)
    policy_net, actor, critic = build_actor_critic(cfg, env)
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=False)
    loss_module = A2CLoss(actor, critic, entropy_bonus=True, entropy_coeff=cfg["entropy_coef"],
                          critic_coeff=cfg["critic_coef"], loss_critic_type="smooth_l1")
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])
    collector = Collector(lambda: make_env(seed), actor, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=total_frames)
    frames, recent = 0, []
    for i, data in enumerate(collector):
        frames += data.numel()
        rets = logger.log_batch_episodes(data, frames)
        recent = (recent + rets.tolist())[-50:]
        with torch.no_grad():
            gae(data)
        out = loss_module(data)
        loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["grad_clip"])
        optim.step()
        mean_ret = sum(recent) / max(1, len(recent))
        ent = float(out["entropy"]) if "entropy" in out.keys() else _entropy(actor, data)
        logger.log_update(frames=frames, loss_objective=float(out["loss_objective"]),
                          loss_critic=float(out["loss_critic"]), entropy=ent, mean_return_50=mean_ret)
        if verbose and i % 200 == 0:
            print(f"[a2c cfg{cfg['config_id']}] frames {frames:>7} | entropy {ent:.3f} | "
                  f"critic {float(out['loss_critic']):.3f} | mean return(50) {mean_ret:7.2f}")
    collector.shutdown()
    return _finish("a2c", policy_net, cfg, run_name, logger, recent)


def _finish(algo, policy_net, cfg, run_name, logger, recent):
    metrics = evaluate(policy_net)
    is_best = save_model(policy_net, algo, run_name, cfg, metrics)
    summary = logger.finish({**metrics, "final_train_return_50": sum(recent) / max(1, len(recent))})
    print(f"[{algo} cfg{cfg['config_id']}] EVAL on 20 unseen maps: {metrics}" + ("  <- new best" if is_best else ""))
    return summary


TRAINERS = {"reinforce": train_reinforce, "ppo": train_ppo, "a2c": train_a2c}


def main():
    p = argparse.ArgumentParser(description="Train policy-gradient agents on SafeBuild Inspector")
    p.add_argument("--algo", choices=["reinforce", "ppo", "a2c", "all"], default="ppo")
    p.add_argument("--config", type=int, default=0, help="config index 0-9")
    p.add_argument("--sweep", action="store_true", help="run all 10 configs")
    p.add_argument("--frames", type=int, default=None, help="frames per run (default per algo)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    algos = ["reinforce", "ppo", "a2c"] if args.algo == "all" else [args.algo]
    for algo in algos:
        frames = args.frames or DEFAULT_FRAMES[algo]
        for idx in (range(10) if args.sweep else [args.config]):
            TRAINERS[algo](get_config(algo, idx), frames, args.seed)


if __name__ == "__main__":
    main()
