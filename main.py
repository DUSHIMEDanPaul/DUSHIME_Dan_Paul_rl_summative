"""
SafeBuild Inspector - run the best trained agent.

    uv run main.py                           # best saved model, 3D OpenGL window
    uv run main.py --algo ppo --episodes 3   # a specific algorithm
    uv run main.py --algo random             # random agent (no training needed)
    uv run main.py --render none             # terminal output only
    uv run main.py --record                  # also save a GIF to assets/

Every step is printed to the terminal (position, action, reward, battery,
event) so the agent's behaviour can be followed alongside the GUI.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from environment.custom_env import ACTION_NAMES, REWARDS, SafeBuildInspectorEnv  # noqa: E402

ROOT = Path(__file__).resolve().parent
ALGOS = ["dqn", "ppo", "a2c", "reinforce"]


def find_model(algo: str):
    """Return (path, algo) for the requested algorithm, or the best one overall."""
    candidates = []
    for a in (ALGOS if algo == "auto" else [algo]):
        sub = "dqn" if a == "dqn" else "pg"
        p = ROOT / "models" / sub / f"best_{a}.pt"
        if p.exists():
            score = torch.load(p, map_location="cpu", weights_only=False)["metrics"]["eval_return_mean"]
            candidates.append((score, p, a))
    if not candidates:
        return None, None
    score, path, a = max(candidates)
    return path, a


def print_header(algo, path, ckpt):
    print("=" * 78)
    print(" SafeBuild Inspector  -  detecting buildings in flood / landslide zones")
    print("=" * 78)
    print(" Objective : classify every building as SAFE or UNSAFE before the battery runs out")
    print(" Actions   : " + ", ".join(f"{k}={v}" for k, v in ACTION_NAMES.items()))
    print(" Rewards   : " + ", ".join(f"{k} {v:+g}" for k, v in REWARDS.items()))
    if path:
        m = ckpt["metrics"]
        print(f" Agent     : {algo.upper()}  ({path.relative_to(ROOT)})")
        print(f"             eval return {m['eval_return_mean']:.1f} | success {m['eval_success_rate']*100:.0f}% "
              f"| accuracy {m['eval_accuracy']*100:.0f}% on 20 unseen maps ({m.get('eval_mode', 'greedy')} mode)")
    else:
        print(f" Agent     : {algo.upper()}")
    print("-" * 78)


def run(args):
    if args.algo == "random":
        net, path, ckpt, algo = None, None, None, "random"
    else:
        path, algo = find_model(args.algo)
        if path is None:
            print(f"No trained model found for '{args.algo}'. Train first, e.g.\n"
                  "  uv run python -m training.dqn_training --config 0\n"
                  "  uv run python -m training.pg_training --algo ppo --config 0\n"
                  "Running a RANDOM agent instead so you can see the environment.\n")
            net, ckpt, algo = None, None, "random"
        else:
            from training.common import load_policy_net
            net, ckpt = load_policy_net(path)

    render_mode = None if args.render == "none" else args.render
    env = SafeBuildInspectorEnv(render_mode=render_mode, renderer=args.renderer)
    env.metadata["render_fps"] = args.fps
    recorder = None
    if args.record:
        from environment.rendering import SoftwareRenderer
        recorder = (SoftwareRenderer(env), [])

    mode = args.mode
    if mode == "auto":
        mode = ckpt["metrics"].get("eval_mode", "greedy") if ckpt else "greedy"
    if net is not None:
        from training.common import _act
        torch.manual_seed(args.seed)
    print_header(algo, path, ckpt)
    if net is not None:
        print(f" Action selection: {mode} "
              f"({'argmax' if mode == 'greedy' else ('epsilon-greedy 0.05' if algo == 'dqn' else 'sample from policy')})")
    rng = np.random.default_rng(args.seed)
    totals = []
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        print(f"\nEPISODE {ep}: {env.n_buildings} buildings, {info['unsafe_total']} truly unsafe "
              f"(drone starts at depot {info['agent_pos']})")
        done, total, step = False, 0.0, 0
        while not done:
            if net is None:
                action = int(rng.integers(env.action_space.n))
            else:
                action = _act(net, obs, mode, "q" if algo == "dqn" else "policy", rng)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            step += 1
            done = term or trunc
            if recorder:
                recorder[1].append(recorder[0].draw(env))
            if not args.quiet:
                print(f" [ep {ep} | step {step:03d}] pos={info['agent_pos']} {ACTION_NAMES[action]:<11} "
                      f"r={reward:+6.2f} total={total:+7.2f} battery={info['battery']*100:5.1f}% "
                      f"| {info['last_event']}")
        outcome = "MISSION COMPLETE" if term else "BATTERY EMPTY (truncated)"
        print(f" -> {outcome}: return {total:.2f} in {step} steps | correct flags {info['correct_flags']}, "
              f"correct clears {info['correct_clears']}, false alarms {info['false_alarms']}, "
              f"missed hazards {info['missed_unsafe']}, undecided {info['remaining']}")
        totals.append(total)
    print(f"\nMean return over {args.episodes} episode(s): {np.mean(totals):.2f}")
    env.close()

    if recorder and recorder[1]:
        from PIL import Image
        out = ROOT / "assets" / f"{algo}_agent.gif"
        frames = [Image.fromarray(f) for f in recorder[1]]
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0)
        print(f"Saved recording to {out.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser(description="Run a trained SafeBuild Inspector agent")
    p.add_argument("--algo", default="auto", choices=["auto", *ALGOS, "random"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--render", default="human", choices=["human", "rgb_array", "none"])
    p.add_argument("--renderer", default="auto", choices=["auto", "opengl", "software"])
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--mode", default="auto", choices=["auto", "greedy", "stochastic"],
                   help="greedy = argmax; stochastic = sample policy (PG) / eps-greedy 0.05 (DQN); "
                        "auto = whichever scored best in evaluation")
    p.add_argument("--seed", type=int, default=20_000, help="maps not used in training/eval")
    p.add_argument("--record", action="store_true", help="save a GIF to assets/")
    p.add_argument("--quiet", action="store_true", help="only print episode summaries")
    args = p.parse_args()
    try:
        run(args)
    except SystemExit as e:
        print(e)
        sys.exit(0)


if __name__ == "__main__":
    main()
