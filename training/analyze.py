"""
Turn the training logs into the figures and tables needed for the report.

    uv run python -m training.analyze

Outputs (in assets/):
  sweep_<algo>.csv            all 10 configs with hyperparameters + eval metrics
  fig_cumulative_rewards.png  episode return curves of the best run of each method
  fig_dqn_loss.png            DQN objective (TD loss) over training
  fig_pg_entropy.png          policy entropy over training (REINFORCE, PPO, A2C)
  fig_sweeps.png              eval return of every config, per algorithm
  fig_generalisation.png      best agents on 20 unseen maps
  summary_table.csv           best config per method, episodes to converge, etc.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS, ASSETS = ROOT / "logs", ROOT / "assets"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
COLOURS = {"dqn": "#1f77b4", "reinforce": "#d62728", "ppo": "#2ca02c", "a2c": "#9467bd"}
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}


def load_sweep(algo):
    p = LOGS / f"{algo}_sweep_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # keep the latest result per config (if a config was re-run)
    return df.drop_duplicates(subset=["run"], keep="last").sort_values("config_id")


def best_run(df):
    return df.loc[df["eval_return_mean"].idxmax()]


def episodes_to_converge(ep_df, threshold_frac=0.9, window=50):
    """First episode where the moving average reaches 90% of its final plateau."""
    ma = ep_df["return"].rolling(window, min_periods=window).mean()
    if ma.dropna().empty:
        return None
    lo, hi = ma.min(), ma.dropna().iloc[-window:].mean()
    target = lo + threshold_frac * (hi - lo)
    hit = ma[ma >= target]
    return int(ep_df.loc[hit.index[0], "episode"]) if not hit.empty else None


def main():
    ASSETS.mkdir(exist_ok=True)
    sweeps = {a: load_sweep(a) for a in ALGOS}
    sweeps = {a: d for a, d in sweeps.items() if d is not None and len(d)}
    if not sweeps:
        print("No logs found. Train first.")
        return

    rows = []
    fig_r, ax_r = plt.subplots(len(sweeps), 1, figsize=(9, 2.6 * len(sweeps)), squeeze=False)
    for k, (algo, df) in enumerate(sweeps.items()):
        df.to_csv(ASSETS / f"sweep_{algo}.csv", index=False)
        best = best_run(df)
        ep = pd.read_csv(LOGS / algo / best["run"] / "episodes.csv")
        conv = episodes_to_converge(ep)
        ax = ax_r[k][0]
        ax.plot(ep["episode"], ep["return"], color=COLOURS[algo], alpha=0.25, lw=0.8)
        ax.plot(ep["episode"], ep["return"].rolling(50, min_periods=1).mean(), color=COLOURS[algo], lw=2,
                label=f"{NAMES[algo]} (config {int(best['config_id'])}) - 50-ep moving avg")
        if conv:
            ax.axvline(conv, ls="--", color="grey", lw=1)
            ax.text(conv, ax.get_ylim()[0], f" ~converged ep {conv}", color="grey", fontsize=8, va="bottom")
        ax.set_ylabel("episode return")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        rows.append({"method": NAMES[algo], "best_config": int(best["config_id"]),
                     "eval_return_mean": round(best["eval_return_mean"], 2),
                     "eval_return_std": round(best["eval_return_std"], 2),
                     "eval_success_rate": round(best["eval_success_rate"], 2),
                     "eval_accuracy": round(best["eval_accuracy"], 2),
                     "eval_mode": best.get("eval_mode", "greedy"),
                     "greedy_return": round(best.get("greedy_return_mean", float("nan")), 2),
                     "stochastic_return": round(best.get("stochastic_return_mean", float("nan")), 2),
                     "episodes_trained": int(best["episodes"]),
                     "episodes_to_converge": conv,
                     "train_minutes": best["train_minutes"]})
    ax_r[-1][0].set_xlabel("episode")
    fig_r.suptitle("Cumulative reward per episode (best run of each method)")
    fig_r.tight_layout()
    fig_r.savefig(ASSETS / "fig_cumulative_rewards.png", dpi=150)

    # DQN objective
    if "dqn" in sweeps:
        best = best_run(sweeps["dqn"])
        up = pd.read_csv(LOGS / "dqn" / best["run"] / "updates.csv")
        fig, ax1 = plt.subplots(figsize=(8, 3.2))
        ax1.plot(up["frames"], up["loss"].rolling(20, min_periods=1).mean(), color=COLOURS["dqn"])
        ax1.set_xlabel("environment frames"); ax1.set_ylabel("TD loss (smooth L1)")
        ax2 = ax1.twinx(); ax2.plot(up["frames"], up["epsilon"], color="orange", ls="--"); ax2.set_ylabel("epsilon")
        ax1.set_title(f"DQN objective - config {int(best['config_id'])}"); ax1.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(ASSETS / "fig_dqn_loss.png", dpi=150)

    # Policy entropy
    pg = [a for a in ("reinforce", "ppo", "a2c") if a in sweeps]
    if pg:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        for algo in pg:
            best = best_run(sweeps[algo])
            up = pd.read_csv(LOGS / algo / best["run"] / "updates.csv")
            ax.plot(up["frames"], up["entropy"].rolling(10, min_periods=1).mean(), color=COLOURS[algo],
                    label=f"{NAMES[algo]} (config {int(best['config_id'])})")
        ax.axhline(1.9459, color="grey", ls=":", lw=1, label="uniform policy (ln 7)")
        ax.set_xlabel("environment frames"); ax.set_ylabel("policy entropy"); ax.legend(fontsize=8)
        ax.set_title("Policy entropy during training"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(ASSETS / "fig_pg_entropy.png", dpi=150)

    # Sweep comparison
    fig, axes = plt.subplots(1, len(sweeps), figsize=(3.4 * len(sweeps), 3.2), squeeze=False)
    for k, (algo, df) in enumerate(sweeps.items()):
        ax = axes[0][k]
        ax.bar(df["config_id"].astype(int), df["eval_return_mean"], color=COLOURS[algo],
               yerr=df["eval_return_std"], capsize=2)
        ax.set_title(NAMES[algo]); ax.set_xlabel("config"); ax.set_xticks(range(10))
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("eval return (20 unseen maps)")
    fig.tight_layout(); fig.savefig(ASSETS / "fig_sweeps.png", dpi=150)

    # Generalisation of the best agents
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(table["method"], table["eval_return_mean"], yerr=table["eval_return_std"], capsize=4,
           color=[COLOURS[a] for a in sweeps])
    for i, r in table.iterrows():
        ax.text(i, r["eval_return_mean"], f"{r['eval_success_rate']*100:.0f}% done\n{r['eval_accuracy']*100:.0f}% acc",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mean return on unseen maps"); ax.set_title("Generalisation: best agent per method")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(ASSETS / "fig_generalisation.png", dpi=150)

    table.to_csv(ASSETS / "summary_table.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nFigures and tables written to {ASSETS}")
    with open(ASSETS / "summary_table.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
