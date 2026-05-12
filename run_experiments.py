from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bandits import (
    arm_selection_frequencies,
    epsilon_greedy,
    rank_arms_by_estimated_creditworthiness,
    thompson_sampling_bernoulli,
    ucb1,
)
from svg_plots import LineSeries, grouped_bar_svg, line_chart_svg


@dataclass(frozen=True)
class Scenario:
    name: str
    true_ps: np.ndarray  # Bernoulli repayment success probabilities per "segment"


def default_credit_scenario() -> Scenario:
    """
    Toy credit scoring reapplication setup:
    - Each arm is a repeat-borrower segment / policy tier
    - Reward=1 means successful repayment; reward=0 means default

    The gaps are intentionally small to mimic realistic segment separations.
    """
    true_ps = np.array([0.58, 0.61, 0.65, 0.69, 0.72, 0.74])
    return Scenario(name="repeat_borrowers_toy", true_ps=true_ps)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def _time_to_reach(avg_reward: np.ndarray, target: float) -> Optional[int]:
    """
    Return the first time step (1-indexed) where avg_reward >= target, else None.
    """
    idx = np.flatnonzero(avg_reward >= target)
    if idx.size == 0:
        return None
    return int(idx[0] + 1)


def _best_arm_rate_in_tail(actions: np.ndarray, best_arm: int, tail_frac: float = 0.1) -> float:
    n = actions.size
    tail_n = max(1, int(np.ceil(n * tail_frac)))
    tail = actions[-tail_n:]
    return float(np.mean(tail == best_arm))


def _format_ranking(indices: np.ndarray) -> str:
    return " > ".join(str(int(i)) for i in indices)


def plot_run(
    outdir: str,
    title: str,
    steps: int,
    runs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    true_ps: Sequence[float],
) -> None:
    """
    runs[label] = (mean_cum_reward, mean_avg_reward, mean_arm_freq)
    """
    os.makedirs(outdir, exist_ok=True)
    t = np.arange(1, steps + 1)

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    labels = list(runs.keys())
    colors = {lbl: palette[i % len(palette)] for i, lbl in enumerate(labels)}

    # Cumulative reward (SVG)
    cum_series = [LineSeries(label=lbl, y=runs[lbl][0], color=colors[lbl]) for lbl in labels]
    cum_svg = line_chart_svg(
        title=f"Cumulative reward over time — {title}",
        x_label="Borrower interactions (t)",
        y_label="Cumulative successful repayments",
        x=t,
        series=cum_series,
    )
    with open(os.path.join(outdir, f"{title}__cumulative_reward.svg"), "w", encoding="utf-8") as f:
        f.write(cum_svg)

    # Average reward (SVG)
    avg_series = [LineSeries(label=lbl, y=runs[lbl][1], color=colors[lbl]) for lbl in labels]
    avg_svg = line_chart_svg(
        title=f"Average reward over time — {title}",
        x_label="Borrower interactions (t)",
        y_label="Average repayment success rate",
        x=t,
        series=avg_series,
        y_lim=(0.0, 1.0),
    )
    with open(os.path.join(outdir, f"{title}__average_reward.svg"), "w", encoding="utf-8") as f:
        f.write(avg_svg)

    # Arm selection frequencies (SVG)
    k = len(true_ps)
    categories = [str(i) for i in range(k)]
    bar_series = [(lbl, runs[lbl][2], colors[lbl]) for lbl in labels]
    bars_svg = grouped_bar_svg(
        title=f"Arm selection frequency — {title}",
        x_label="Arm (segment / policy variant)",
        y_label="Selection frequency",
        categories=categories,
        series=bar_series,
        overlay_line=("True p(repay)", np.asarray(true_ps, dtype=float), "#000000"),
        y_lim=(0.0, 1.0),
    )
    with open(os.path.join(outdir, f"{title}__arm_frequencies.svg"), "w", encoding="utf-8") as f:
        f.write(bars_svg)


def summarize_table(
    labels: List[str],
    totals: List[float],
    steps: int,
    outpath: str,
) -> None:
    best_idx = int(np.argmax(np.asarray(totals)))
    lines = []
    lines.append("## Experiment summary (mean over simulations)")
    lines.append("")
    lines.append("| Method | Total reward | Avg reward |")
    lines.append("|---|---:|---:|")
    for label, total in zip(labels, totals):
        lines.append(f"| {label} | {total:.1f} | {total/steps:.4f} |")
    lines.append("")
    lines.append(f"Best (by total reward): **{labels[best_idx]}**")
    lines.append("")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_report(
    outdir: str,
    scenario: Scenario,
    steps: int,
    sims: int,
    epsilon_results: Dict[str, Dict[str, float]],
    ucb_results: Dict[str, Dict[str, float]],
    comparison_results: Dict[str, Dict[str, float]],
) -> None:
    best_arm = int(np.argmax(scenario.true_ps))
    best_p = float(np.max(scenario.true_ps))
    lines: List[str] = []
    lines.append("# Bandit Algorithms for Adaptive Credit Scoring — Report")
    lines.append("")
    lines.append("## Scenario (repeat borrower reapplication)")
    lines.append(f"- **Arms (K)**: {scenario.true_ps.size}")
    lines.append(f"- **Steps (T)**: {steps}")
    lines.append(f"- **Simulations**: {sims}")
    lines.append(f"- **True repayment probabilities**: `{scenario.true_ps.tolist()}`")
    lines.append(f"- **Best arm (most creditworthy segment)**: arm **{best_arm}** with p(repay)=**{best_p:.2f}**")
    lines.append("")
    lines.append("Interpretation: each algorithm learns an estimated repayment success probability per arm. This acts as a **creditworthiness score**, used to **rank** arms and select which borrower segment/policy to allocate decisions to over time.")
    lines.append("")

    def section(title: str, rows: Dict[str, Dict[str, float]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Setting / Method | Mean total reward | Mean avg reward | Mean regret | Tail best-arm rate | Convergence t* |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for label, m in rows.items():
            conv = m.get("convergence_t", float("nan"))
            conv_str = f"{int(conv)}" if np.isfinite(conv) else "—"
            lines.append(
                f"| {label} | {m['total_reward']:.1f} | {m['avg_reward']:.4f} | {m['regret']:.1f} | {m['tail_best_arm_rate']:.3f} | {conv_str} |"
            )
        lines.append("")
        lines.append(r"- **Mean regret** uses oracle reward \(T\cdot p^* - \text{total reward}\), where \(p^*\) is the best arm’s true repayment probability.")
        lines.append("- **Tail best-arm rate** is the fraction of pulls of the true best arm in the last 10% of interactions (how strongly it exploits once learned).")
        lines.append("- **Convergence t\\*** is the first step where the running average reward reaches **95%** of the best arm’s true probability (lower is faster).")
        lines.append("")

    section("Epsilon-greedy sweep (exploration vs exploitation)", epsilon_results)
    section("UCB sweep (uncertainty-driven exploration)", ucb_results)

    lines.append("## Final comparison (best settings + Thompson sampling)")
    lines.append("")
    lines.append("| Method | Mean total reward | Mean avg reward | Mean regret | Tail best-arm rate | Convergence t* | Mean final ranking |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for label, m in comparison_results.items():
        conv = m.get("convergence_t", float("nan"))
        conv_str = f"{int(conv)}" if np.isfinite(conv) else "—"
        lines.append(
            f"| {label} | {m['total_reward']:.1f} | {m['avg_reward']:.4f} | {m['regret']:.1f} | {m['tail_best_arm_rate']:.3f} | {conv_str} | {m['mean_ranking']} |"
        )
    lines.append("")

    lines.append("## Discussion guide (map to assignment prompts)")
    lines.append("")
    lines.append("- **Total reward earned**: use the tables above and `comparison__summary.md` to state which algorithm earns most repayments.")
    lines.append("- **Exploration efficiency**: compare regret + how quickly the best arm becomes dominant in the arm-frequency plots.")
    lines.append("- **Exploitation effectiveness**: compare tail best-arm rates and arm-frequency concentration on the best arm.")
    lines.append(r"- **Convergence speed**: compare convergence \(t^*\) and the average-reward curves approaching \(p^*\).")
    lines.append("- **Identifying most creditworthy borrowers**: look for high selection frequency of the best arm and a ranking that places the best arm first.")
    lines.append("- **Suitability for adaptive credit scoring**:")
    lines.append("  - epsilon-greedy: simplest; needs tuning of ε; may waste pulls if ε too high; may get stuck if ε too low.")
    lines.append("  - UCB: principled exploration via uncertainty; deterministic given history; often strong early learning.")
    lines.append("  - Thompson sampling: Bayesian uncertainty; typically strong balance; produces an interpretable posterior for each segment’s repayment probability.")
    lines.append("")
    lines.append("## Plots to cite in your submission")
    lines.append("")
    lines.append("- `epsilon_sweep__cumulative_reward.svg`, `epsilon_sweep__average_reward.svg`, `epsilon_sweep__arm_frequencies.svg`")
    lines.append("- `ucb_sweep__cumulative_reward.svg`, `ucb_sweep__average_reward.svg`, `ucb_sweep__arm_frequencies.svg`")
    lines.append("- `comparison__cumulative_reward.svg`, `comparison__average_reward.svg`, `comparison__arm_frequencies.svg`")
    lines.append("")

    with open(os.path.join(outdir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandit algorithms for adaptive credit scoring (repeat borrowers).")
    parser.add_argument("--steps", type=int, default=5000, help="Number of borrower interactions.")
    parser.add_argument("--sims", type=int, default=50, help="Number of Monte Carlo simulations per setting.")
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed.")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to write plots/results.")
    parser.add_argument(
        "--true-ps",
        type=str,
        default="",
        help="Optional comma-separated Bernoulli repayment probabilities per arm (overrides default scenario). Example: 0.55,0.6,0.7",
    )
    args = parser.parse_args()

    scenario = default_credit_scenario()
    if args.true_ps.strip():
        ps = np.array(parse_float_list(args.true_ps), dtype=float)
        scenario = Scenario(name="custom", true_ps=ps)
    k = scenario.true_ps.size
    steps = int(args.steps)
    sims = int(args.sims)

    # Hyperparameter sweeps requested by the assignment
    epsilons = [0.01, 0.05, 0.10, 0.20]
    explorations = [0.25, 0.50, 1.00, 2.00]

    # For epsilon-greedy and UCB we treat true_reward_dists as Bernoulli(p)
    true_dists = [float(p) for p in scenario.true_ps]
    best_arm = int(np.argmax(scenario.true_ps))
    best_p = float(np.max(scenario.true_ps))
    oracle_total = steps * best_p

    def mc_epsilon(eps: float):
        cum = np.zeros(steps)
        avg = np.zeros(steps)
        freq = np.zeros(k)
        total = 0.0
        regret = 0.0
        tail_best = 0.0
        conv_times: List[int] = []
        rankings: List[np.ndarray] = []
        for s in range(sims):
            run = epsilon_greedy(k, true_dists, steps, eps, seed=args.seed + 1000 * s)
            cum += run.cumulative_reward
            avg += run.average_reward
            freq += arm_selection_frequencies(run.actions, k)
            total += run.total_reward
            regret += oracle_total - run.total_reward
            tail_best += _best_arm_rate_in_tail(run.actions, best_arm, 0.1)
            ct = _time_to_reach(run.average_reward, 0.95 * best_p)
            if ct is not None:
                conv_times.append(ct)
            rankings.append(rank_arms_by_estimated_creditworthiness(run))
        metrics = {
            "total_reward": total / sims,
            "avg_reward": (total / sims) / steps,
            "regret": regret / sims,
            "tail_best_arm_rate": tail_best / sims,
            "convergence_t": float(np.mean(conv_times)) if conv_times else float("nan"),
            "mean_ranking": _format_ranking(np.round(np.mean(np.stack(rankings), axis=0)).astype(int)),
        }
        return cum / sims, avg / sims, freq / sims, metrics

    def mc_ucb(c: float):
        cum = np.zeros(steps)
        avg = np.zeros(steps)
        freq = np.zeros(k)
        total = 0.0
        regret = 0.0
        tail_best = 0.0
        conv_times: List[int] = []
        rankings: List[np.ndarray] = []
        for s in range(sims):
            run = ucb1(k, true_dists, steps, exploration=c, seed=args.seed + 2000 * s)
            cum += run.cumulative_reward
            avg += run.average_reward
            freq += arm_selection_frequencies(run.actions, k)
            total += run.total_reward
            regret += oracle_total - run.total_reward
            tail_best += _best_arm_rate_in_tail(run.actions, best_arm, 0.1)
            ct = _time_to_reach(run.average_reward, 0.95 * best_p)
            if ct is not None:
                conv_times.append(ct)
            rankings.append(rank_arms_by_estimated_creditworthiness(run))
        metrics = {
            "total_reward": total / sims,
            "avg_reward": (total / sims) / steps,
            "regret": regret / sims,
            "tail_best_arm_rate": tail_best / sims,
            "convergence_t": float(np.mean(conv_times)) if conv_times else float("nan"),
            "mean_ranking": _format_ranking(np.round(np.mean(np.stack(rankings), axis=0)).astype(int)),
        }
        return cum / sims, avg / sims, freq / sims, metrics

    def mc_ts():
        cum = np.zeros(steps)
        avg = np.zeros(steps)
        freq = np.zeros(k)
        total = 0.0
        regret = 0.0
        tail_best = 0.0
        conv_times: List[int] = []
        rankings: List[np.ndarray] = []
        for s in range(sims):
            run = thompson_sampling_bernoulli(k, scenario.true_ps, steps, seed=args.seed + 3000 * s)
            cum += run.cumulative_reward
            avg += run.average_reward
            freq += arm_selection_frequencies(run.actions, k)
            total += run.total_reward
            regret += oracle_total - run.total_reward
            tail_best += _best_arm_rate_in_tail(run.actions, best_arm, 0.1)
            ct = _time_to_reach(run.average_reward, 0.95 * best_p)
            if ct is not None:
                conv_times.append(ct)
            rankings.append(rank_arms_by_estimated_creditworthiness(run))
        metrics = {
            "total_reward": total / sims,
            "avg_reward": (total / sims) / steps,
            "regret": regret / sims,
            "tail_best_arm_rate": tail_best / sims,
            "convergence_t": float(np.mean(conv_times)) if conv_times else float("nan"),
            "mean_ranking": _format_ranking(np.round(np.mean(np.stack(rankings), axis=0)).astype(int)),
        }
        return cum / sims, avg / sims, freq / sims, metrics

    # Plot: epsilon sweep
    eps_runs = {}
    eps_totals = []
    eps_labels = []
    eps_metrics: Dict[str, Dict[str, float]] = {}
    for eps in epsilons:
        label = f"epsilon-greedy (ε={eps:.2f})"
        cum, avg, freq, metrics = mc_epsilon(eps)
        eps_runs[label] = (cum, avg, freq)
        eps_totals.append(metrics["total_reward"])
        eps_labels.append(label)
        eps_metrics[label] = metrics
    plot_run(args.outdir, "epsilon_sweep", steps, eps_runs, scenario.true_ps)
    summarize_table(eps_labels, eps_totals, steps, os.path.join(args.outdir, "epsilon_sweep__summary.md"))

    # Plot: UCB exploration sweep
    ucb_runs = {}
    ucb_totals = []
    ucb_labels = []
    ucb_metrics: Dict[str, Dict[str, float]] = {}
    for c in explorations:
        label = f"UCB1 (c={c:.2f})"
        cum, avg, freq, metrics = mc_ucb(c)
        ucb_runs[label] = (cum, avg, freq)
        ucb_totals.append(metrics["total_reward"])
        ucb_labels.append(label)
        ucb_metrics[label] = metrics
    plot_run(args.outdir, "ucb_sweep", steps, ucb_runs, scenario.true_ps)
    summarize_table(ucb_labels, ucb_totals, steps, os.path.join(args.outdir, "ucb_sweep__summary.md"))

    # Plot: best-of sweeps vs Thompson Sampling
    best_eps = float(epsilons[int(np.argmax(np.asarray(eps_totals)))])
    best_c = float(explorations[int(np.argmax(np.asarray(ucb_totals)))])

    comp_runs = {}
    comp_totals = []
    comp_labels = []
    comp_metrics: Dict[str, Dict[str, float]] = {}

    label_eps = f"epsilon-greedy (best ε={best_eps:.2f})"
    cum, avg, freq, metrics = mc_epsilon(best_eps)
    comp_runs[label_eps] = (cum, avg, freq)
    comp_totals.append(metrics["total_reward"])
    comp_labels.append(label_eps)
    comp_metrics[label_eps] = metrics

    label_ucb = f"UCB1 (best c={best_c:.2f})"
    cum, avg, freq, metrics = mc_ucb(best_c)
    comp_runs[label_ucb] = (cum, avg, freq)
    comp_totals.append(metrics["total_reward"])
    comp_labels.append(label_ucb)
    comp_metrics[label_ucb] = metrics

    label_ts = "Thompson Sampling (Bernoulli)"
    cum, avg, freq, metrics = mc_ts()
    comp_runs[label_ts] = (cum, avg, freq)
    comp_totals.append(metrics["total_reward"])
    comp_labels.append(label_ts)
    comp_metrics[label_ts] = metrics

    plot_run(args.outdir, "comparison", steps, comp_runs, scenario.true_ps)
    summarize_table(comp_labels, comp_totals, steps, os.path.join(args.outdir, "comparison__summary.md"))

    write_report(args.outdir, scenario, steps, sims, eps_metrics, ucb_metrics, comp_metrics)
    print(f"Wrote plots and summaries to: {args.outdir}/")


if __name__ == "__main__":
    main()

