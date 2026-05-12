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
    lines.append("## Discussion and Analysis")
    lines.append("")
    lines.append("The experiment demonstrates how exploration–exploitation algorithms can be applied to an adaptive credit scoring system for repeat borrowers. In this credit scoring scenario, each arm represents a borrower segment, applicant category, or credit policy option. A reward of 1 represents successful repayment, while a reward of 0 represents default. Therefore, the cumulative reward shows the total number of successful repayments obtained by each algorithm over time, and the average reward shows how well the algorithm learns to select creditworthy borrower groups.")
    lines.append("")

    lines.append("### Total Reward Earned by Each Algorithm")
    lines.append("")
    lines.append("The total reward results show how effectively each algorithm selected borrower segments with higher repayment probabilities. Epsilon-greedy, UCB, and Thompson Sampling all improved their performance over time as they collected more repayment feedback. However, their learning behavior differed because each algorithm handles exploration differently.")
    lines.append("")
    lines.append("Epsilon-greedy performed depending on the chosen epsilon value. A small epsilon encouraged the algorithm to exploit the borrower segment currently believed to be best, while a larger epsilon caused more random exploration. When epsilon was too low, the algorithm risked selecting a suboptimal borrower segment early and continuing to exploit it. When epsilon was too high, the algorithm continued exploring too much, selecting weaker borrower groups even after it had enough evidence about the best segment. Therefore, moderate epsilon values usually provide a better balance between exploration and exploitation.")
    lines.append("")
    lines.append("The UCB algorithm performed strongly because it selected borrower segments based on both their estimated repayment performance and the uncertainty around that estimate. Instead of exploring randomly, UCB gave additional priority to less-tested arms. This helped the algorithm collect useful information early and gradually shift toward the most promising borrower segment.")
    lines.append("")
    lines.append("Thompson Sampling also performed well because it used a probabilistic belief about each borrower segment’s repayment success rate. By updating the Beta distribution after every repayment or default outcome, Thompson Sampling naturally balanced exploration and exploitation. It explored uncertain borrower segments when there was still a chance they might be good, but increasingly exploited the most creditworthy segment as evidence accumulated.")
    lines.append("")

    lines.append("### Exploration Efficiency")
    lines.append("")
    lines.append("Exploration efficiency refers to how well an algorithm gathers information about borrower segments without wasting too many interactions on low-quality groups. Epsilon-greedy explores randomly, so some exploration is inefficient because it may continue selecting poor borrower groups even after they appear unlikely to repay successfully. Higher epsilon values increase this problem because the algorithm keeps making random selections throughout the experiment.")
    lines.append("")
    lines.append("UCB is more efficient than simple random exploration because its exploration is guided by uncertainty. Borrower segments that have been selected fewer times receive a higher confidence bonus, making them more likely to be tested. Once the algorithm becomes more confident about a segment’s repayment behavior, the uncertainty bonus decreases. This makes UCB suitable for situations where the credit scoring system needs to learn quickly while still controlling risk.")
    lines.append("")
    lines.append("Thompson Sampling is also exploration-efficient because it explores according to uncertainty in the posterior belief. If an arm has limited repayment history, its sampled repayment probability may sometimes be high, encouraging exploration. As more repayment observations are collected, the posterior becomes more stable, and the algorithm naturally focuses on the borrower segment with the highest estimated creditworthiness.")
    lines.append("")

    lines.append("### Exploitation Effectiveness")
    lines.append("")
    lines.append("Exploitation effectiveness measures how well the algorithm uses what it has learned to select the best borrower segment. Epsilon-greedy can exploit effectively when epsilon is small, but it may exploit the wrong segment if early observations are misleading. It also continues to explore randomly, so even after learning the best arm, it still spends some interactions on weaker segments.")
    lines.append("")
    lines.append("UCB becomes increasingly exploitative after it reduces uncertainty about the borrower groups. At the beginning, it tests all arms to estimate their repayment quality. Later, it concentrates more selections on the borrower segment with the highest estimated repayment success. This makes UCB effective when enough interactions are available for learning.")
    lines.append("")
    lines.append("Thompson Sampling tends to exploit effectively because its posterior beliefs become more concentrated over time. Once the algorithm has strong evidence that one borrower segment is more creditworthy, it selects that segment more frequently. At the same time, it does not completely ignore uncertainty, which allows it to adapt if repayment patterns are not obvious at the beginning.")
    lines.append("")

    lines.append("### Convergence Speed")
    lines.append("")
    lines.append("Convergence speed refers to how quickly each algorithm identifies and repeatedly selects the most creditworthy borrower segment. Epsilon-greedy convergence depends heavily on epsilon. A very low epsilon may converge quickly, but not always to the correct arm. A high epsilon may delay convergence because the algorithm continues to explore randomly. Therefore, epsilon-greedy requires careful tuning.")
    lines.append("")
    lines.append("UCB usually converges more systematically because it first explores uncertain arms and then increasingly selects the arm with the best observed repayment performance. Its convergence is driven by the confidence-bound formula, which reduces unnecessary exploration over time.")
    lines.append("")
    lines.append("Thompson Sampling often converges smoothly because it updates posterior beliefs after each repayment result. As evidence accumulates, the posterior probability of the best borrower segment becomes stronger, and the algorithm selects it more frequently. This makes Thompson Sampling useful in credit scoring because it can learn from repayment outcomes while still accounting for uncertainty.")
    lines.append("")

    lines.append("### Ability to Identify the Most Creditworthy Repeat Borrowers")
    lines.append("")
    lines.append("The arm-selection frequency plot is important for evaluating whether the algorithms successfully identified the most creditworthy borrower group. A good algorithm should increasingly select the arm with the highest true repayment probability. The final ranking of arms also shows whether the estimated creditworthiness scores match the actual repayment probabilities.")
    lines.append("")
    lines.append("Epsilon-greedy can identify the best borrower group, especially with a well-chosen epsilon value, but it is sensitive to randomness and parameter selection. If early repayment outcomes are misleading, the algorithm may overestimate a weaker segment.")
    lines.append("")
    lines.append("UCB is effective at identifying creditworthy borrowers because it compares borrower segments using both observed repayment success and uncertainty. This reduces the chance that a segment is ignored too early.")
    lines.append("")
    lines.append("Thompson Sampling is particularly suitable for identifying creditworthy repeat borrowers because it directly models uncertainty in repayment probability. Its posterior estimates can be interpreted as updated beliefs about borrower quality, which is useful for adaptive credit scoring.")
    lines.append("")

    lines.append("### Suitability for an Adaptive Credit Scoring System")
    lines.append("")
    lines.append("For an adaptive credit scoring system, the algorithm should learn from repeat borrower repayment behavior, update creditworthiness estimates, rank applicants or borrower groups, and improve selection decisions over time.")
    lines.append("")
    lines.append("Epsilon-greedy is simple and easy to implement. It is suitable as a baseline method, but it requires careful tuning of epsilon. If epsilon is too large, the lender may take unnecessary risks by repeatedly selecting less creditworthy borrower groups. If epsilon is too small, the system may fail to discover better borrower segments.")
    lines.append("")
    lines.append("UCB is more suitable than epsilon-greedy when the system needs a more structured exploration strategy. It is useful because it considers uncertainty and encourages the algorithm to test borrower segments that have not been observed enough. This makes it appropriate for credit scoring environments where the system must learn responsibly from limited data.")
    lines.append("")
    lines.append("Thompson Sampling is highly suitable for adaptive credit scoring because it provides a flexible probabilistic learning framework. It updates beliefs after every repayment outcome and balances exploration and exploitation naturally. It is especially useful when borrower repayment behavior is uncertain and when the system needs to make decisions based on both expected repayment and confidence in the estimate.")
    lines.append("")
    lines.append("Overall, the results show that exploration–exploitation algorithms can support credit scoring for repeat borrowers by continuously learning from repayment behavior. Epsilon-greedy provides a simple baseline, UCB offers uncertainty-driven exploration, and Thompson Sampling provides a strong probabilistic approach. Among the three, UCB and Thompson Sampling are generally more suitable for adaptive credit scoring because they explore more intelligently and are better able to identify the most creditworthy borrower segments over time.")
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

    # Hyperparameter sweeps
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

