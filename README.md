# Exploration–Exploitation Algorithms for Credit Scoring (Repeat Borrowers)

This repo contains **three multi-armed bandit exploration–exploitation algorithms** applied to a **credit scoring reapplication** scenario, where the model learns from **repeat borrowers' repayment outcomes** to rank/select the most promising applicants over time.

## What “arms” mean in this assignment

In a credit scoring setting, each arm can represent a:
- borrower segment (e.g., “stable income”, “thin file”, “high utilization”)
- credit policy variant (approval rule, limit, pricing tier)
- applicant category (risk bucket, score band)

Each time-step \(t\) is a borrower interaction (reapplication / new decision). A reward \(r_t\in\{0,1\}\) represents repayment success (1) vs default (0).

## Algorithms implemented

- **Epsilon-greedy** (`epsilon_greedy` in `bandits.py`)
- **Upper Confidence Bound (UCB1)** (`ucb1` in `bandits.py`)
- **Thompson Sampling (Bernoulli)** (`thompson_sampling_bernoulli` in `bandits.py`)

## How to run

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run experiments (generates plots + markdown summaries in `outputs/`):

```bash
python run_experiments.py --steps 5000 --sims 50 --seed 7 --outdir outputs
```

## Outputs you’ll get

For each sweep and for the final comparison, the runner writes:
- `*_cumulative_reward.svg`: cumulative reward over time
- `*_average_reward.svg`: average reward over time
- `*_arm_frequencies.svg`: arm-selection frequency (with the true repayment probabilities shown)
- `*_summary.md`: mean total reward table (over simulations)
- `REPORT.md`: a ready-to-submit analysis template with extra metrics (regret, convergence, tail best-arm rate, rankings)

## Custom scenario

You can override the default toy repeat-borrower probabilities:

```bash
python run_experiments.py --true-ps 0.55,0.60,0.62,0.70 --steps 5000 --sims 50
```

## Analysis notes (what to discuss in your write-up)

Use the generated plots/summaries to discuss:
- **Total reward**: which method yields the most successful repayments over time
- **Exploration efficiency**: how quickly it finds the best arm(s) with minimal wasted pulls
- **Exploitation effectiveness**: how strongly it concentrates on the best segment once identified
- **Convergence speed**: how fast the average reward approaches the best arm’s true probability
- **Identification of the most creditworthy repeat borrowers**: whether selection frequency concentrates on the highest-\(p\) arm
- **Suitability for adaptive credit scoring**:
  - epsilon-greedy is simple but needs careful \(\epsilon\) tuning
  - UCB is deterministic given history and explores “optimistically” when uncertain
  - Thompson sampling is Bayesian, often strong in practice, and naturally balances uncertainty vs reward

