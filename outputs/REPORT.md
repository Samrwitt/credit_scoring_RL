# Bandit Algorithms for Adaptive Credit Scoring — Report

## Scenario (repeat borrower reapplication)
- **Arms (K)**: 6
- **Steps (T)**: 2000
- **Simulations**: 20
- **True repayment probabilities**: `[0.58, 0.61, 0.65, 0.69, 0.72, 0.74]`
- **Best arm (most creditworthy segment)**: arm **5** with p(repay)=**0.74**

Interpretation: each algorithm learns an estimated repayment success probability per arm. This acts as a **creditworthiness score**, used to **rank** arms and select which borrower segment/policy to allocate decisions to over time.

## Epsilon-greedy sweep (exploration vs exploitation)

| Setting / Method | Mean total reward | Mean avg reward | Mean regret | Tail best-arm rate | Convergence t* |
|---|---:|---:|---:|---:|---:|
| epsilon-greedy (ε=0.01) | 1421.2 | 0.7106 | 58.8 | 0.497 | 9 |
| epsilon-greedy (ε=0.05) | 1453.4 | 0.7267 | 26.6 | 0.769 | 59 |
| epsilon-greedy (ε=0.10) | 1438.6 | 0.7193 | 41.4 | 0.570 | 7 |
| epsilon-greedy (ε=0.20) | 1431.5 | 0.7157 | 48.5 | 0.597 | 13 |

- **Mean regret** uses oracle reward \(T\cdot p^* - \text{total reward}\), where \(p^*\) is the best arm’s true repayment probability.
- **Tail best-arm rate** is the fraction of pulls of the true best arm in the last 10% of interactions (how strongly it exploits once learned).
- **Convergence t\*** is the first step where the running average reward reaches **95%** of the best arm’s true probability (lower is faster).

## UCB sweep (uncertainty-driven exploration)

| Setting / Method | Mean total reward | Mean avg reward | Mean regret | Tail best-arm rate | Convergence t* |
|---|---:|---:|---:|---:|---:|
| UCB1 (c=0.25) | 1451.9 | 0.7260 | 28.1 | 0.653 | 22 |
| UCB1 (c=0.50) | 1420.8 | 0.7104 | 59.2 | 0.494 | 35 |
| UCB1 (c=1.00) | 1392.9 | 0.6965 | 87.1 | 0.352 | 27 |
| UCB1 (c=2.00) | 1367.1 | 0.6835 | 112.9 | 0.266 | 27 |

- **Mean regret** uses oracle reward \(T\cdot p^* - \text{total reward}\), where \(p^*\) is the best arm’s true repayment probability.
- **Tail best-arm rate** is the fraction of pulls of the true best arm in the last 10% of interactions (how strongly it exploits once learned).
- **Convergence t\*** is the first step where the running average reward reaches **95%** of the best arm’s true probability (lower is faster).

## Final comparison (best settings + Thompson sampling)

| Method | Mean total reward | Mean avg reward | Mean regret | Tail best-arm rate | Convergence t* | Mean final ranking |
|---|---:|---:|---:|---:|---:|---|
| epsilon-greedy (best ε=0.05) | 1453.4 | 0.7267 | 26.6 | 0.769 | 59 | 5 > 3 > 3 > 2 > 1 > 1 |
| UCB1 (best c=0.25) | 1451.9 | 0.7260 | 28.1 | 0.653 | 22 | 5 > 3 > 2 > 2 > 1 > 2 |
| Thompson Sampling (Bernoulli) | 1427.2 | 0.7136 | 52.8 | 0.601 | 5 | 5 > 4 > 3 > 1 > 1 > 1 |

## Discussion guide (map to assignment prompts)

- **Total reward earned**: use the tables above and `comparison__summary.md` to state which algorithm earns most repayments.
- **Exploration efficiency**: compare regret + how quickly the best arm becomes dominant in the arm-frequency plots.
- **Exploitation effectiveness**: compare tail best-arm rates and arm-frequency concentration on the best arm.
- **Convergence speed**: compare convergence \(t^*\) and the average-reward curves approaching \(p^*\).
- **Identifying most creditworthy borrowers**: look for high selection frequency of the best arm and a ranking that places the best arm first.
- **Suitability for adaptive credit scoring**:
  - epsilon-greedy: simplest; needs tuning of ε; may waste pulls if ε too high; may get stuck if ε too low.
  - UCB: principled exploration via uncertainty; deterministic given history; often strong early learning.
  - Thompson sampling: Bayesian uncertainty; typically strong balance; produces an interpretable posterior for each segment’s repayment probability.

## Plots to cite in your submission

- `epsilon_sweep__cumulative_reward.svg`, `epsilon_sweep__average_reward.svg`, `epsilon_sweep__arm_frequencies.svg`
- `ucb_sweep__cumulative_reward.svg`, `ucb_sweep__average_reward.svg`, `ucb_sweep__arm_frequencies.svg`
- `comparison__cumulative_reward.svg`, `comparison__average_reward.svg`, `comparison__arm_frequencies.svg`
