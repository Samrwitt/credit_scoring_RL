# Bandit Algorithms for Adaptive Credit Scoring — Report

## Scenario (repeat borrower reapplication)
- **Arms (K)**: 6
- **Steps (T)**: 5000
- **Simulations**: 50
- **True repayment probabilities**: `[0.58, 0.61, 0.65, 0.69, 0.72, 0.74]`
- **Best arm (most creditworthy segment)**: arm **5** with p(repay)=**0.74**

Interpretation: each algorithm learns an estimated repayment success probability per arm. This acts as a **creditworthiness score**, used to **rank** arms and select which borrower segment/policy to allocate decisions to over time.

## Epsilon-greedy sweep (exploration vs exploitation)

| Setting / Method | Mean total reward | Mean avg reward | Mean regret | Tail best-arm rate | Convergence t* |
|---|---:|---:|---:|---:|---:|
| epsilon-greedy (ε=0.01) | 3575.7 | 0.7151 | 124.3 | 0.555 | 268 |
| epsilon-greedy (ε=0.05) | 3622.7 | 0.7245 | 77.3 | 0.708 | 96 |
| epsilon-greedy (ε=0.10) | 3614.1 | 0.7228 | 85.9 | 0.690 | 239 |
| epsilon-greedy (ε=0.20) | 3587.0 | 0.7174 | 113.0 | 0.640 | 91 |

- **Mean regret** uses oracle reward \(T\cdot p^* - \text{total reward}\), where \(p^*\) is the best arm’s true repayment probability.
- **Tail best-arm rate** is the fraction of pulls of the true best arm in the last 10% of interactions (how strongly it exploits once learned).
- **Convergence t\*** is the first step where the running average reward reaches **95%** of the best arm’s true probability (lower is faster).

## UCB sweep (uncertainty-driven exploration)

| Setting / Method | Mean total reward | Mean avg reward | Mean regret | Tail best-arm rate | Convergence t* |
|---|---:|---:|---:|---:|---:|
| UCB1 (c=0.25) | 3634.3 | 0.7269 | 65.7 | 0.718 | 16 |
| UCB1 (c=0.50) | 3594.9 | 0.7190 | 105.1 | 0.694 | 51 |
| UCB1 (c=1.00) | 3503.5 | 0.7007 | 196.5 | 0.451 | 182 |
| UCB1 (c=2.00) | 3429.0 | 0.6858 | 271.0 | 0.300 | 23 |

- **Mean regret** uses oracle reward \(T\cdot p^* - \text{total reward}\), where \(p^*\) is the best arm’s true repayment probability.
- **Tail best-arm rate** is the fraction of pulls of the true best arm in the last 10% of interactions (how strongly it exploits once learned).
- **Convergence t\*** is the first step where the running average reward reaches **95%** of the best arm’s true probability (lower is faster).

## Final comparison (best settings + Thompson sampling)

| Method | Mean total reward | Mean avg reward | Mean regret | Tail best-arm rate | Convergence t* | Mean final ranking |
|---|---:|---:|---:|---:|---:|---|
| epsilon-greedy (best ε=0.05) | 3622.7 | 0.7245 | 77.3 | 0.708 | 96 | 5 > 4 > 3 > 2 > 1 > 0 |
| UCB1 (best c=0.25) | 3634.3 | 0.7269 | 65.7 | 0.718 | 16 | 5 > 4 > 3 > 2 > 1 > 0 |
| Thompson Sampling (Bernoulli) | 3626.4 | 0.7253 | 73.6 | 0.840 | 6 | 5 > 4 > 3 > 2 > 1 > 0 |

## Discussion and Analysis

The experiment demonstrates how exploration–exploitation algorithms can be applied to an adaptive credit scoring system for repeat borrowers. In this credit scoring scenario, each arm represents a borrower segment, applicant category, or credit policy option. A reward of 1 represents successful repayment, while a reward of 0 represents default. Therefore, the cumulative reward shows the total number of successful repayments obtained by each algorithm over time, and the average reward shows how well the algorithm learns to select creditworthy borrower groups.

### Total Reward Earned by Each Algorithm

The total reward results show how effectively each algorithm selected borrower segments with higher repayment probabilities. Epsilon-greedy, UCB, and Thompson Sampling all improved their performance over time as they collected more repayment feedback. However, their learning behavior differed because each algorithm handles exploration differently.

Epsilon-greedy performed depending on the chosen epsilon value. A small epsilon encouraged the algorithm to exploit the borrower segment currently believed to be best, while a larger epsilon caused more random exploration. When epsilon was too low, the algorithm risked selecting a suboptimal borrower segment early and continuing to exploit it. When epsilon was too high, the algorithm continued exploring too much, selecting weaker borrower groups even after it had enough evidence about the best segment. Therefore, moderate epsilon values usually provide a better balance between exploration and exploitation.

The UCB algorithm performed strongly because it selected borrower segments based on both their estimated repayment performance and the uncertainty around that estimate. Instead of exploring randomly, UCB gave additional priority to less-tested arms. This helped the algorithm collect useful information early and gradually shift toward the most promising borrower segment.

Thompson Sampling also performed well because it used a probabilistic belief about each borrower segment’s repayment success rate. By updating the Beta distribution after every repayment or default outcome, Thompson Sampling naturally balanced exploration and exploitation. It explored uncertain borrower segments when there was still a chance they might be good, but increasingly exploited the most creditworthy segment as evidence accumulated.

### Exploration Efficiency

Exploration efficiency refers to how well an algorithm gathers information about borrower segments without wasting too many interactions on low-quality groups. Epsilon-greedy explores randomly, so some exploration is inefficient because it may continue selecting poor borrower groups even after they appear unlikely to repay successfully. Higher epsilon values increase this problem because the algorithm keeps making random selections throughout the experiment.

UCB is more efficient than simple random exploration because its exploration is guided by uncertainty. Borrower segments that have been selected fewer times receive a higher confidence bonus, making them more likely to be tested. Once the algorithm becomes more confident about a segment’s repayment behavior, the uncertainty bonus decreases. This makes UCB suitable for situations where the credit scoring system needs to learn quickly while still controlling risk.

Thompson Sampling is also exploration-efficient because it explores according to uncertainty in the posterior belief. If an arm has limited repayment history, its sampled repayment probability may sometimes be high, encouraging exploration. As more repayment observations are collected, the posterior becomes more stable, and the algorithm naturally focuses on the borrower segment with the highest estimated creditworthiness.

### Exploitation Effectiveness

Exploitation effectiveness measures how well the algorithm uses what it has learned to select the best borrower segment. Epsilon-greedy can exploit effectively when epsilon is small, but it may exploit the wrong segment if early observations are misleading. It also continues to explore randomly, so even after learning the best arm, it still spends some interactions on weaker segments.

UCB becomes increasingly exploitative after it reduces uncertainty about the borrower groups. At the beginning, it tests all arms to estimate their repayment quality. Later, it concentrates more selections on the borrower segment with the highest estimated repayment success. This makes UCB effective when enough interactions are available for learning.

Thompson Sampling tends to exploit effectively because its posterior beliefs become more concentrated over time. Once the algorithm has strong evidence that one borrower segment is more creditworthy, it selects that segment more frequently. At the same time, it does not completely ignore uncertainty, which allows it to adapt if repayment patterns are not obvious at the beginning.

### Convergence Speed

Convergence speed refers to how quickly each algorithm identifies and repeatedly selects the most creditworthy borrower segment. Epsilon-greedy convergence depends heavily on epsilon. A very low epsilon may converge quickly, but not always to the correct arm. A high epsilon may delay convergence because the algorithm continues to explore randomly. Therefore, epsilon-greedy requires careful tuning.

UCB usually converges more systematically because it first explores uncertain arms and then increasingly selects the arm with the best observed repayment performance. Its convergence is driven by the confidence-bound formula, which reduces unnecessary exploration over time.

Thompson Sampling often converges smoothly because it updates posterior beliefs after each repayment result. As evidence accumulates, the posterior probability of the best borrower segment becomes stronger, and the algorithm selects it more frequently. This makes Thompson Sampling useful in credit scoring because it can learn from repayment outcomes while still accounting for uncertainty.

### Ability to Identify the Most Creditworthy Repeat Borrowers

The arm-selection frequency plot is important for evaluating whether the algorithms successfully identified the most creditworthy borrower group. A good algorithm should increasingly select the arm with the highest true repayment probability. The final ranking of arms also shows whether the estimated creditworthiness scores match the actual repayment probabilities.

Epsilon-greedy can identify the best borrower group, especially with a well-chosen epsilon value, but it is sensitive to randomness and parameter selection. If early repayment outcomes are misleading, the algorithm may overestimate a weaker segment.

UCB is effective at identifying creditworthy borrowers because it compares borrower segments using both observed repayment success and uncertainty. This reduces the chance that a segment is ignored too early.

Thompson Sampling is particularly suitable for identifying creditworthy repeat borrowers because it directly models uncertainty in repayment probability. Its posterior estimates can be interpreted as updated beliefs about borrower quality, which is useful for adaptive credit scoring.

### Suitability for an Adaptive Credit Scoring System

For an adaptive credit scoring system, the algorithm should learn from repeat borrower repayment behavior, update creditworthiness estimates, rank applicants or borrower groups, and improve selection decisions over time.

Epsilon-greedy is simple and easy to implement. It is suitable as a baseline method, but it requires careful tuning of epsilon. If epsilon is too large, the lender may take unnecessary risks by repeatedly selecting less creditworthy borrower groups. If epsilon is too small, the system may fail to discover better borrower segments.

UCB is more suitable than epsilon-greedy when the system needs a more structured exploration strategy. It is useful because it considers uncertainty and encourages the algorithm to test borrower segments that have not been observed enough. This makes it appropriate for credit scoring environments where the system must learn responsibly from limited data.

Thompson Sampling is highly suitable for adaptive credit scoring because it provides a flexible probabilistic learning framework. It updates beliefs after every repayment outcome and balances exploration and exploitation naturally. It is especially useful when borrower repayment behavior is uncertain and when the system needs to make decisions based on both expected repayment and confidence in the estimate.

Overall, the results show that exploration–exploitation algorithms can support credit scoring for repeat borrowers by continuously learning from repayment behavior. Epsilon-greedy provides a simple baseline, UCB offers uncertainty-driven exploration, and Thompson Sampling provides a strong probabilistic approach. Among the three, UCB and Thompson Sampling are generally more suitable for adaptive credit scoring because they explore more intelligently and are better able to identify the most creditworthy borrower segments over time.
