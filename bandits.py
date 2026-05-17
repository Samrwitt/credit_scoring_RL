from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

RewardDist = Union[float, Callable[[np.random.Generator], float]]


def _validate_num_arms(num_arms: int) -> None:
    if not isinstance(num_arms, int) or num_arms <= 0:
        raise ValueError("num_arms must be a positive integer")


def _as_distributions(num_arms: int, true_reward_dists: Sequence[RewardDist]) -> List[RewardDist]:
    _validate_num_arms(num_arms)
    if len(true_reward_dists) != num_arms:
        raise ValueError(f"true_reward_dists must have length {num_arms}")
    return list(true_reward_dists)


def _sample_reward(dist: RewardDist, rng: np.random.Generator) -> float:
    """
    Sample a reward from an arm's "true" distribution.

    Supported encodings:
    - float in [0, 1]: interpreted as Bernoulli success probability (reward in {0, 1})
    - callable(rng) -> float: user-defined reward sampler
    """
    if callable(dist):
        r = float(dist(rng))
        return r

    p = float(dist)
    if not (0.0 <= p <= 1.0):
        raise ValueError("Float reward distributions must be Bernoulli probabilities in [0, 1]")
    return float(rng.random() < p)


@dataclass(frozen=True)
class BanditRun:
    total_reward: float
    rewards: np.ndarray  # shape: (steps,)
    actions: np.ndarray  # shape: (steps,) integer arm indices
    cumulative_reward: np.ndarray  # shape: (steps,)
    average_reward: np.ndarray  # shape: (steps,)

    # Learned quantities at end of run (for "creditworthiness estimation" and ranking)
    counts: np.ndarray  # shape: (num_arms,)
    value_estimates: np.ndarray  # shape: (num_arms,) frequentist mean reward estimate
    posterior_alpha: Optional[np.ndarray] = None  # Thompson only, shape: (num_arms,)
    posterior_beta: Optional[np.ndarray] = None  # Thompson only, shape: (num_arms,)


def epsilon_greedy(
    num_arms: int,
    true_reward_dists: Sequence[RewardDist],
    steps: int,
    epsilon: float,
    *,
    seed: Optional[int] = None,
    init_value: float = 0.0,
) -> BanditRun:
    """
    Epsilon-greedy multi-armed bandit.

    Args:
        num_arms: number of arms (e.g., borrower segments or policy variants).
        true_reward_dists: per-arm reward distribution. Use float p for Bernoulli(p)
            repayment success, or a callable(rng)->reward for custom distributions.
        steps: number of borrower interactions.
        epsilon: exploration probability in [0, 1].
        seed: RNG seed for reproducibility.
        init_value: initial estimated value for each arm (optimistic init allowed).
    """
    _validate_num_arms(num_arms)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not (0.0 <= float(epsilon) <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")

    dists = _as_distributions(num_arms, true_reward_dists)
    rng = np.random.default_rng(seed)

    q = np.full(num_arms, float(init_value), dtype=float)  # value estimates
    n = np.zeros(num_arms, dtype=int)  # counts

    actions = np.empty(steps, dtype=int)
    rewards = np.empty(steps, dtype=float)

    for t in range(steps):
        explore = rng.random() < epsilon
        if explore:
            a = int(rng.integers(num_arms))
        else:
            # deterministic argmax tie-break: random among max arms
            max_q = np.max(q)
            candidates = np.flatnonzero(q == max_q)
            a = int(rng.choice(candidates))

        r = _sample_reward(dists[a], rng)

        n[a] += 1
        q[a] += (r - q[a]) / n[a]

        actions[t] = a
        rewards[t] = r

    cumulative = np.cumsum(rewards)
    avg = cumulative / (np.arange(steps) + 1)
    return BanditRun(
        total_reward=float(cumulative[-1]),
        rewards=rewards,
        actions=actions,
        cumulative_reward=cumulative,
        average_reward=avg,
        counts=n.copy(),
        value_estimates=q.copy(),
    )


def ucb1(
    num_arms: int,
    true_reward_dists: Sequence[RewardDist],
    steps: int,
    exploration: float,
    *,
    seed: Optional[int] = None,
    init_value: float = 0.0,
) -> BanditRun:
    """
    UCB1 algorithm (Auer et al.) with tunable exploration coefficient.

    Selection rule at time t (1-indexed):
        argmax_a [ Q_a + exploration * sqrt( (2 ln t) / N_a ) ]

    Args:
        exploration: >= 0. Higher means more exploration.
    """
    _validate_num_arms(num_arms)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if float(exploration) < 0.0:
        raise ValueError("exploration must be >= 0")

    dists = _as_distributions(num_arms, true_reward_dists)
    rng = np.random.default_rng(seed)

    q = np.full(num_arms, float(init_value), dtype=float)
    n = np.zeros(num_arms, dtype=int)

    actions = np.empty(steps, dtype=int)
    rewards = np.empty(steps, dtype=float)

    # Pull each arm once first (or as many steps as available)
    t = 0
    for a in range(min(num_arms, steps)):
        r = _sample_reward(dists[a], rng)
        n[a] += 1
        q[a] += (r - q[a]) / n[a]
        actions[t] = a
        rewards[t] = r
        t += 1

    for idx in range(t, steps):
        tt = idx + 1  # 1-indexed time for log
        bonus = exploration * np.sqrt((2.0 * np.log(tt)) / n)
        ucb = q + bonus
        max_ucb = np.max(ucb)
        candidates = np.flatnonzero(ucb == max_ucb)
        a = int(rng.choice(candidates))

        r = _sample_reward(dists[a], rng)
        n[a] += 1
        q[a] += (r - q[a]) / n[a]

        actions[idx] = a
        rewards[idx] = r

    cumulative = np.cumsum(rewards)
    avg = cumulative / (np.arange(steps) + 1)
    return BanditRun(
        total_reward=float(cumulative[-1]),
        rewards=rewards,
        actions=actions,
        cumulative_reward=cumulative,
        average_reward=avg,
        counts=n.copy(),
        value_estimates=q.copy(),
    )


def thompson_sampling_bernoulli(
    num_arms: int,
    true_ps: Sequence[float],
    steps: int,
    *,
    seed: Optional[int] = None,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> BanditRun:
    """
    Thompson sampling for Bernoulli bandits using Beta posteriors.

    Args:
        true_ps: per-arm repayment success probabilities.
        prior_alpha/prior_beta: Beta prior parameters (>= 0, typically 1 for uniform).
    """
    _validate_num_arms(num_arms)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if len(true_ps) != num_arms:
        raise ValueError(f"true_ps must have length {num_arms}")
    if prior_alpha < 0.0 or prior_beta < 0.0:
        raise ValueError("prior_alpha/prior_beta must be >= 0")

    ps = np.asarray(true_ps, dtype=float)
    if np.any((ps < 0.0) | (ps > 1.0)):
        raise ValueError("true_ps values must be in [0, 1]")

    rng = np.random.default_rng(seed)

    alpha = np.full(num_arms, float(prior_alpha), dtype=float)
    beta = np.full(num_arms, float(prior_beta), dtype=float)

    actions = np.empty(steps, dtype=int)
    rewards = np.empty(steps, dtype=float)

    for t in range(steps):
        theta = rng.beta(alpha, beta)
        max_theta = np.max(theta)
        candidates = np.flatnonzero(theta == max_theta)
        a = int(rng.choice(candidates))

        r = float(rng.random() < ps[a])

        alpha[a] += r
        beta[a] += 1.0 - r

        actions[t] = a
        rewards[t] = r

    cumulative = np.cumsum(rewards)
    avg = cumulative / (np.arange(steps) + 1)
    return BanditRun(
        total_reward=float(cumulative[-1]),
        rewards=rewards,
        actions=actions,
        cumulative_reward=cumulative,
        average_reward=avg,
        counts=np.bincount(actions.astype(int), minlength=num_arms),
        value_estimates=(alpha / (alpha + beta)).copy(),
        posterior_alpha=alpha.copy(),
        posterior_beta=beta.copy(),
    )


def arm_selection_frequencies(actions: np.ndarray, num_arms: int) -> np.ndarray:
    _validate_num_arms(num_arms)
    counts = np.bincount(actions.astype(int), minlength=num_arms)
    return counts / max(1, actions.size)


def rank_arms_by_values(values: np.ndarray) -> np.ndarray:
    """Return arm indices sorted by creditworthiness scores (descending)."""
    return np.argsort(-np.asarray(values, dtype=float), kind="stable")


def rank_arms_by_estimated_creditworthiness(run: BanditRun) -> np.ndarray:
    """
    Returns arm indices sorted by estimated creditworthiness (descending).

    In this assignment framing, the per-arm expected reward is interpreted as
    an estimated repayment success probability / creditworthiness score.
    """
    return rank_arms_by_values(run.value_estimates)

