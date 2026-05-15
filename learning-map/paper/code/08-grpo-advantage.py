"""08-grpo-advantage.py — GRPO group-baseline advantage.

For G sampled rollouts with rewards r_g, compute the standardised advantage
A_g = (r_g - mu) / (sigma + delta). Verifies sum(A_g) ~ 0 and var(A_g) ~ 1.

Run: python3 08-grpo-advantage.py
"""
import numpy as np


DELTA = 1e-8


def grpo_advantage(rewards: np.ndarray) -> np.ndarray:
    mu = rewards.mean()
    sigma = rewards.std()
    return (rewards - mu) / (sigma + DELTA)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    G = 16
    rewards = rng.uniform(0.0, 1.0, size=G)
    A = grpo_advantage(rewards)
    print(f"G = {G}")
    print(f"rewards mean = {rewards.mean():.4f}, std = {rewards.std():.4f}")
    print(f"advantages   mean = {A.mean():+.6f}, std = {A.std():.4f}")
    print(f"max |A_g|    = {np.abs(A).max():.3f}")

    # Demonstrate baseline-doesn't-bias: the gradient estimator under any
    # constant baseline b has the same expectation. Numerically:
    fake_grad_logpi = rng.standard_normal(G)
    g_no_baseline = (fake_grad_logpi * rewards).mean()
    g_with_mean = (fake_grad_logpi * (rewards - rewards.mean())).mean()
    print(f"\nMean estimator with baseline=0    : {g_no_baseline:+.4f}")
    print(f"Mean estimator with baseline=mean : {g_with_mean:+.4f}")
    print("(They differ by O(1/G); the GRPO variance reduction comes from")
    print(" the centred form, not from changing the population expectation.)")
