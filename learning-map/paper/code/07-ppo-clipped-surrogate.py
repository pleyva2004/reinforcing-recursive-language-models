"""07-ppo-clipped-surrogate.py — numerical PPO clipped surrogate.

Computes the per-token PPO clipped surrogate for a fixed (rho, A) and shows
the asymmetric clip behaviour: positive A lets rho exceed 1+eps unboundedly
*through min*, but is capped at (1+eps)A.

Run: python3 07-ppo-clipped-surrogate.py
"""
import numpy as np


EPS = 0.2


def ppo_surrogate(rho: np.ndarray, A: float, eps: float = EPS) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    clipped = np.clip(rho, 1 - eps, 1 + eps)
    return np.minimum(rho * A, clipped * A)


if __name__ == "__main__":
    rhos = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    print("PPO clipped surrogate per token:")
    print(f"{'rho':>6}  {'A=+1':>10}  {'A=-1':>10}")
    for r in rhos:
        s_pos = ppo_surrogate(np.array([r]), +1.0).item()
        s_neg = ppo_surrogate(np.array([r]), -1.0).item()
        print(f"{r:>6.2f}  {s_pos:>10.4f}  {s_neg:>10.4f}")

    print(f"\neps = {EPS}, clip range = [{1-EPS}, {1+EPS}]")
    print("Observe: with A>0, rho>1+eps is capped at (1+eps)*A=+1.20 (no bonus)")
    print("         with A<0, rho>1+eps stays at rho*A (the min picks the smaller)")
    print("This asymmetry is what gives PPO its 'pessimistic' update.")
