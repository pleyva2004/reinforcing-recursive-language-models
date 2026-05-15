"""10-kg-averaging.py — show the 1/k_g term keeps per-root contribution
balanced.

Compares two policies: (a) raw sum over children (large branching factor
dominates), (b) 1/k_g-averaged sum (every root contributes 1 unit). Plots
per-root contribution as a function of k_g.

Run: python3 10-kg-averaging.py
"""
import numpy as np


def per_root_contribution_raw(k_g, child_loss=1.0):
    return child_loss * k_g


def per_root_contribution_balanced(k_g, child_loss=1.0):
    if k_g == 0:
        return 0.0
    return (child_loss * k_g) / k_g  # = child_loss


if __name__ == "__main__":
    ks = [0, 1, 2, 4, 8, 16, 64]
    print(f"{'k_g':>6}  {'raw sum':>12}  {'1/k averaged':>14}")
    for k in ks:
        r = per_root_contribution_raw(k)
        b = per_root_contribution_balanced(k)
        print(f"{k:>6}  {r:>12.2f}  {b:>14.2f}")

    # Also: what happens if the optimiser wants to maximise per-root reward?
    # Without the 1/k factor, just spawning more children is "rewarded" by
    # having larger contribution to the (negative-loss) objective.
    print("\nPerverse-incentive check:")
    print(" Without 1/k_g, spawning more children gives more loss-magnitude,")
    print(" which the optimiser can exploit (spawn-happy degenerate policies).")
    print(" With 1/k_g, per-root contribution is invariant to k_g.")
