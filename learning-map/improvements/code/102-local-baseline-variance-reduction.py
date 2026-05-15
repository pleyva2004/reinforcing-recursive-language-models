"""102-local-baseline-variance-reduction.py

Toy 2-arm RLM bandit. Each parent samples 2 children, each child gets a
state s in {0, 1}. Baseline b(s) = E[A_g | s]. Show that subtracting
b(s) reduces estimator variance without changing E[grad].

Run: python3 102-local-baseline-variance-reduction.py
"""
import numpy as np


def simulate_with_baseline(n_steps: int, use_baseline: bool, seed: int):
    rng = np.random.default_rng(seed)
    grads = []
    for _ in range(n_steps):
        # parent reward: depends on hidden state; A_g standardised across batch later
        A_g = rng.choice([+1.0, -1.0])
        # child state s: 0 = "easy", 1 = "hard". On easy, A_g tends positive.
        s = rng.choice([0, 1])
        if s == 0:
            local_A = A_g + 0.5
        else:
            local_A = A_g - 0.5
        if use_baseline:
            # b(s) = expected local_A given s -> mean shift only
            b = +0.5 if s == 0 else -0.5
            adv = local_A - b  # = A_g  (centred)
        else:
            adv = local_A
        # toy "score function": +1 (gradient direction is fixed for this synthetic)
        grads.append(adv)
    grads = np.array(grads)
    return grads.mean(), grads.var()


def measure() -> dict:
    m_no, v_no = simulate_with_baseline(5000, False, 0)
    m_yes, v_yes = simulate_with_baseline(5000, True, 0)
    return {
        "mean_no_baseline": float(m_no),
        "mean_with_baseline": float(m_yes),
        "var_no_baseline": float(v_no),
        "var_with_baseline": float(v_yes),
        "ratio": float(v_yes / max(v_no, 1e-12)),
    }


if __name__ == "__main__":
    out = measure()
    for k, v in out.items():
        print(f"{k:>22}: {v:.4f}")
    print("\nKey property: ratio < 1.0 -> baseline reduced variance.")
    print("Means coincide -> estimator remains unbiased.")
