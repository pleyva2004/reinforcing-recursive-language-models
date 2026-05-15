"""104-rlm-as-options.py

Demonstrates the RLM <-> options correspondence: build a tiny SMDP whose
high-level steps are 'invoke option' and compute the SMDP Q-function via
the hierarchical Bellman equation. Confirms that treating rlm_query calls
as options yields a coherent value-function view.

Run: python3 104-rlm-as-options.py
"""
import numpy as np


# Two states, three actions = {primitive, option_A, option_B}.
# Option A and B run multi-step internal policies and terminate with reward.
def smdp_bellman_iter(n_iter: int = 200) -> dict:
    Q = np.zeros((2, 3))
    gamma = 0.95
    # transition / reward / duration tables (toy)
    # action 0 (primitive): one step, reward 0.1, stays in state
    # action 1 (option A): 3 steps, terminal reward 1.0, transitions to state 1
    # action 2 (option B): 5 steps, terminal reward 1.5, transitions to state 0
    R = np.array([[0.1, 1.0, 1.5], [0.1, 1.0, 1.5]])
    K = np.array([[1, 3, 5], [1, 3, 5]])  # option durations
    next_s = np.array([[0, 1, 0], [1, 1, 0]])

    for _ in range(n_iter):
        Q_new = np.zeros_like(Q)
        for s in range(2):
            for a in range(3):
                ns = next_s[s, a]
                k = K[s, a]
                Q_new[s, a] = R[s, a] + (gamma ** k) * Q[ns].max()
        if np.allclose(Q, Q_new, atol=1e-10):
            Q = Q_new
            break
        Q = Q_new

    return {"Q": Q, "best_action_per_state": [int(Q[s].argmax()) for s in range(2)]}


if __name__ == "__main__":
    out = smdp_bellman_iter()
    print("SMDP Q-table (rows = states, cols = {primitive, optA, optB}):")
    print(np.round(out["Q"], 3))
    print(f"Best action per state: {out['best_action_per_state']}")
    print("\nKey property: the SMDP Bellman backup with gamma**K (K = option")
    print("duration) is the natural extension of the flat Bellman equation; this")
    print("is the value-function side of the rlm_query-as-options correspondence.")
