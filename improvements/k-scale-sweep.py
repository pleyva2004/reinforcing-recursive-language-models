"""
k-scale-sweep.py
================

Empirical $k_g$ scale-law experiment for RLM shared-policy GRPO.
Cross-references `05-improvements.tex`, section "Experimental Extensions".

We hold total compute (= G_roots * (1 + k_g)) ~constant at C = 32 child+parent
generations per gradient step, sweeping k_g in {1, 2, 4, 8} with G adjusted
inversely:

    k_g = 1 -> G = 16
    k_g = 2 -> G = 11
    k_g = 4 -> G =  6
    k_g = 8 -> G =  4

Hypothesis: there's a knee — too few children = under-utilized compute,
too many children = noisy/redundant.

Run
---
    python3 k-scale-sweep.py
"""

import json
import numpy as np

np.random.seed(0)

P_PAPERS = 8  # bigger paper set so k_g up to 8 is meaningful
A_PASSAGES = 3
LR = 0.10
N_STEPS = 150
COMPUTE_BUDGET = 32  # total parent+child gens per grad step


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def make_env(seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, A_PASSAGES, size=P_PAPERS)


def train(k_g: int, G_roots: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    correct = make_env(seed)
    parent = np.zeros(P_PAPERS)
    child = np.zeros((P_PAPERS, A_PASSAGES))
    rewards = []
    threshold = 0.7
    steps_to_threshold = None

    for step in range(N_STEPS):
        rewards_g = np.zeros(G_roots)
        rolls = []
        for g in range(G_roots):
            p_par = softmax(parent)
            # k_g without replacement; if k_g > P_PAPERS clamp
            k_eff = min(k_g, P_PAPERS)
            papers = np.random.choice(
                P_PAPERS, size=k_eff, replace=False, p=p_par / p_par.sum()
            )
            child_actions = []
            child_rewards = []
            for pid in papers:
                p_c = softmax(child[pid])
                a = np.random.choice(A_PASSAGES, p=p_c)
                child_actions.append(a)
                child_rewards.append(float(a == correct[pid]))
            r_g = float(np.mean(child_rewards))
            rewards_g[g] = r_g
            rolls.append((papers, child_actions))

        baseline = rewards_g.mean()
        std = rewards_g.std() + 1e-6
        adv_g = (rewards_g - baseline) / std

        gp = np.zeros_like(parent)
        gc = np.zeros_like(child)
        for g, (papers, child_actions) in enumerate(rolls):
            A_g = adv_g[g]
            p_p = softmax(parent)
            for pid in papers:
                d = -p_p.copy()
                d[pid] += 1.0
                gp += A_g * d
            for pid, a in zip(papers, child_actions):
                p_c = softmax(child[pid])
                d = -p_c.copy()
                d[a] += 1.0
                gc[pid] += (A_g / max(1, k_g)) * d

        parent += LR * gp / G_roots
        child += LR * gc / G_roots

        rewards.append(rewards_g.mean())
        if steps_to_threshold is None and rewards_g.mean() > threshold:
            steps_to_threshold = step

    return {
        "final_reward": float(np.mean(rewards[-10:])),
        "steps_to_threshold": steps_to_threshold if steps_to_threshold is not None else N_STEPS,
        "reward_curve": rewards,
    }


def measure() -> dict:
    sweep = {}
    schedule = []
    for k_g in [1, 2, 4, 8]:
        # G adjusted to hold parent+child compute ~constant
        G_roots = max(2, int(round(COMPUTE_BUDGET / (1 + k_g))))
        schedule.append((k_g, G_roots))
        out = train(k_g=k_g, G_roots=G_roots, seed=0)
        sweep[str(k_g)] = {
            "G_roots": G_roots,
            "final_reward": round(out["final_reward"], 4),
            "steps_to_threshold": out["steps_to_threshold"],
        }
    # Print human-readable table
    print("k_g  G_roots   final_reward   steps_to_threshold")
    print("---------------------------------------------------")
    for k_g, G in schedule:
        s = sweep[str(k_g)]
        print(f"{k_g:>3}  {G:>7}   {s['final_reward']:>12.4f}   {s['steps_to_threshold']:>4}")
    optimal_k_g = max(sweep.keys(), key=lambda k: sweep[k]["final_reward"])
    return {
        "by_k_g": sweep,
        "optimal_k_g": int(optimal_k_g),
        "compute_budget": COMPUTE_BUDGET,
    }


if __name__ == "__main__":
    out = measure()
    print()
    print(json.dumps(out, indent=2))
