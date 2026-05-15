"""
toy_recursive_bandit.py
=======================

Math-clean demonstration of the Recursive Language Models (RLM) GRPO loss on a
tree-structured contextual bandit.

A SINGLE shared softmax policy plays two roles in the recursive loop:
  - Parent (decomposer): given state "root", pick k_g of P papers to dispatch.
  - Child  (sub-agent):  given a dispatched paper p, pick 1 of A passages.

Each child rollout y_{g,i} INHERITS its root's group-relative advantage A_g,
and the child loss is averaged with a 1/k_g factor so a root that spawns more
children does not dominate the gradient. This implements eq. (full per-root loss)
from 02-math-deep-dive.md:

    L_g^full(theta) = L_node(y_g, A_g; theta)
                    + (1/k_g) * sum_{i=1..k_g} L_node(y_{g,i}, A_g; theta)

with each L_node a PPO-clipped surrogate (epsilon=0.2). We snapshot theta_old
once per training step, then optimise theta against it for one inner step
(matches GRPO's typical "1 inner step per outer iteration" simplification).

Run:  python3 toy_recursive_bandit.py
"""

import math
import time
import numpy as np

np.random.seed(0)

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
P_PAPERS = 3        # number of papers
A_PASSAGES = 4      # passages per paper
TARGET_PASSAGE = np.array([0, 2, 1])   # the one correct passage in each paper
# Reward structure: when the parent dispatches a subset S of papers, and each
# dispatched paper p returns the child's chosen passage c_p, the root reward is
# the fraction of dispatched papers whose child picked the correct passage.
# This couples parent and child: the parent should learn to dispatch papers
# its children can solve; the children should learn the correct passage.

# ----------------------------------------------------------------------------
# Policy parameters (single shared policy, two heads)
# ----------------------------------------------------------------------------
# Parent head: logits over P papers (we sample top-k_g without replacement).
# Child  head: per-paper logits over A passages.
parent_logits = np.zeros(P_PAPERS, dtype=np.float64)
child_logits = np.zeros((P_PAPERS, A_PASSAGES), dtype=np.float64)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def parent_sample_topk(rng, logits, k):
    """Sequential softmax sampling without replacement -> ordered list of k papers.

    Returns (chosen list, log_prob of the joint sequence under current logits).
    log p(s_1..s_k) = sum_t log softmax(logits_remaining)[s_t]
    """
    remaining = list(range(len(logits)))
    chosen = []
    logp = 0.0
    for _ in range(k):
        sub_logits = logits[remaining]
        p = softmax(sub_logits)
        idx_in_remaining = int(rng.choice(len(remaining), p=p))
        logp += float(np.log(p[idx_in_remaining] + 1e-12))
        chosen.append(remaining[idx_in_remaining])
        remaining.pop(idx_in_remaining)
    return chosen, logp


def parent_logprob(logits, chosen):
    """Recompute log p(chosen sequence) under the given parent logits."""
    remaining = list(range(len(logits)))
    logp = 0.0
    for s in chosen:
        sub_logits = logits[remaining]
        p = softmax(sub_logits)
        idx = remaining.index(s)
        logp += float(np.log(p[idx] + 1e-12))
        remaining.pop(idx)
    return logp


def child_sample(rng, logits_paper):
    p = softmax(logits_paper)
    a = int(rng.choice(len(p), p=p))
    return a, float(np.log(p[a] + 1e-12))


def child_logprob(logits_paper, action):
    p = softmax(logits_paper)
    return float(np.log(p[action] + 1e-12))


# ----------------------------------------------------------------------------
# Numerical gradient of log pi(a) w.r.t. logits, for the simple categorical heads
# (cleaner than autograd for a 12-parameter policy).
# d log softmax(z)[a] / d z_j = (a==j) - softmax(z)[j]
# ----------------------------------------------------------------------------
def grad_logprob_categorical(logits, action):
    p = softmax(logits)
    g = -p.copy()
    g[action] += 1.0
    return g  # same shape as logits


def grad_logprob_parent_seq(logits, chosen):
    """Sum of per-step gradients for sequential top-k sampling without replacement."""
    g_full = np.zeros_like(logits)
    remaining = list(range(len(logits)))
    for s in chosen:
        sub_logits = logits[remaining]
        p_sub = softmax(sub_logits)
        idx = remaining.index(s)
        # gradient of log p_sub[idx] w.r.t. sub_logits = e_idx - p_sub
        g_sub = -p_sub.copy()
        g_sub[idx] += 1.0
        # scatter back
        for j, r in enumerate(remaining):
            g_full[r] += g_sub[j]
        remaining.pop(idx)
    return g_full


# ----------------------------------------------------------------------------
# PPO-clipped surrogate (per node)
#   ratio = exp(log pi_new - log pi_old)
#   L_node = -mean( min( ratio*A, clip(ratio, 1-eps, 1+eps)*A ) )
# We compute it analytically for a single (node, A) pair: returns L and dL/dlogits.
# ----------------------------------------------------------------------------
EPS = 0.2


def ppo_loss_and_grad(logits, action, logp_old, advantage, kind, chosen=None):
    """Return (scalar loss, grad_w.r.t._logits_block).

    kind = "categorical": logits shape (A,), action int
    kind = "parent_seq": logits shape (P,), chosen=list of ints
    """
    if kind == "categorical":
        logp_new = child_logprob(logits, action)
        d_logp = grad_logprob_categorical(logits, action)
    elif kind == "parent_seq":
        logp_new = parent_logprob(logits, chosen)
        d_logp = grad_logprob_parent_seq(logits, chosen)
    else:
        raise ValueError(kind)

    ratio = math.exp(logp_new - logp_old)
    clipped = max(1.0 - EPS, min(1.0 + EPS, ratio))
    unclipped_term = ratio * advantage
    clipped_term = clipped * advantage
    # PPO uses the MIN of the two surrogate objectives; we are MINIMISING the
    # negative surrogate, so loss = -min(unclipped, clipped).
    if unclipped_term < clipped_term:
        chosen_term = unclipped_term
        # d(ratio * A) / d logits = ratio * A * d_logp
        d_chosen = ratio * advantage * d_logp
    else:
        # clipped branch was selected: gradient is zero unless we are inside
        # the clip interval (i.e. unclipped == clipped means we are inside).
        # When we are at the boundary, treat grad as zero (standard PPO).
        if 1.0 - EPS < ratio < 1.0 + EPS:
            chosen_term = unclipped_term  # equals clipped_term inside interval
            d_chosen = ratio * advantage * d_logp
        else:
            chosen_term = clipped_term
            d_chosen = np.zeros_like(d_logp)
    loss = -chosen_term
    grad = -d_chosen
    return loss, grad


# ----------------------------------------------------------------------------
# Rollout + reward
# ----------------------------------------------------------------------------
def rollout(rng, k_g):
    """One root rollout. Returns dict with everything needed for the loss."""
    chosen, parent_logp_old = parent_sample_topk(rng, parent_logits, k_g)
    children = []
    correct = 0
    for paper in chosen:
        a, child_logp_old = child_sample(rng, child_logits[paper])
        children.append(dict(paper=paper, action=a, logp_old=child_logp_old))
        if a == TARGET_PASSAGE[paper]:
            correct += 1
    reward = correct / k_g  # in [0, 1]
    return dict(chosen=chosen, parent_logp_old=parent_logp_old,
                children=children, reward=reward)


# ----------------------------------------------------------------------------
# Training loop (GRPO with recursive child loss)
# ----------------------------------------------------------------------------
def train(steps=200, G=8, k_g=2, lr=0.05, log_every=20):
    global parent_logits, child_logits
    rng = np.random.default_rng(0)
    history = []
    for step in range(steps):
        # 1. Sample G root rollouts under theta_old (= current params before update)
        rollouts = [rollout(rng, k_g) for _ in range(G)]
        rewards = np.array([r["reward"] for r in rollouts])
        mu = rewards.mean()
        sd = rewards.std() + 1e-8
        advantages = (rewards - mu) / sd

        # 2. Compute the per-rollout full loss + accumulate gradients
        d_parent = np.zeros_like(parent_logits)
        d_child = np.zeros_like(child_logits)
        total_loss = 0.0

        for r, A in zip(rollouts, advantages):
            # Root (parent) PPO term
            l_parent, g_parent = ppo_loss_and_grad(
                parent_logits, action=None, logp_old=r["parent_logp_old"],
                advantage=A, kind="parent_seq", chosen=r["chosen"],
            )
            d_parent += g_parent
            total_loss += l_parent

            # Children PPO terms with 1/k_g averaging (the balancing trick).
            child_loss_sum = 0.0
            child_grad_sum = np.zeros_like(child_logits)
            for c in r["children"]:
                paper = c["paper"]
                l_c, g_c = ppo_loss_and_grad(
                    child_logits[paper], action=c["action"],
                    logp_old=c["logp_old"], advantage=A, kind="categorical",
                )
                child_loss_sum += l_c
                child_grad_sum[paper] += g_c
            d_child += child_grad_sum / k_g
            total_loss += child_loss_sum / k_g

        # Mean over G (matches "every root contributes equally to the batch")
        d_parent /= G
        d_child /= G
        total_loss /= G

        # 3. Plain SGD step
        parent_logits = parent_logits - lr * d_parent
        child_logits = child_logits - lr * d_child

        history.append(dict(step=step, reward=float(rewards.mean()),
                            loss=float(total_loss)))
        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:4d}  mean_reward {rewards.mean():.3f}  "
                  f"loss {total_loss:+.4f}")
    return history


def policy_entropy():
    p_parent = softmax(parent_logits)
    h_parent = -(p_parent * np.log(p_parent + 1e-12)).sum()
    p_child = softmax(child_logits, axis=-1)
    h_child = -(p_child * np.log(p_child + 1e-12)).sum(axis=-1).mean()
    return float(h_parent), float(h_child)


def summary_table():
    print("\n=== Final policy ===")
    p_parent = softmax(parent_logits)
    print("Parent paper distribution: " +
          " ".join(f"p{p}={pp:.3f}" for p, pp in enumerate(p_parent)))
    p_child = softmax(child_logits, axis=-1)
    print("Child passage distributions:")
    for p in range(P_PAPERS):
        marker = ["  "] * A_PASSAGES
        marker[TARGET_PASSAGE[p]] = "* "
        row = "  ".join(f"{m}{q:.3f}" for m, q in zip(marker, p_child[p]))
        print(f"  paper {p} (target=passage {TARGET_PASSAGE[p]}): {row}")


def main():
    print("=" * 64)
    print("toy_recursive_bandit.py  — RLM-GRPO on a tree contextual bandit")
    print("=" * 64)
    print(f"P_PAPERS={P_PAPERS}  A_PASSAGES={A_PASSAGES}  "
          f"TARGET_PASSAGE={TARGET_PASSAGE.tolist()}")
    print(f"G=8  k_g=2  lr=0.20  PPO eps={EPS}")
    print()

    # Baseline reward at random init
    rng0 = np.random.default_rng(1)
    init_rewards = np.array([rollout(rng0, k_g=2)["reward"] for _ in range(64)])
    init_reward = float(init_rewards.mean())
    print(f"initial (uniform) mean reward over 64 rollouts: {init_reward:.3f}")

    t0 = time.time()
    # Note: reward = fraction of dispatched papers whose child found the
    # target passage. Since every paper has a target, the parent is roughly
    # indifferent between papers and the dominant signal trains the children.
    # We use a slightly higher lr (0.2) so the child heads concentrate
    # cleanly within 200 steps; final reward should approach 1.0.
    hist = train(steps=200, G=8, k_g=2, lr=0.2, log_every=20)
    elapsed = time.time() - t0
    print(f"\ntraining done in {elapsed:.2f}s")

    final_reward = float(np.mean([h["reward"] for h in hist[-10:]]))
    h_parent, h_child = policy_entropy()
    print(f"\n=== Validation ===")
    print(f"initial mean reward (random policy) : {init_reward:.3f}")
    print(f"final mean reward (last 10 steps)   : {final_reward:.3f}")
    print(f"parent entropy (max log P={math.log(P_PAPERS):.3f})        : "
          f"{h_parent:.3f}")
    print(f"child entropy avg (max log A={math.log(A_PASSAGES):.3f})  : "
          f"{h_child:.3f}")
    summary_table()

    # Plot reward curve (table fallback if matplotlib unavailable)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [h["step"] for h in hist]
        rewards = [h["reward"] for h in hist]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(steps, rewards, label="mean reward (G=8)")
        ax.axhline(1.0, color="g", ls="--", alpha=0.5, label="optimum")
        ax.set_xlabel("training step")
        ax.set_ylabel("reward")
        ax.set_title("Recursive GRPO on toy contextual bandit")
        ax.legend()
        out = "/tmp/toy_bandit_reward.png"
        fig.tight_layout()
        fig.savefig(out)
        print(f"\nwrote reward curve to {out}")
    except Exception as e:
        print(f"\nmatplotlib unavailable ({e}); reward table:")
        for i in range(0, len(hist), 20):
            print(f"  step {hist[i]['step']:4d}  reward {hist[i]['reward']:.3f}")

    # Return a one-line machine summary the verifier can grep
    print(f"\nRESULT init_reward={init_reward:.4f} "
          f"final_reward={final_reward:.4f} runtime_s={elapsed:.2f}")
    return init_reward, final_reward, elapsed


if __name__ == "__main__":
    main()
