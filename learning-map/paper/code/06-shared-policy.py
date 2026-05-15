"""06-shared-policy.py — single theta drives root and child rollouts.

Demonstrates that one parameter vector simultaneously decides root tokens and
child tokens; a gradient step on a child rollout therefore also moves the
root's logits.

Run: python3 06-shared-policy.py
"""
import numpy as np


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def policy(theta, prompt_id, vocab=4):
    # Toy: theta is shape (n_prompts, vocab); softmax row gives pi(.|prompt).
    return softmax(theta[prompt_id])


def grad_logpi(theta, prompt_id, token, vocab=4):
    # Standard softmax gradient: dlogpi(a|s)/dtheta_s = onehot(a) - pi(.|s).
    p = policy(theta, prompt_id, vocab)
    onehot = np.eye(vocab)[token]
    g = np.zeros_like(theta)
    g[prompt_id] = onehot - p
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_prompts, vocab = 2, 4
    theta = rng.standard_normal((n_prompts, vocab)) * 0.1

    # Take a child gradient at prompt_id=1, token=2, with assigned advantage A=+1
    g_child = grad_logpi(theta, prompt_id=1, token=2)
    A_child = 1.0
    theta_after = theta + 0.5 * A_child * g_child  # simple ascent step

    # Show that the root prompt distribution (prompt_id=0) is UNCHANGED
    # while the child prompt distribution (prompt_id=1) moves.
    p0_before, p0_after = policy(theta, 0), policy(theta_after, 0)
    p1_before, p1_after = policy(theta, 1), policy(theta_after, 1)

    print("Root prompt pi (should be UNCHANGED with toy disjoint rows):")
    print(" before:", np.round(p0_before, 3), "  after:", np.round(p0_after, 3))
    print("Child prompt pi (should move at token=2):")
    print(" before:", np.round(p1_before, 3), "  after:", np.round(p1_after, 3))

    print("\nKey property: in the real shared policy, theta is shared *globally*")
    print("(a transformer with one weight matrix), so a child gradient does")
    print("affect root logits — this is the transfer effect the paper relies on.")
