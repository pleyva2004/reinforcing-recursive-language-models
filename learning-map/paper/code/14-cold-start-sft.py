"""14-cold-start-sft.py — toy SFT step for harness syntax.

Demonstrates the role of SFT: shifts the policy mass towards a 'correct
syntax' next-token distribution. We train a tiny softmax classifier with NTP
loss on a few demo (prompt, syntax-correct token) pairs and show the
probability of the correct syntax token rises.

Run: python3 14-cold-start-sft.py
"""
import numpy as np


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    vocab = 5  # tokens: 0=FINAL, 1=FINAL_VAR, 2=rlm_query, 3=garbage, 4=other
    n_prompts = 3

    # one-hot 'correct token' label per prompt — these are syntax-essentials
    labels = np.array([0, 1, 2])  # prompt 0 must end with FINAL, etc.
    theta = rng.standard_normal((n_prompts, vocab)) * 0.1

    # Initial pi(correct | prompt)
    p0 = softmax(theta)
    print("Before SFT, P(correct token | prompt):")
    for i in range(n_prompts):
        print(f"  prompt {i}: {p0[i, labels[i]]:.3f}")

    # SFT: 200 steps of NTP loss gradient
    lr = 0.5
    for _ in range(200):
        p = softmax(theta)
        for i in range(n_prompts):
            grad = p[i].copy()
            grad[labels[i]] -= 1.0
            theta[i] -= lr * grad

    p1 = softmax(theta)
    print("\nAfter SFT, P(correct token | prompt):")
    for i in range(n_prompts):
        print(f"  prompt {i}: {p1[i, labels[i]]:.3f}")
    print("\nKey property: SFT alone moves syntax-correct mass from ~0.2 to >0.99.")
    print("Without it, RL has 0 positive reward to amplify (pass@16 = 0 at 4B).")
