"""01-rlm-defn.py — minimal RLM definition.

Builds a tiny Recursive Language Model whose 'tokens' are characters from
{a, b, q, F} where 'q' opens a recursive child rollout and 'F' terminates the
current rollout. Demonstrates the tree structure produced by sampling.

Run: python3 01-rlm-defn.py
"""
import random


def rollout(depth=0, max_depth=3, rng=None):
    """Sample a single RLM rollout. 'q' spawns a child; 'F' finalises."""
    rng = rng or random.Random()
    seq = []
    children = []
    for _ in range(8):
        if depth >= max_depth:
            tok = rng.choice(["a", "b", "F"])
        else:
            tok = rng.choice(["a", "b", "q", "F"])
        seq.append(tok)
        if tok == "q":
            children.append(rollout(depth + 1, max_depth, rng))
        if tok == "F":
            break
    return {"tokens": seq, "children": children}


def pretty(node, indent=0):
    print(" " * indent + "node tokens=" + "".join(node["tokens"]))
    for c in node["children"]:
        pretty(c, indent + 2)


if __name__ == "__main__":
    rng = random.Random(0)
    root = rollout(rng=rng)
    print("Sampled RLM tree (root + recursive children):")
    pretty(root)
    print("\nKey property: each node is itself a sequence of tokens drawn from")
    print("the same policy; 'q' tokens are the recursive ingredient.")
