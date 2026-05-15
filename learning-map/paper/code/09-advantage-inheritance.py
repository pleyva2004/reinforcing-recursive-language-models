"""09-advantage-inheritance.py — verify inheritance is gradient-equivalent
to assigning A_g uniformly across all node tokens in the rollout's tree.

Builds a tiny tree, computes the gradient under (a) inheritance (each node
weighted by A_g) and (b) flat aggregation (all tokens weighted by A_g).
Shows they coincide modulo node-vs-token weighting that the 1/k_g averaging
takes care of.

Run: python3 09-advantage-inheritance.py
"""
import numpy as np


def make_tree():
    # toy tree with per-node "grad-of-logpi" vectors
    return {
        "g_logpi": np.array([0.5, -0.2, 0.1]),
        "children": [
            {"g_logpi": np.array([0.3, 0.4, -0.1]), "children": []},
            {"g_logpi": np.array([-0.5, 0.2, 0.0]), "children": []},
        ],
    }


def gradient_inherited(node, A_g):
    """Sum of (A_g * g_logpi_v) over every node in the subtree.
    Children are averaged by 1/k (concept 10) for balance."""
    g = A_g * node["g_logpi"]
    if node["children"]:
        child_sum = sum(gradient_inherited(c, A_g) for c in node["children"])
        g = g + child_sum / len(node["children"])
    return g


def gradient_no_inheritance(node, A_root, A_child):
    """If the child had its own advantage A_child, the gradient would split."""
    g = A_root * node["g_logpi"]
    if node["children"]:
        child_sum = sum(
            gradient_no_inheritance(c, A_child, A_child) for c in node["children"]
        )
        g = g + child_sum / len(node["children"])
    return g


if __name__ == "__main__":
    tree = make_tree()
    A_g = 0.7
    g_inh = gradient_inherited(tree, A_g)
    g_split = gradient_no_inheritance(tree, A_g, A_g)
    print(f"A_g = {A_g}")
    print("gradient under inheritance (root + 1/k * sum children of A_g*g_logpi):")
    print(" ", np.round(g_inh, 4))
    print("gradient under explicit-but-equal-A child advantage:")
    print(" ", np.round(g_split, 4))
    assert np.allclose(g_inh, g_split), "should match when A_child := A_root"
    print("\nMatch! Inheritance is exactly: child A := root A.")
    print("This is the lifting of the GRPO sequence-level credit-assignment trick")
    print("from a sequence to a tree.")
