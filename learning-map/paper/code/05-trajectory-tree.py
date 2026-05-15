"""05-trajectory-tree.py — enumerate a finite RLM trajectory tree.

Builds a tiny RLM tree by hand and traverses it to confirm the structural
invariants (root has reward, children don't; total log-prob factorises over
nodes).

Run: python3 05-trajectory-tree.py
"""
import math


def make_tree():
    # toy log-pi values per node — pretend each token has fixed logprob
    return {
        "id": "root",
        "tokens": ["a", "q", "b", "F"],
        "logprob": -2.5,
        "reward": 1.0,
        "children": [
            {
                "id": "child-1",
                "tokens": ["c", "F"],
                "logprob": -1.0,
                "reward": None,
                "children": [],
            },
            {
                "id": "child-2",
                "tokens": ["d", "q", "F"],
                "logprob": -1.4,
                "reward": None,
                "children": [
                    {
                        "id": "grand-1",
                        "tokens": ["e", "F"],
                        "logprob": -0.8,
                        "reward": None,
                        "children": [],
                    }
                ],
            },
        ],
    }


def tree_logprob(node):
    return node["logprob"] + sum(tree_logprob(c) for c in node["children"])


def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node["children"])


def find_rewarded(node):
    out = []
    if node["reward"] is not None:
        out.append(node["id"])
    for c in node["children"]:
        out.extend(find_rewarded(c))
    return out


if __name__ == "__main__":
    tree = make_tree()
    print(f"Total nodes: {count_nodes(tree)}")
    print(f"Total log-prob (factorised over nodes): {tree_logprob(tree):.3f}")
    print(f"Nodes that received reward: {find_rewarded(tree)}")
    assert find_rewarded(tree) == ["root"], "only root has reward"
    print("\nKey property: log p(T) = sum of per-node log-pi; only root scored.")
