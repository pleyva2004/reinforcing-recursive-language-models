"""11-recursive-subtree-loss.py — recursive aggregation across arbitrary depth.

Implements L_subtree(v, A) = L_node(v, A) + (1/k_v) * sum L_subtree(child_i, A).
Builds a depth-3 tree and confirms the recursive aggregation matches an
equivalent flat sum over nodes (with the right 1/k weights).

Run: python3 11-recursive-subtree-loss.py
"""


def make_deep_tree():
    return {
        "id": "root",
        "L_node": 1.0,
        "children": [
            {
                "id": "c1",
                "L_node": 0.5,
                "children": [
                    {"id": "g1", "L_node": 0.2, "children": []},
                    {"id": "g2", "L_node": 0.4, "children": []},
                ],
            },
            {"id": "c2", "L_node": 0.7, "children": []},
        ],
    }


def L_subtree(node):
    L = node["L_node"]
    if node["children"]:
        L += sum(L_subtree(c) for c in node["children"]) / len(node["children"])
    return L


def L_subtree_explicit(node, weight=1.0):
    """Same value computed by traversing the tree and accumulating each
    node's contribution * its accumulated 1/k weight."""
    contributions = [(node["id"], weight, node["L_node"])]
    if node["children"]:
        w = weight / len(node["children"])
        for c in node["children"]:
            contributions.extend(L_subtree_explicit(c, w))
    return contributions


if __name__ == "__main__":
    tree = make_deep_tree()
    L = L_subtree(tree)
    parts = L_subtree_explicit(tree)
    L_check = sum(w * v for (_id, w, v) in parts)
    print(f"L_subtree (recursive)        = {L:.4f}")
    print(f"L_subtree (explicit weights) = {L_check:.4f}")
    print("Per-node weighted contributions:")
    for nid, w, v in parts:
        print(f"  {nid:>6}: weight={w:.3f}, L_node={v:.3f}, w*L_node={w*v:.4f}")
    assert abs(L - L_check) < 1e-12
    print("\nKey property: arbitrary-depth aggregation collapses to a")
    print("nested 1/k_v average — identity holds at any depth.")
