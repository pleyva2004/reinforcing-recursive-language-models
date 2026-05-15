# Recursive subtree loss (arbitrary depth)

**Level:** advanced
**Prerequisites:** [10-kg-averaging](10-kg-averaging.md)
**Used by:** [104-rlm-as-options](../../improvements/concepts/104-rlm-as-options.md)

## Plain-English intro

The depth-1 loss (concept 10) generalises to arbitrary depth via simple recursion: the loss at a node is its own per-node PPO surrogate plus the *average* of the recursive losses of its children. The advantage threaded through the recursion is the root's $A_g$ — every node in the subtree sees the same scalar.

## Formal definition

Define for each node $v$ in tree $\mathcal{T}_g$ rooted at $g$:

$$
\mathcal{L}^{\text{subtree}}(y_v, A_g; \theta) = \mathcal{L}^{\text{node}}(y_v, A_g; \theta) + \frac{1}{k_v} \sum_{i=1}^{k_v} \mathcal{L}^{\text{subtree}}(y_{v_i}, A_g; \theta),
$$

with $k_v = |C(v)|$ and the convention $\frac{1}{k_v}\sum_{i=1}^{0} \cdot := 0$ at leaves. The total batch loss is

$$
\mathcal{L}(\theta) = -\frac{1}{G} \sum_{g=1}^G \mathcal{L}^{\text{subtree}}(y_g, A_g; \theta).
$$

## Why this matters for the paper

It's the closed-form, depth-agnostic objective. Whether a tree is depth 2 or depth 7, training code uses the same recursion, no special-casing.

## Code

See [`../code/11-recursive-subtree-loss.py`](../code/11-recursive-subtree-loss.py).

## Cross-link to the chain

Tree-recursive lift of the flat sequence-level loss in chain Ch 31; analogous to the recursive Bellman backup of chain Ch 29 (`pleyva2004/first-principles-to-llms` Ch 29 — *MDP foundations*) but applied to losses rather than values.
