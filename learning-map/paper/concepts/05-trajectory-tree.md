# RLM trajectory tree

**Level:** intermediate
**Prerequisites:** [03-root-rollout](03-root-rollout.md), [04-child-rollout](04-child-rollout.md)
**Used by:** [09-advantage-inheritance](09-advantage-inheritance.md), [11-recursive-subtree-loss](11-recursive-subtree-loss.md)

## Plain-English intro

A full RLM session is a *finite tree* whose root is the user-prompted root rollout, whose internal nodes are RLM nodes that called `rlm_query`, and whose leaves are RLM nodes that called `FINAL` without calling `rlm_query`. Edges encode "node $u$ spawned node $v$ via its $i$-th `rlm_query`". The tree is the trajectory; credit assignment must respect its tree shape rather than treat the whole transcript as a flat sequence.

## Formal definition

Let $\mathcal{T}$ be a rooted finite tree with node set $V$. Each $v \in V$ is decorated with:

- a sequence $y_v \in \Sigma^*$ over the model vocabulary,
- a list of children $C(v) = (v_1, \dots, v_{k_v})$ in the order their `rlm_query` calls appear inside $y_v$,
- a (sole) reward $r_v$ which is non-zero only at the root.

Formally $\mathcal{T} = (V, E, y, r)$ with $E = \{(v, c) : v \in V,\ c \in C(v)\}$. The likelihood under $\pi_\theta$ factorises as $p_\theta(\mathcal{T}) = \prod_{v \in V} \pi_\theta(y_v \mid \mathrm{context}_v)$.

## Why this matters for the paper

Once the trajectory is a tree rather than a sequence, the GRPO baseline / advantage / loss machinery (concepts 7-11) all need a tree-aware extension. The $1/k_g$ averaging and recursive subtree loss are exactly that.

## Code

See [`../code/05-trajectory-tree.py`](../code/05-trajectory-tree.py).

## Cross-link to the chain

Generalises the flat trajectory of chain Ch 31 (`pleyva2004/first-principles-to-llms` Ch 31 — *policy gradient, GRPO*) from $\Sigma^*$ to a tree over $\Sigma^*$.
