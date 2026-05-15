# Advantage inheritance child-from-parent

**Level:** advanced
**Prerequisites:** [05-trajectory-tree](05-trajectory-tree.md), [08-grpo-advantage](08-grpo-advantage.md)
**Used by:** [10-kg-averaging](10-kg-averaging.md), [11-recursive-subtree-loss](11-recursive-subtree-loss.md), [101-tighter-inheritance-unbiasedness](../../improvements/concepts/101-tighter-inheritance-unbiasedness.md), [102-local-baseline-variance-reduction](../../improvements/concepts/102-local-baseline-variance-reduction.md)

## Plain-English intro

Children get no scalar reward, so we have nothing to baseline. The paper's choice: have each child rollout *inherit* its parent's GRPO advantage. So if root $g$ has advantage $A_g$, every child $i$ of $g$ uses $A_{g,i} = A_g$, and every grand-child uses $A_g$ too. This is the same trick GRPO already uses for tokens within a sequence, lifted to nodes within a tree.

## Formal definition

For root $g$ with advantage $A_g$ (concept 8), for every descendant $v$ of $g$ in the tree $\mathcal{T}_g$ rooted at $g$,

$$
A_v := A_g.
$$

The per-rollout PPO surrogate at $v$ uses this assigned $A_v$:

$$
\mathcal{L}^{\text{node}}_v = \mathcal{L}^{\text{node}}(y_v, A_g; \theta).
$$

Approximate unbiasedness rests on a conditional-independence assumption $(*)$: $\mathbb{E}[r_g \mid y_v, \theta] = \mathbb{E}[r_g \mid \mathrm{ancestors}(v), \theta]$, made formal in [proofs/inheritance-unbiasedness.tex](../../../proofs/inheritance-unbiasedness.tex).

## Why this matters for the paper

This is the load-bearing credit-assignment idea: it kills the need for a per-child reward, makes the shared-policy choice possible, and reduces the whole setup to "GRPO over trees instead of sequences".

## Code

See [`../code/09-advantage-inheritance.py`](../code/09-advantage-inheritance.py).

## Cross-link to the chain

Extends chain Ch 31's GRPO sequence-level credit assignment to tree-level credit assignment.
