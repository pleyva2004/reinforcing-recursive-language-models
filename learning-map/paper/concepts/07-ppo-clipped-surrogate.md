# PPO clipped surrogate

**Level:** intermediate
**Prerequisites:** [06-shared-policy](06-shared-policy.md)
**Used by:** [08-grpo-advantage](08-grpo-advantage.md), [11-recursive-subtree-loss](11-recursive-subtree-loss.md)

## Plain-English intro

Per-node, the loss is the standard PPO clipped-surrogate from Schulman et al. (2017). For each token $t$ in a node's sequence $y$, you compute the importance ratio $\rho_\theta = \pi_\theta(y^{(t)} \mid \cdot) / \pi_{\theta_\text{old}}(y^{(t)} \mid \cdot)$ and use it to weight the (assigned) advantage $A$, with a clip to keep updates trust-region-like.

## Formal definition

For a node sequence $y = (y^{(1)}, \dots, y^{(|y|)})$ assigned advantage $A \in \mathbb{R}$,

$$
\mathcal{L}_{\text{node}}(y, A; \theta) = \frac{1}{|y|} \sum_{t=1}^{|y|} \min\!\Big(\rho_\theta(y^{(t)})\, A,\ \mathrm{clip}\big(\rho_\theta(y^{(t)}), 1-\varepsilon, 1+\varepsilon\big)\, A\Big),
$$

with $\rho_\theta(y^{(t)}) = \pi_\theta(y^{(t)} \mid y^{(<t)}) / \pi_{\theta_\text{old}}(y^{(t)} \mid y^{(<t)})$ and $\varepsilon \in (0,1)$ the clip width (typically $0.2$).

## Why this matters for the paper

This is the per-node atom out of which the tree-aware loss (concept 11) is built. The tree shape only affects *which* $A$ gets assigned and *how the per-node losses are aggregated*; the per-node form is unchanged from PPO.

## Code

See [`../code/07-ppo-clipped-surrogate.py`](../code/07-ppo-clipped-surrogate.py).

## Cross-link to the chain

Direct lift from chain Ch 28 (`pleyva2004/first-principles-to-llms` Ch 28 — *RLHF/PPO/DPO*) and Ch 31 (*GRPO*).
