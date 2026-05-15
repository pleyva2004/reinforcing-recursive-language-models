# $1/k_g$ child contribution averaging

**Level:** advanced
**Prerequisites:** [09-advantage-inheritance](09-advantage-inheritance.md)
**Used by:** [11-recursive-subtree-loss](11-recursive-subtree-loss.md), [103-kg-scale-law](../../improvements/concepts/103-kg-scale-law.md)

## Plain-English intro

If a root spawns 1 child it contributes 1 child loss; if it spawns 100 children it contributes 100 child losses. Without normalisation, "spawn-happy" roots dominate the gradient. The paper divides the sum of child losses by $k_g$, the number of children spawned by root $g$, so each root contributes equally regardless of branching factor.

## Formal definition

Let root $g$ have $k_g = |\mathrm{Children}(g)|$ children $\{y_{g,i}\}_{i=1}^{k_g}$, each assigned advantage $A_g$ (concept 9). The per-root loss is

$$
\mathcal{L}_g^{\text{full}}(\theta) = \mathcal{L}_g^{\text{root}}(\theta) + \frac{1}{k_g} \sum_{i=1}^{k_g} \mathcal{L}^{\text{node}}(y_{g,i}, A_g; \theta).
$$

(Convention: $\frac{1}{k_g} \sum_{i=1}^{0} \cdot := 0$ when $k_g = 0$.)

## Why this matters for the paper

Removes the perverse incentive to spawn more children just to up-weight a root. Keeps the per-root contribution to the batch gradient bounded and balanced, which is critical for stable training when branching factors vary across rollouts.

## Code

See [`../code/10-kg-averaging.py`](../code/10-kg-averaging.py).

## Cross-link to the chain

A specific instance of the per-trajectory loss-balancing trick alluded to in chain Ch 31's GRPO discussion (where token-level losses are averaged by sequence length).
