# $k_g$ scale-law experiment

**Level:** intermediate
**Prerequisites:** [10-kg-averaging](../../paper/concepts/10-kg-averaging.md)
**Used by:** none

## Plain-English intro

The $1/k_g$ averaging makes the loss invariant to branching factor — but at fixed *compute budget* (fixed total tokens generated per gradient step), there should be a sweet-spot $k_g^\star$. More children per parent → richer reward signal per parent but fewer parents per step (lower batch diversity). The improvement: a controlled sweep over $k_g \in \{1, 2, 4, 8, 16\}$ at fixed total token budget, plotting reward vs $k_g$.

## Formal setup

Hold total per-step token budget $B$ fixed. For each $k$, set the number of root rollouts per step to $G_k = B / (k \cdot \bar L)$ where $\bar L$ is the mean rollout length. Train for a fixed number of steps and record the final reward $R(k)$. Hypothesis:

$$
R(k) = R_\infty - \alpha / k - \beta \cdot k,
$$

with the maximum achieved at $k^\star = \sqrt{\alpha / \beta}$.

## Why this matters

Empirically grounds the design choice the blog made implicitly ($k_g \approx 16$ in their experiments) and gives a recipe for choosing $k_g$ on new tasks/budgets.

## Code

See [`../code/103-kg-scale-law.py`](../code/103-kg-scale-law.py). Validation mode: **MEASUREMENT** — toy bandit, sweep $k$, return reward + variance per $k$.

Cross-references the validation file [`../../../improvements/k-scale-sweep.py`](../../../improvements/k-scale-sweep.py) authored by Agent C.

## Cross-link to the chain

Echoes chain Ch 31's discussion of effective batch size for GRPO; here lifted to the tree-branching dimension.
