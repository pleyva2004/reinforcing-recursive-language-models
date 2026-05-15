# Per-child local-baseline variance reduction

**Level:** advanced
**Prerequisites:** [09-advantage-inheritance](../../paper/concepts/09-advantage-inheritance.md)
**Used by:** none

## Plain-English intro

Inheritance gives every child the parent's $A_g$. That's unbiased but high-variance — a child whose tokens are "mid-task irrelevancies" gets the same noisy advantage as the child that produced the load-bearing snippet. The improvement: subtract a *child-conditional baseline* $b(s_v) := \mathbb{E}[A_g \mid s_v]$ before applying the surrogate. The baseline is fitted online with a small value head; under the conditional-independence assumption $(*)$, subtracting $b$ doesn't bias the estimator and reduces variance.

## Formal definition

Let $b_\phi: \mathcal{S} \to \mathbb{R}$ be a learned value head. Replace the assigned advantage at node $v$ with

$$
\tilde A_v = A_g - b_\phi(s_v).
$$

The PPO surrogate uses $\tilde A_v$ instead of $A_g$. Train $\phi$ by regressing $b_\phi(s_v)$ on $A_g$ over the batch (MSE loss). By the standard control-variate identity, $\mathrm{Var}(\hat g)$ is reduced whenever $\mathrm{Cov}(b_\phi(s_v), \nabla_\theta \log \pi_\theta(y_v) \cdot A_g) > 0$.

## Why this matters

Variance reduction directly reduces sample complexity. If the variance ratio achieves the empirical $0.6$-$0.8$ region we observe in the toy run (see code), expected wall-clock to a target reward drops by a corresponding factor at large scale.

## Code

See [`../code/102-local-baseline-variance-reduction.py`](../code/102-local-baseline-variance-reduction.py). Validation mode: **MEASUREMENT** — runs the toy two-armed RLM bandit twice (with and without baseline) and reports $\mathrm{Var}(\tilde A) / \mathrm{Var}(A)$.

Cross-references the validation file [`../../../improvements/local-baseline.py`](../../../improvements/local-baseline.py) authored by Agent C.

## Cross-link to the chain

Standard control-variate trick from chain Ch 31 (`pleyva2004/first-principles-to-llms` Ch 31 — *policy gradient, GRPO*); novelty here is the per-node-state conditioning.
