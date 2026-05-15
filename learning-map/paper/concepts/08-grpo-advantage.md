# Group-relative advantage (GRPO)

**Level:** intermediate
**Prerequisites:** [07-ppo-clipped-surrogate](07-ppo-clipped-surrogate.md)
**Used by:** [09-advantage-inheritance](09-advantage-inheritance.md), [102-local-baseline-variance-reduction](../../improvements/concepts/102-local-baseline-variance-reduction.md)

## Plain-English intro

GRPO (Group-Relative Policy Optimisation; DeepSeek-Math 2024) replaces the value-network baseline of PPO with a *group baseline*: for each prompt, sample $G$ rollouts, take their reward mean and standard deviation, and normalise. The resulting standardised reward is used as the advantage $A_g$ assigned to every token in rollout $g$.

## Formal definition

For a prompt $x$, sample $G$ rollouts $\{y_g\}_{g=1}^G \sim \pi_{\theta_\text{old}}(\cdot \mid x)$ with rewards $\{r_g\}$. Let $\mu_r = \frac{1}{G}\sum_g r_g$ and $\sigma_r = \sqrt{\frac{1}{G}\sum_g (r_g - \mu_r)^2}$. Then the per-rollout advantage is

$$
A_g = \frac{r_g - \mu_r}{\sigma_r + \delta}, \qquad \delta > 0\ \text{small.}
$$

Every token in $y_g$ inherits the same $A_g$ (this is the standard sequence-level credit-assignment used by GRPO).

## Why this matters for the paper

GRPO is the base RL algorithm. The paper extends GRPO from "every token gets the same $A_g$" to "every node in the tree gets the same $A_g$" — see [09-advantage-inheritance](09-advantage-inheritance.md).

## Code

See [`../code/08-grpo-advantage.py`](../code/08-grpo-advantage.py).

## Cross-link to the chain

Chain Ch 31 (`pleyva2004/first-principles-to-llms` Ch 31 — *policy gradient, GRPO, RLHF/DPO bridge*) covers GRPO end-to-end with the baseline-doesn't-bias proof.
