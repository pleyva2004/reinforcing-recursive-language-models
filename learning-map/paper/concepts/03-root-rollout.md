# Root RLM rollout

**Level:** intro
**Prerequisites:** [01-rlm-defn](01-rlm-defn.md), [02-python-repl-environment](02-python-repl-environment.md)
**Used by:** [04-child-rollout](04-child-rollout.md), [05-trajectory-tree](05-trajectory-tree.md)

## Plain-English intro

The *root rollout* is the top-of-tree session that receives the user's task as its prompt and is what gets scored. It autoregressively generates code, executes it, may call `rlm_query` to spawn children, and finally calls `FINAL(...)`. The reward $r_g$ is computed *only* on the root's `FINAL(...)` answer — children get no direct reward.

## Formal definition

Given a task prompt $x$, the root rollout is the random sequence

$$
y_g \sim \pi_\theta(\cdot \mid x, o^{(<t)}_g) \quad t = 1, \dots, T_g
$$

where $o_g^{(t)}$ is the REPL observation after the $t$-th turn. Termination is the first $t$ with $\mathrm{FINAL} \in c_t$. The rollout returns answer $a_g = \mathrm{FINAL\text{-}arg}(c_{T_g})$ and receives scalar reward

$$
r_g = R(a_g, x).
$$

## Why this matters for the paper

Root rollouts are the only nodes whose reward enters the GRPO baseline directly; children inherit the root's *advantage* (id 9), so the root carries the entire reward signal for the whole tree it spawns. Understanding root rollouts is prerequisite to seeing why advantage inheritance is the natural credit-assignment choice.

## Code

See [`../code/03-root-rollout.py`](../code/03-root-rollout.py).

## Cross-link to the chain

Same structure as the agentic-LM rollouts of chain Ch 31 (`pleyva2004/first-principles-to-llms` Ch 31 — *policy gradient, GRPO, RLHF/DPO bridge*); the only addition is that some emitted tokens may spawn children.
