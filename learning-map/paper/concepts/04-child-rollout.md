# Child RLM rollout (`rlm_query`)

**Level:** intro
**Prerequisites:** [03-root-rollout](03-root-rollout.md)
**Used by:** [05-trajectory-tree](05-trajectory-tree.md), [06-shared-policy](06-shared-policy.md), [104-rlm-as-options](../../improvements/concepts/104-rlm-as-options.md)

## Plain-English intro

A *child rollout* is what `rlm_query(prompt)` produces. The harness pauses the parent, runs a brand-new RLM rollout under the *same policy* with `prompt` as input, waits for that child's `FINAL(...)`, and resumes the parent with the child's answer assigned to the LHS of the `rlm_query` call. Children are full RLM rollouts and may themselves call `rlm_query` — recursion.

## Formal definition

For a parent node with token sequence $y_g$, let $\mathrm{Children}(g) = \{i : y_g^{(t)} \text{ emits } \texttt{rlm\_query}(p_i) \text{ for some } t\}$. Each child $i$ is a rollout

$$
y_{g,i} \sim \pi_\theta(\cdot \mid p_i, o^{(<t)}_{g,i}),
$$

terminating with answer $a_{g,i}$ which is fed back into the parent's REPL as the return value of the corresponding `rlm_query` call. **Crucially**, child rollouts receive no scalar reward of their own; the only reward in the system is $r_g$ at the root.

## Why this matters for the paper

The lack of a child reward is exactly what motivates *advantage inheritance*: we still need a learning signal for the child's tokens, and the cleanest unbiased choice is to use the parent's GRPO advantage.

## Code

See [`../code/04-child-rollout.py`](../code/04-child-rollout.py).

## Cross-link to the chain

Children are sub-MDPs of the per-token causal-LM MDP of chain Ch 25, executed under the same shared $\pi_\theta$ that drives the root.
