# RLM children as options (Sutton-Precup-Singh)

**Level:** advanced
**Prerequisites:** [04-child-rollout](../../paper/concepts/04-child-rollout.md), [11-recursive-subtree-loss](../../paper/concepts/11-recursive-subtree-loss.md)
**Used by:** none

## Plain-English intro

A `rlm_query(prompt)` call is an *option* in the Sutton-Precup-Singh (1999) hierarchical-RL sense: an initiation set (the parent's REPL state when `rlm_query` was emitted), an internal policy (the same $\pi_\theta$, conditioned on the prompt), and a termination condition (the child's `FINAL(...)`). Recognising this opens the door to importing the full options theory: SMDP Q-learning convergence, hierarchical Bellman equations, options-as-actions for higher-level planning.

## Formal mapping

Let $o_i = (I_i, \pi_{o_i}, \beta_i)$ be an option triple where:

- $I_i \subseteq \mathcal{S}$ is the set of REPL states from which $o_i$ can be invoked (= states where the policy emits `rlm_query(p_i)`),
- $\pi_{o_i}(\cdot \mid s) = \pi_\theta(\cdot \mid s)$ — the same shared policy, conditioned on prompt $p_i$,
- $\beta_i(s) \in \{0, 1\}$ is 1 iff $s$ contains a `FINAL(...)` invocation.

A `rlm_query` call instantiates option $o_i$. The whole RLM session is a semi-Markov decision process (SMDP) over high-level steps = one root token plus one full child rollout per `rlm_query`.

## Why this matters

Imports SMDP Q-learning convergence (Sutton et al. 1999, Theorem 1) for free under the natural conditions; gives a hierarchical Bellman equation that justifies advantage inheritance from a value-function angle (not just a credit-assignment angle); opens the door to multi-level planning (higher-level options that compose `rlm_query` calls).

## Code

See [`../code/104-rlm-as-options.py`](../code/104-rlm-as-options.py). Validation mode: **PROOF** — formal theorem statement + sketch in [`../../../proofs/rlm-as-options.tex`](../../../proofs/rlm-as-options.tex).

## Cross-link to the chain

Hierarchical extension of the MDP framework of chain Ch 29 (`pleyva2004/first-principles-to-llms` Ch 29 — *MDP foundations*); the SMDP angle is a natural future chain chapter.
