# Recursive Language Model

**Level:** intro
**Prerequisites:** none (entry point)
**Used by:** [02-python-repl-environment](02-python-repl-environment.md), [03-root-rollout](03-root-rollout.md), [04-child-rollout](04-child-rollout.md)

## Plain-English intro

A *Recursive Language Model* (RLM) is a language model that, while generating tokens, can call **itself** as a subroutine via a built-in `rlm_query(prompt)` function. The recursive call spawns a fresh rollout of the same policy on a sub-problem; the spawned rollout returns its `FINAL(...)` answer back into the parent's REPL, where the parent continues generating. The structure of one session is therefore a *tree* of language-model rollouts, not a flat sequence.

## Formal definition

Let $\pi_\theta$ be an autoregressive policy. An RLM session is a finite tree $\mathcal{T}$ where each node $n \in \mathcal{T}$ is a sequence $y_n = (y_n^{(1)}, \dots, y_n^{(|y_n|)})$ sampled token-by-token from $\pi_\theta(\cdot \mid \text{state}_n)$. Each token is either ordinary text or a special *recursion token* whose decoded form is `rlm_query(p)` for some sub-prompt $p$; the action of emitting such a token *spawns a child node* with prompt $p$ and policy $\pi_\theta$. A node terminates by emitting `FINAL(answer)`.

## Why this matters for the paper

The RLM definition is the substrate on which all of the paper's machinery (advantage inheritance, $1/k_g$ averaging, shared policy) is built. Without the recursive call, the system reduces to a standard agentic LM with a flat trajectory and the paper's contributions vanish.

## Code

See [`../code/01-rlm-defn.py`](../code/01-rlm-defn.py).

## Cross-link to the chain

Single nodes are just causal-LM MDPs from chain Ch 25 (`pleyva2004/first-principles-to-llms` Ch 25 — *causal LM as MDP*). The recursion is the paper's novel layer on top.
