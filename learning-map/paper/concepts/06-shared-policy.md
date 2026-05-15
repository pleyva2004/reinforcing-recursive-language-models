# Shared policy (one model, both roles)

**Level:** intermediate
**Prerequisites:** [04-child-rollout](04-child-rollout.md)
**Used by:** [07-ppo-clipped-surrogate](07-ppo-clipped-surrogate.md), [14-cold-start-sft](14-cold-start-sft.md)

## Plain-English intro

The original RLM paper trained the *root* policy with RL but used a frozen LM for children. The blog's central engineering claim is to share a **single** $\pi_\theta$ for both roles. Every gradient step updates the same weights from both root rollouts and child rollouts; deployment serves one model.

## Formal definition

Let $\theta \in \mathbb{R}^d$ parameterise an autoregressive policy $\pi_\theta$. The shared-policy assumption is

$$
\pi_\text{root}(\cdot) = \pi_\text{child}(\cdot) = \pi_\theta(\cdot).
$$

The training objective is therefore

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\mathcal{T} \sim p_\theta} \Big[\nabla_\theta \log p_\theta(\mathcal{T}) \cdot R(\mathcal{T})\Big]
$$

with $\log p_\theta(\mathcal{T}) = \sum_{v \in V} \log \pi_\theta(y_v \mid \mathrm{context}_v)$ — a single $\theta$ appears in every node.

## Why this matters for the paper

Sharing $\theta$ removes the need for a second reward signal (no per-child F1), halves the optimizer-state memory, and creates *transfer*: a gradient step from a child rollout reshapes $\theta$ in ways that benefit *all* future children and the root. It is the load-bearing engineering decision.

## Code

See [`../code/06-shared-policy.py`](../code/06-shared-policy.py).

## Cross-link to the chain

Same parameter-sharing idea as a multi-task language model in chain Ch 27 (`pleyva2004/first-principles-to-llms` Ch 27 — *tiny GPT pre-training*); RLM applies it across roles within an agentic loop instead of across tasks.
