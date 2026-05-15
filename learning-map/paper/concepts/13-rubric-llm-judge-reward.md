# Rubric-based LLM-judge reward

**Level:** intermediate
**Prerequisites:** [12-evidence-selection-task](12-evidence-selection-task.md)
**Used by:** none directly (used implicitly by all training)

## Plain-English intro

For evidence selection, gold answers admit multiple valid text spans, so a verifiable F1-of-snippets reward is too brittle (the blog tried it; "proved to be very noisy"). Instead, the reward is a *rubric score* produced by an LLM judge: a frozen larger model is given the question, the gold answer, and the agent's selected spans, and asked to score on a fixed rubric (e.g., relevance, completeness, parsimony). The score in $[0, 1]$ becomes $r_g$.

## Formal definition

Let $J$ be a frozen judge LM with rubric prompt template $\tau$. Then

$$
r_g = J\big(\tau(q, a^\star, a_g)\big) \in [0, 1].
$$

The judge is held fixed across training (no gradient flows through $J$). The advantage feeding GRPO is then $A_g = (r_g - \mu_r)/(\sigma_r + \delta)$ as in concept 8.

## Why this matters for the paper

The choice of LLM-judge over verifiable F1 is the *training-signal design* decision that makes this task RL-trainable at 4B scale. It's a reminder that for fuzzy NLP tasks, training-signal design is as important as architecture or algorithm.

## Code

See [`../code/13-rubric-llm-judge-reward.py`](../code/13-rubric-llm-judge-reward.py).

## Cross-link to the chain

Same RLHF-style preference-model framing as chain Ch 28 (`pleyva2004/first-principles-to-llms` Ch 28 — *RLHF/PPO, DPO*); rubric scoring is the practitioners' workhorse alternative to per-step preference data.
