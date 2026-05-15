# Cold-start SFT for RLM harness syntax

**Level:** intermediate
**Prerequisites:** [06-shared-policy](06-shared-policy.md)
**Used by:** none (foundational pre-step before RL)

## Plain-English intro

A 4B base model has zero pass-rate on the RLM harness from a cold start — it doesn't reliably emit `FINAL(...)` vs `FINAL_VAR(...)` in the right format, doesn't know how to call `rlm_query` syntactically, and produces malformed code blocks. The blog runs a *cold-start SFT* phase first: supervised fine-tuning on a small set of demonstration trajectories (root + child rollouts) showing correct syntax, then warm-starts RL from that checkpoint.

## Formal definition

Let $\mathcal{D}_\text{sft} = \{(x_i, y_i)\}_{i=1}^N$ be a dataset of (prompt, demonstration) pairs where $y_i$ is a syntactically correct RLM rollout (possibly with `rlm_query` calls and their child rollouts as part of the demonstration). The SFT loss is the standard next-token-prediction loss

$$
\mathcal{L}_\text{SFT}(\theta) = -\frac{1}{N} \sum_{i=1}^N \frac{1}{|y_i|} \sum_t \log \pi_\theta(y_i^{(t)} \mid y_i^{(<t)}, x_i).
$$

After SFT to convergence, $\theta_\text{SFT}$ is used as the starting point for the GRPO / advantage-inheritance training.

## Why this matters for the paper

Without SFT, $\text{pass@}16 = 0$ at 4B — RL alone can't bootstrap from zero pass-rate (no positive reward to amplify). Cold-start SFT is therefore a *prerequisite*, not an ablation.

## Code

See [`../code/14-cold-start-sft.py`](../code/14-cold-start-sft.py).

## Cross-link to the chain

Direct lift from chain Ch 28 (`pleyva2004/first-principles-to-llms` Ch 28 — *SFT, RLHF/PPO, DPO*).
