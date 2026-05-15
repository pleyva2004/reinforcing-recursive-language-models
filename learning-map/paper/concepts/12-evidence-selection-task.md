# Evidence selection task

**Level:** intro
**Prerequisites:** none
**Used by:** [13-rubric-llm-judge-reward](13-rubric-llm-judge-reward.md)

## Plain-English intro

The blog evaluates RLM training on an *evidence selection* task: given a long document (or set of documents) and a question, the agent must return a small set of text spans that justify the answer. This is the workload that motivates recursive decomposition — the document is too long for the policy's context, so children are spawned to summarise/extract from chunks.

## Formal definition

Input $x = (\text{question } q, \text{document } D)$ where $|D|$ exceeds the policy's effective context length $\ell$. Output $a \subseteq D$ is a set of text spans. Ground truth is a reference answer $a^\star$. The agent's goal is to maximise a rubric-based reward $R(a, a^\star, q) \in [0, 1]$ — see [13-rubric-llm-judge-reward](13-rubric-llm-judge-reward.md).

## Why this matters for the paper

Evidence selection is the canonical "cleanly decomposable, but per-chunk reward is noisy" task: it's the sweet spot where shared-policy advantage-inheritance shines. The paper's headline result (a 4B trained with this method matches Sonnet 4.6 on rubric score at 7s vs 60s wall-clock) is on this benchmark.

## Code

See [`../code/12-evidence-selection-task.py`](../code/12-evidence-selection-task.py).

## Cross-link to the chain

A specific instance of the task-design considerations in chain Ch 28's RLHF section (`pleyva2004/first-principles-to-llms` Ch 28 — *SFT, RLHF/PPO, DPO*) — choosing what to score and how.
