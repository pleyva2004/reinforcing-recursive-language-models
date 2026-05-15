# Paper concept graph — Reinforcing Recursive Language Models

This is the dependency DAG for the 14 paper concepts. Each node links to its concept page; each concept page has an aligned `code/` runnable demo.

```mermaid
graph TD
    classDef intro fill:#dff5d8,stroke:#3d8b3d,color:#000
    classDef intermediate fill:#ffe9b3,stroke:#b88a00,color:#000
    classDef advanced fill:#ffc9c9,stroke:#a02525,color:#000

    n1["1. Recursive Language Model"]:::intro
    n2["2. Python REPL environment"]:::intro
    n3["3. Root rollout"]:::intro
    n4["4. Child rollout (rlm_query)"]:::intro
    n5["5. RLM trajectory tree"]:::intermediate
    n6["6. Shared policy"]:::intermediate
    n7["7. PPO clipped surrogate"]:::intermediate
    n8["8. Group-relative advantage (GRPO)"]:::intermediate
    n9["9. Advantage inheritance"]:::advanced
    n10["10. 1/k_g averaging"]:::advanced
    n11["11. Recursive subtree loss"]:::advanced
    n12["12. Evidence selection task"]:::intro
    n13["13. Rubric LLM-judge reward"]:::intermediate
    n14["14. Cold-start SFT"]:::intermediate

    n1 --> n2
    n1 --> n3
    n2 --> n3
    n3 --> n4
    n3 --> n5
    n4 --> n5
    n4 --> n6
    n6 --> n7
    n7 --> n8
    n5 --> n9
    n8 --> n9
    n9 --> n10
    n10 --> n11
    n12 --> n13
    n6 --> n14

    click n1 "concepts/01-rlm-defn.md"
    click n2 "concepts/02-python-repl-environment.md"
    click n3 "concepts/03-root-rollout.md"
    click n4 "concepts/04-child-rollout.md"
    click n5 "concepts/05-trajectory-tree.md"
    click n6 "concepts/06-shared-policy.md"
    click n7 "concepts/07-ppo-clipped-surrogate.md"
    click n8 "concepts/08-grpo-advantage.md"
    click n9 "concepts/09-advantage-inheritance.md"
    click n10 "concepts/10-kg-averaging.md"
    click n11 "concepts/11-recursive-subtree-loss.md"
    click n12 "concepts/12-evidence-selection-task.md"
    click n13 "concepts/13-rubric-llm-judge-reward.md"
    click n14 "concepts/14-cold-start-sft.md"
```

## Skill levels

- **intro** (green): the substrate — what an RLM rollout *is*.
- **intermediate** (amber): the standard RL machinery (PPO, GRPO, SFT) plus the shared-policy choice.
- **advanced** (red): the paper's three novel ideas — advantage inheritance, $1/k_g$ averaging, recursive subtree loss.

## Foundation prereqs (chain)

The prereqs that live *outside* this paper graph are foundational chapters of [`pleyva2004/first-principles-to-llms`](https://github.com/pleyva2004/first-principles-to-llms):

- **Ch 25** — causal LM as MDP (every node)
- **Ch 27** — tiny GPT pre-training (the policy)
- **Ch 28** — SFT, RLHF/PPO, DPO (cold-start, per-node loss)
- **Ch 29** — MDP foundations (trajectory + Bellman)
- **Ch 30** — max-ent + soft Bellman (entropy bonus)
- **Ch 31** — policy gradient, GRPO, RLHF/DPO bridge (the chapter most directly extended)

See [`../tour.md`](../tour.md) §2 "Foundations walk" for the topologically-ordered reading plan.
