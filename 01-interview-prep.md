# Reinforcing Recursive Language Models — talking points

> alphaXiv blog (NovaSky-AI / SkyRL contributors). Source: https://www.alphaxiv.org/blog/reinforcement-learning-for-rlms · Code: https://github.com/NovaSky-AI/SkyRL/pull/1596

## What's novel

A *single shared policy* trained to play both **parent decomposer** and **child sub-agent** in a recursive language-model loop, with **child rollouts inheriting the parent's GRPO advantage** rather than carrying their own reward signal. Concretely: for each query they sample $G$ root rollouts, compute group-relative advantages $A_g$ as in standard GRPO, then add to the loss every child rollout that root $g$ spawned with the *same* $A_g$ — averaged by $k_g$ (children-per-root) so spawning more sub-calls doesn't reweight a root's gradient. The result: one model, one reward signal, one optimizer state — recursive agentic behavior trained end-to-end.

## What's mathematically clever

Three things stand out.

1. **Treating the recursive call tree as one trajectory for credit-assignment purposes.** This is a structural extension of the GRPO trick: GRPO already assigns the same sequence-level advantage to every token regardless of that token's actual contribution; this paper extends the same logic to every *node in the tree* getting the root's advantage. Same unbiased-estimator rationale, same convergence intuition.
2. **The $1/k_g$ child-averaging term** in the loss. Without it, a root that spawns more children would dominate the gradient. With it, the per-rollout loss contribution stays balanced across recursion depth — which means the optimizer doesn't develop a perverse incentive to spawn more children just to weight that root higher.
3. **Recursive-subtree loss** $\mathcal{L}_{\text{subtree}}(y, A) = \mathcal{L}_{\text{node}}(y, A) + \frac{1}{k_y}\sum_i \mathcal{L}_{\text{subtree}}(y_i, A)$ generalises the depth-1 setup to arbitrary depth without changing the math.

## What I'd push back on

The unbiasedness argument is *informal*. Inheriting the parent's advantage is unbiased only if the conditional reward distribution of the child given the parent doesn't depend on $\theta$ within the rollout window — which holds for evidence selection (parent reward is roughly the sum of child rewards, child reward is a function of the snippet returned), but breaks for tasks where children have non-additive interactions (e.g. tool-use chains where child A's output is child B's input). The blog flags this as a future-work direction ("more fine-grained credit assignment can lead to faster convergence") but the current paper doesn't formalise the load-bearing assumption.

Also: the trained 4B model still trails Sonnet 4.6 on rubric score (0.6 vs 0.607). The win is wall-clock (7s vs 60s) and cost — a real production win, but not a quality breakthrough.

## Open questions

- **Depth scaling.** The recursive-subtree loss generalises to arbitrary depth, but they only train depth-1 (root + children). Does this still work at depth 2 (root → children → grandchildren)? Variance of the inherited-advantage estimator presumably grows with depth.
- **$k_g$ scale law.** Optimal $k_g$ at fixed compute is unstudied. Sweeping $k_g \in \{1, 2, 4, 8\}$ would tell us whether more children is always better or has a knee.
- **Strategy discovery.** They currently *prompt* the strategy (search → expand → extract). Can RL learn the strategy from scratch given only the reward? The "A-ha! moment" they flag in the future-work section.

## One-sentence elevator

RL fine-tune one small model to play *both* roles in an agentic loop by treating the entire recursive call tree as a single GRPO trajectory — children inherit the root's advantage, no separate per-child reward needed.
