# Math deep dive — Reinforcing Recursive Language Models

> Source: alphaXiv blog by NovaSky-AI / SkyRL contributors (https://www.alphaxiv.org/blog/reinforcement-learning-for-rlms). Code: SkyRL PR #1596 (https://github.com/NovaSky-AI/SkyRL/pull/1596). Primary reference: *Recursive Language Models* (Zhang et al., alphaxiv:2512.24601).

This document treats the blog as a research artifact and goes through the load-bearing math step by step. Where the math foundations are non-trivial I cite the canonical chapter from [`pleyva2004/first-principles-to-llms`](https://github.com/pleyva2004/first-principles-to-llms) (the **chain repo**), which is the atlas this study draws on.

## Notation key

| Symbol | Meaning |
| --- | --- |
| $\pi_\theta(a \mid s)$ | Parameterised policy (autoregressive LM): probability of token $a$ given context $s$. |
| $\pi_{\theta_{\text{old}}}$ | Behaviour policy used to sample rollouts (frozen during the gradient update; PPO-style). |
| $x$ | A query (the user input fed to the root RLM). |
| $y_g$ | The $g$-th root RLM rollout: a sequence of tokens including any `rlm_query(...)` calls that spawn children. |
| $G$ | Number of root rollouts per query (the GRPO group size). |
| $k_g$ | Number of child RLM rollouts spawned by root $g$. |
| $y_{g,i}$ | The $i$-th child rollout of root $g$. |
| $r_g \in [0,1]$ | Rubric reward for root $g$ (LLM-judge score on the final answer). |
| $A_g$ | Group-relative advantage for root $g$: $A_g = (r_g - \mu_r)/\sigma_r$ where $\mu_r, \sigma_r$ are the mean/std over the $G$-group. |
| $\rho_\theta(y^{(t)})$ | Importance ratio at token $t$: $\pi_\theta(y^{(t)} \mid s^{(t)}) / \pi_{\theta_{\text{old}}}(y^{(t)} \mid s^{(t)})$. |
| $\epsilon$ | PPO clipping radius (typically $\varepsilon = 0.2$). |
| $\mathcal{T}$ | The full trajectory tree (root + descendants). |

For the broader probability/RL notation (sample spaces, KL, expectations, etc.) see the [math-foundations glossary](https://github.com/pleyva2004/math-foundations/blob/main/NOTATION.md) and chain Chapters 8-11.

## 1. The RLM trajectory tree

Following the original RLM paper (Zhang et al., alphaxiv:2512.24601), each RLM session lives inside a Python **REPL** sandbox that exposes built-in functions:

- `FINAL(answer)` / `FINAL_VAR(name)` — terminate the rollout, returning the answer (literal or REPL-variable lookup).
- `rlm_query(prompt, context=None)` — spawn a *single* child RLM rollout under the same policy.
- `rlm_query_batched(prompts, context=None)` — spawn a *list* of child rollouts.

A **rollout** is a finite sequence of (model output, REPL response) turn pairs that ends in `FINAL(...)`. Each `rlm_query(...)` call inside a rollout spawns a **child** that is itself a full rollout — recursion. The full session is a tree $\mathcal{T}$ rooted at the root RLM, with internal nodes = RLMs that called `rlm_query`, leaves = RLMs whose action was `FINAL(...)`.

This is **not** a flat MDP. It's a tree-structured trajectory where every node is itself a sequence drawn from $\pi_\theta$, and edges correspond to `rlm_query` invocations.

Connect to the chain: the per-token causal-LM MDP from chain Ch 25 (state = prefix, action = next token) is recovered at every node. The tree structure is the new ingredient.

## 2. The training objective

### 2.1 Per-rollout PPO-clipped surrogate

Each rollout (root or child) contributes a token-level PPO-clipped surrogate, identical in form to the standard PPO objective from chain Ch 28 / Ch 31:

$$
\mathcal{L}_{\text{node}}(y, A; \theta) = \frac{1}{|y|} \sum_{t=1}^{|y|} \min\!\Big(\rho_\theta(y^{(t)}) \, A, \ \mathrm{clip}\big(\rho_\theta(y^{(t)}), 1-\varepsilon, 1+\varepsilon\big) \, A\Big).
$$

Here $A$ is *whatever advantage gets assigned to that node*; for the root, $A = A_g$ (its own GRPO advantage), for a child, $A = A_g$ as well (inherited — see below).

### 2.2 Root loss (standard GRPO)

For each query $x$ in the batch, sample $G$ root rollouts $\{y_g\}_{g=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)$ and reward each one with $r_g$. Compute group-relative advantages

$$
A_g = \frac{r_g - \mu_r}{\sigma_r + \delta}, \qquad \mu_r = \frac{1}{G}\sum_g r_g, \quad \sigma_r = \mathrm{stddev}(r_g), \quad \delta \to 0^+ \text{ for stability}.
$$

The per-root contribution is:

$$
\mathcal{L}_g^{\text{root}}(\theta) = \mathcal{L}_{\text{node}}(y_g, A_g; \theta).
$$

### 2.3 Child loss (advantage inheritance + $1/k_g$ averaging)

Each child rollout $y_{g,i}$ inherits the parent's advantage: $A_{g,i} := A_g$. Its contribution is the same PPO-clipped surrogate but evaluated on $y_{g,i}$:

$$
\mathcal{L}_{g,i}^{\text{child}}(\theta) = \mathcal{L}_{\text{node}}(y_{g,i}, A_g; \theta).
$$

The full per-rollout loss for root $g$ is

$$
\mathcal{L}_g^{\text{full}}(\theta) = \mathcal{L}_g^{\text{root}}(\theta) + \frac{1}{k_g} \sum_{i=1}^{k_g} \mathcal{L}_{g,i}^{\text{child}}(\theta).
$$

The $1/k_g$ factor is the **balancing trick**: without it, a root that spawns $k_g \gg 1$ children would dominate the gradient. With it, every root contributes equally to the batch gradient regardless of how many children it spawned.

### 2.4 Recursive subtree loss (arbitrary depth)

The depth-1 case generalises. For any rollout $y$ with children $\{y_i\}_{i=1}^{k_y}$:

$$
\mathcal{L}_{\text{subtree}}(y, A; \theta) = \mathcal{L}_{\text{node}}(y, A; \theta) + \frac{1}{k_y} \sum_{i=1}^{k_y} \mathcal{L}_{\text{subtree}}(y_i, A; \theta).
$$

This is a clean recursive definition. Every descendant in the tree carries the *same* advantage $A$ — the root's GRPO advantage. The $1/k_y$ at every level keeps depth-balanced contribution.

## 3. Why advantage inheritance is (approximately) unbiased

The blog claims:

> The rationale behind having children inherit their parent trajectory's final advantage is the same rationale behind assigning every token the same sequence-level advantage in GRPO, even though different tokens make different contributions to the sequence-level advantage. This is an unbiased estimator of the true gradient that, with sufficient training steps, yields stable results.

Let's unpack that.

### 3.1 The flat-GRPO unbiasedness lemma

In flat GRPO (chain Ch 31), the policy gradient estimator at token $t$ in rollout $g$ is

$$
\hat g_g^{(t)} = \nabla_\theta \log \pi_\theta(y_g^{(t)} \mid y_g^{(<t)}) \cdot A_g.
$$

It's unbiased for the true policy gradient $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(y) \cdot R(y)]$ in the sense that, under the group baseline (mean of $r_g$),

$$
\mathbb{E}_{y \sim \pi_\theta}\Big[\nabla_\theta \log \pi_\theta(y) \cdot (R(y) - b)\Big] = \mathbb{E}\Big[\nabla_\theta \log \pi_\theta(y) \cdot R(y)\Big]
$$

for any $b$ that doesn't depend on $y$ (chain Ch 31 baseline-doesn't-bias proof). The mean of $\{r_g\}$ is *almost* such a $b$ — it depends weakly on $y_g$ through its inclusion in the group, but the bias is $O(1/G)$ and vanishes asymptotically.

### 3.2 Tree extension

For a child rollout $y_{g,i}$, the gradient contribution is

$$
\hat g_{g,i}^{(t)} = \nabla_\theta \log \pi_\theta(y_{g,i}^{(t)} \mid s_{g,i}^{(t)}) \cdot A_g.
$$

This is unbiased for the *child's true policy gradient* under the assumption:

> **(*)** The conditional reward distribution $p(r_g \mid \text{root } y_g)$ is determined entirely by the root rollout $y_g$ (which includes the children's outputs as REPL responses). In particular, the reward $r_g$ is a deterministic-or-stochastic function of the ENTIRE tree's outputs. Under this assumption, $A_g$ aggregates credit for everything the tree did, including the child's contribution.

This holds for evidence selection: the rubric reward is on the final answer, which is computed from the snippets returned by the children. Children that picked good snippets caused a high $A_g$; children that picked bad ones caused a low $A_g$. The advantage *flows* through the tree.

### 3.3 What can break

(*) breaks when:

- **Children have non-additive interactions** — e.g. tool chains where child A's output is child B's input, and the bug is in the chain rather than any individual call. The advantage of the root then doesn't reflect any single child's contribution.
- **Children have observed but unrewarded sub-tasks** — e.g. a child that returns a summary the parent ignores. Inheriting the parent's advantage credits/discredits the child for work that didn't matter.
- **Variance scales with $k_g$** — even in the well-behaved case, the per-child gradient is $A_g$-weighted but $A_g$ has variance $\Theta(1/G)$; with $k_g$ children, the per-root variance contribution is $\Theta(k_g / G)$. Larger $k_g$ → noisier root-level gradient.

The blog flags the first two as future work ("more fine-grained credit assignment can lead to faster convergence") but doesn't bound the bias or variance formally.

## 4. Why one shared policy beats two

The original RLM paper trained only the root with a frozen LM for children. The blog's central engineering claim is that one shared policy is strictly better:

1. **No second reward signal.** Training a separate child policy requires per-child rewards (e.g. F1 of the snippet selection). The blog tried this — *"verifiable rewards like F1 of selected snippets... proved to be very noisy."* The shared policy bypasses the need by piggybacking on the parent reward.
2. **Compute / memory.** One model in memory, one optimizer state, one set of gradient buffers. With AdamW (chain Ch 14), the optimizer state is ~3× model size; halving the number of policies is a 2× memory win (or 2× larger model at the same memory budget).
3. **Transfer.** The same parameters are updated by both root and child gradients. A child rollout's gradient reshapes the policy in ways that benefit *all* future children — and the root, which uses the same parameters to decide when to call `rlm_query` and what prompt to give the children. The model develops a coherent self-model.
4. **Deployment.** One model to serve. Cost-, throughput-, and infrastructure-friendly.

## 5. Empirical results

- Backbone: Qwen3.5-4B (after a small cold-start SFT from teacher rollouts generated by Qwen3.5-397B-A17B).
- Training: 8×H200 node, batch 16, 8 root samples per prompt, $k_g$ up to 4 → up to 512 concurrent rollouts.
- Average rubric score on training set: $0.3 \to 0.6$ post-RL.
- Eval: $\approx 0.6$ vs Sonnet 4.6's $0.607$ (matches). Wall-clock: $7\,$s vs Sonnet's $60\,$s (8.5× faster).

The pattern matches the chain Ch 31 GRPO-on-tiny-GPT toy: a small, well-targeted, group-baseline-trained model can ride the variance reduction of its own samples to match much larger models on a structured task.

## 6. Connections to the chain

- **Chain Ch 25** (causal LM = MDP): every node in the RLM tree is a Ch 25 MDP; the tree structure is built on top.
- **Chain Ch 27** (tiny GPT pre-training): the policy is an autoregressive LM trained the same way (the SFT phase here uses the same NTP loss).
- **Chain Ch 28** (SFT, RLHF/PPO/GRPO, DPO): the cold-start SFT and the PPO-clipped per-node loss are direct lifts.
- **Chain Ch 29** (MDP foundations): the trajectory + Bellman setup, lifted to a tree.
- **Chain Ch 30** (max-ent + soft Bellman): the soft-policy framework gives a principled way to add an entropy bonus to the per-node loss if exploration is needed.
- **Chain Ch 31** (policy gradient, GRPO, RLHF/DPO bridge): this is the chain's chapter most directly extended by the blog. The blog's recursive-subtree loss is a generalisation of Ch 31's GRPO objective from a sequence to a tree.

## 7. Open mathematical questions

1. **Tight unbiasedness conditions.** Formalise (*) and bound the bias of inheritance at depth $d > 1$. This is the math improvement proposed in `05-improvements.tex`.
2. **Variance bound.** What's $\mathrm{Var}(\hat g_\text{tree})$ as a function of $k_g$ and tree depth $d$? Likely scales as $O(k_g \cdot d / G)$ — needs a proof.
3. **Optimal $k_g$.** At fixed compute (fixed total tokens generated per gradient step), what is the optimal number of children per parent? More children → more reward signal per parent but smaller batch (fewer parents per step). A scale-law experiment is the experimental improvement in `05-improvements.tex`.
4. **Connection to options framework.** Can RLM children be modelled as options (Sutton-Precup-Singh 1999) with init-set = current REPL state, term-set = `FINAL(...)` calls? If so, what known theorems carry over? Theoretical improvement in `05-improvements.tex`.

## 8. Notes on engineering subtleties

- **Race conditions on REPL timeouts.** With 512 concurrent rollouts and `rlm_query` calls, the SkyRL implementation initially had race conditions around child-RLM REPL timeouts; fixing them was prerequisite to the stable reward curve they report. Practical reminder that scaling agentic-RL infrastructure is harder than the math suggests.
- **Cold-start SFT is essential at 4B.** Without it, pass@16 = 0 — the harness syntax (especially `FINAL(...)` vs `FINAL_VAR(...)` and code-block formatting) is outside the model's edge of competence. RL alone can't bootstrap from zero pass.
- **Strategy must be in the prompt.** They don't expect the model to discover the search → expand → extract strategy; it's prompted in. Strategy discovery is on the future-work list.
- **Rubric reward (LLM-as-judge) beats verifiable F1 reward** for this task because gold answers admit multiple valid text spans. Training-signal design ≠ ground-truth design.
