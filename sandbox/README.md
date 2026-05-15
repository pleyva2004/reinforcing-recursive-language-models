# Sandbox: Reinforcing Recursive Language Models

Two CPU-runnable demos that reproduce the central training-time mechanic of
the alphaXiv blog "Reinforcing Recursive Language Models": a **single shared
policy** trained with GRPO across both **parent (decomposer)** and **child
(sub-agent)** rollouts, where every child inherits the parent's group-relative
advantage and the per-rollout loss is balanced with a `1/k_g` factor.

The goal is pedagogical: every line is hand-derived numpy with no autograd,
no GPU, no opaque library. Each script verifies that the **mathematics of the
recursive GRPO loss** trains a working policy on a tractable toy.

## Files

| File                       | What it demonstrates                                                                                  | Runtime  |
|----------------------------|-------------------------------------------------------------------------------------------------------|----------|
| `toy_recursive_bandit.py`  | RLM-GRPO on a 3-paper / 4-passage tree contextual bandit. Closed-form softmax policy, two heads.      | <1s      |
| `tiny_gpt_rlm.py`          | RLM-GRPO on the Ch. 27 tiny char-level GPT (1 layer, d=16). Shared-policy "decompose then extract".   | ~1-5 min |
| `requirements.txt`         | numpy + matplotlib (matplotlib optional, gracefully falls back to a printed table).                   | -        |

## How to run

```bash
pip install -r requirements.txt          # numpy >= 1.24, matplotlib >= 3.5
python3 toy_recursive_bandit.py
python3 tiny_gpt_rlm.py
```

Both scripts use `np.random.seed(0)` (and explicit `default_rng` instances) so
results are reproducible.

## Mapping back to the math (`02-math-deep-dive.md`)

The blog's central training objective is

```
L_g^full(theta) = L_node(y_g, A_g; theta)              # parent / root term
                + (1/k_g) sum_{i=1..k_g} L_node(y_{g,i}, A_g; theta)  # child terms
```

with each `L_node` a PPO-clipped surrogate, `A_g = (r_g - mu_r) / sigma_r` the
group-relative advantage, `G` the group size, and `k_g` the children-per-root.

### `toy_recursive_bandit.py`

| Math object                                                | Code site                                          |
|------------------------------------------------------------|----------------------------------------------------|
| Parent policy `pi_theta(action | "root")`                  | `parent_logits` + `parent_sample_topk`             |
| Child policy `pi_theta(action | "paper p")`                | `child_logits[paper]` + `child_sample`             |
| Group-relative advantage `A_g`                             | `advantages = (rewards - mu) / sd`                 |
| PPO-clipped surrogate `L_node`                             | `ppo_loss_and_grad`                                |
| Per-root full loss with `1/k_g` averaging                  | `train()` body: `d_child += child_grad_sum / k_g`  |
| GRPO mean-over-G                                           | `d_parent /= G; d_child /= G`                      |

The policy is just two softmax heads, so the gradient of `log pi(a)` is the
exact `e_a - softmax(z)` formula. PPO clipping is implemented branch-by-branch
(no autograd). After 200 steps the children concentrate on the target
passage of each paper and the mean reward saturates near 1.0.

### `tiny_gpt_rlm.py`

| Math object                                                                | Code site                                  |
|----------------------------------------------------------------------------|--------------------------------------------|
| Shared policy `pi_theta` (one tiny GPT)                                    | `params` + `forward_logits`                |
| Root rollout `y_g`                                                         | `rollout()` until first `"="`              |
| Child rollouts `y_{g,i}` (each `<...>` block)                              | `rollout()` records with `is_child=True`   |
| Group-relative advantage `A_g`                                             | `advantages = (rewards - mu) / sd`         |
| Per-token PPO-clipped loss + `1/k_g` child averaging                       | `compute_grad_and_loss()`                  |
| AdamW step                                                                 | `class AdamW`                              |
| KL(initial || final) diagnostic                                            | `categorical_kl_to(...)`                   |

The model is the Ch. 27 tiny GPT (1-layer pre-norm transformer, hand-coded
forward + backward) with two simplifications matching the Ch. 31 GRPO recipe:

1. The transformer block is **frozen**; we only train the token embedding `E`
   (which doubles as the output head via weight tying) and the position
   embedding `P`. This isolates the policy-head signal.
2. We bootstrap the cached `theta_old` log-probs **at sample time** and do one
   inner step per outer iteration (standard "1-step PPO" simplification).

The reward is shaped (`+0.5` for using the `"="` answer protocol, `+0.5` for a
correct final letter, `+0.2` partial for seeing the target letter at all). Even
with the frozen transformer block, the trainable head learns to emit the
answer protocol within ~50 GRPO steps and reward roughly doubles.

## Limitations

These demos verify the **algorithmic core** of recursive-policy GRPO. They do
NOT reproduce the blog's empirical claims at scale:

- We use numpy, not PyTorch / vLLM. No GPU, no batched generation, no
  asynchronous sampler.
- The base model is a 18 K-parameter character-level toy, not Qwen-3-4B.
- The "REPL" stub returns a single character, not a real Python interpreter.
- We freeze the transformer block in `tiny_gpt_rlm.py`; the full model only
  trains the embedding + head. (The toy-bandit script trains the full
  policy parameters, which there are exactly two heads of softmax logits.)
- We do not reproduce the BrowseComp-Plus or DeepResearch-Bench evals.
- We use 1-step PPO (recompute `logp_old` at sample time, take one inner
  optimisation step). The full GRPO uses multiple inner steps with a
  dedicated `theta_old` snapshot.

What you should take away: the `1/k_g` balancing trick + advantage
inheritance is well-defined, has a closed-form gradient, and trains a
tractable shared policy on a tree task — exactly the property the blog
exploits to scale RL across a recursive sub-agent stack.
