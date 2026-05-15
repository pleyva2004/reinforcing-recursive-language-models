# Tour — Reinforcing Recursive Language Models

A guided walk through the learning-map. Read the paper graph + improvements graph + foundation prereqs in the order given here.

## 1. Reader's contract

**Who this is for.** A reader who has at least the level of an upper-undergraduate ML student — comfortable with PyTorch, has seen REINFORCE / PPO once, has read a transformer-pre-training tutorial. The "advanced" concept pages assume you can manipulate score-function estimators and clipped surrogates without consulting a textbook.

**Expected reading time.** 3-4 hours for the paper graph + 1-2 hours for the improvements graph + 30 min for the proofs. Roughly one full day of focused study.

**Math-foundations entry point.** If you are *not* yet comfortable with REINFORCE / PPO / GRPO at the equation level, start at [`pleyva2004/first-principles-to-llms` Ch 31](https://github.com/pleyva2004/first-principles-to-llms) and work backwards from there. If you have not yet built a transformer LM end-to-end, start at Ch 27. If you are unsure about the MDP framing, start at Ch 29.

**Pacing convention used below.** Each prereq is tagged `skim` (5-10 min refresh), `read` (30-45 min careful read), or `drill` (1-2 h with paper-and-pencil derivations).

## 2. Foundations walk (chain prereqs)

Topologically-ordered foundation prereqs from the [chain repo](https://github.com/pleyva2004/first-principles-to-llms):

1. **Ch 25 — Causal LM as MDP** *(skim).* Why this paper needs it: every node in the RLM tree is one Ch 25 MDP; nothing in the paper changes that, only the inter-node structure.
2. **Ch 27 — Tiny GPT pre-training** *(skim).* Why: the policy $\pi_\theta$ is just an autoregressive transformer; the SFT phase reuses Ch 27's NTP loss.
3. **Ch 28 — SFT, RLHF/PPO, DPO** *(read).* Why: cold-start SFT (concept 14) is a direct lift, and the per-node loss (concept 7) is PPO. If you have not done PPO end-to-end before, this is the chapter to spend time on.
4. **Ch 29 — MDP foundations** *(read).* Why: the trajectory tree (concept 5) is a tree of MDP rollouts; the Bellman setup is what gets lifted to SMDPs in improvement 104.
5. **Ch 30 — Max-ent + soft Bellman** *(skim).* Why: optional — gives a principled way to add an entropy bonus to the per-node loss; not needed for understanding the paper, but useful if you want to extend it.
6. **Ch 31 — Policy gradient, GRPO, RLHF/DPO bridge** *(drill).* Why: this is *the* chapter the paper extends. The baseline-doesn't-bias proof is what makes inheritance unbiased; the GRPO sequence-level credit-assignment trick is what gets lifted to trees. If you do nothing else from the chain, drill this.

After this walk you have all the substrate you need.

## 3. Paper concepts walk

Topologically-ordered list of the 14 paper concepts ([`paper/README.md`](paper/README.md) for the DAG):

1. [01-rlm-defn](paper/concepts/01-rlm-defn.md) — what an RLM rollout *is*.
2. [02-python-repl-environment](paper/concepts/02-python-repl-environment.md) — the Python REPL that hosts every node.
3. [03-root-rollout](paper/concepts/03-root-rollout.md) — the only node that gets reward.
4. [04-child-rollout](paper/concepts/04-child-rollout.md) — what `rlm_query` produces.
5. [05-trajectory-tree](paper/concepts/05-trajectory-tree.md) — the full session is a tree.
6. [06-shared-policy](paper/concepts/06-shared-policy.md) — one $\theta$ for both roles.
7. [07-ppo-clipped-surrogate](paper/concepts/07-ppo-clipped-surrogate.md) — the per-node atom.
8. [08-grpo-advantage](paper/concepts/08-grpo-advantage.md) — group-relative baseline.
9. [09-advantage-inheritance](paper/concepts/09-advantage-inheritance.md) — child's $A := A_g$.
10. [10-kg-averaging](paper/concepts/10-kg-averaging.md) — the $1/k_g$ balancing trick.
11. [11-recursive-subtree-loss](paper/concepts/11-recursive-subtree-loss.md) — depth-agnostic loss.
12. [12-evidence-selection-task](paper/concepts/12-evidence-selection-task.md) — the workload.
13. [13-rubric-llm-judge-reward](paper/concepts/13-rubric-llm-judge-reward.md) — the training signal.
14. [14-cold-start-sft](paper/concepts/14-cold-start-sft.md) — bootstrap before RL.

**Reading strategy.** 1-6 are scaffolding; spend 10-15 minutes each. 7-11 are the load-bearing math; spend 30-45 minutes each, run the corresponding `code/` files, and verify the printed output matches your mental model. 12-14 are the experimental setup; skim unless you plan to reproduce.

## 4. Improvements walk

Four improvement proposals — two **PROOF**-validated math results and two **MEASUREMENT**-validated experiments. The full list with DAG is [`improvements/README.md`](improvements/README.md).

### 101. Tighter unbiasedness theorem for advantage inheritance — PROOF

The blog asserts inheritance is "unbiased in the same sense as GRPO" but doesn't formalise the assumption. The proposed theorem makes the conditional-independence assumption $(*)$ precise, proves unbiasedness under it, and quantifies bias when it fails (e.g., tool-use chains where child A's output is child B's input — the conditional reward is no longer a deterministic function of independent child outputs). Validation: [`proofs/inheritance-unbiasedness.tex`](../proofs/inheritance-unbiasedness.tex).

### 102. Per-child local-baseline variance reduction — MEASUREMENT

Subtract a per-state baseline $b(s_v)$ from the inherited advantage before applying the surrogate. Standard control-variate trick; under $(*)$ it's exactly unbiased and reduces variance whenever the baseline correlates with the score-function-weighted advantage. Validation: [`improvements/local-baseline.py`](../improvements/local-baseline.py) — toy bandit, expects ratio $\mathrm{Var}(\tilde A) / \mathrm{Var}(A) < 1$.

### 103. $k_g$ scale-law experiment — MEASUREMENT

Sweep $k_g \in \{1, 2, 4, 8, 16, 32\}$ at fixed compute budget; expect an interior optimum where the trade-off between per-parent reward signal and per-step batch diversity is balanced. Validation: [`improvements/k-scale-sweep.py`](../improvements/k-scale-sweep.py).

### 104. RLM children as options (Sutton-Precup-Singh) — PROOF

Map `rlm_query` calls to options $o = (I, \pi_o, \beta)$, lift the whole RLM training problem to an SMDP, and import SMDP Q-learning convergence (Sutton-Precup-Singh 1999 Theorem 1) under the natural conditions. The mapping is mostly a translation, so the "proof" sketches the correspondence and cites the source theorem. Validation: [`proofs/rlm-as-options.tex`](../proofs/rlm-as-options.tex).

## 5. What to do next

1. **Reproduce the toy bandit reward-variance gain.** Run `python3 improvements/local-baseline.py`; check that the printed `ratio < 1.0`. If it is not, the local baseline is mis-specified — debug.
2. **Run the $k_g$ sweep on a non-toy task.** The bandit interior-optimum is a small effect. Re-run the sweep on a real arxiv-paper retrieval workload (or your own) to see if the optimum is at $k = 4$, $k = 16$, or larger. The answer determines deployment cost.
3. **Read [`proofs/rlm-as-options.tex`](../proofs/rlm-as-options.tex) and identify which corollaries you most want to prove.** The SMDP framing imports a lot of theory for free; the question is which corollary (e.g., interruptibility, intra-option learning) is most actionable for the next-paper-up. Pick one and write a 5-page extension.
