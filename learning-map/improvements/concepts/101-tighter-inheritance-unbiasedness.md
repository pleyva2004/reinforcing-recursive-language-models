# Tighter unbiasedness theorem for advantage inheritance

**Level:** advanced
**Prerequisites:** [09-advantage-inheritance](../../paper/concepts/09-advantage-inheritance.md), [10-kg-averaging](../../paper/concepts/10-kg-averaging.md)
**Used by:** none (terminal — this *is* the math improvement)

## Plain-English intro

The blog asserts informally that inheritance is "unbiased in the same sense as GRPO". The improvement: state and prove the precise conditional-independence assumption $(*)$ under which inheritance is exactly unbiased, and quantify the bias when it breaks.

## Formal statement

**Theorem (PROOF).** Let $\mathcal{T}$ be an RLM tree with root $g$, and let $\hat g_v(\theta) = \nabla_\theta \log \pi_\theta(y_v) \cdot A_g$ be the inheritance estimator at node $v$. Assume

$$
(*)\quad \mathbb{E}\Big[\nabla_\theta \log \pi_\theta(y_v)\, r_g\Big] = \mathbb{E}\Big[\nabla_\theta \log \pi_\theta(y_v)\, \mathbb{E}[r_g \mid \mathrm{ancestors}(v), y_v]\Big],
$$

i.e., the score function at $v$ is uncorrelated with conditional reward fluctuations not explained by the ancestor history. Then

$$
\mathbb{E}\Big[\sum_{v \in \mathcal{T}_g} \frac{1}{w_v} \hat g_v(\theta)\Big] = \nabla_\theta J(\theta),
$$

with $w_v$ the accumulated $1/k$ weights along the path to $v$.

The full proof is in [`../../../proofs/inheritance-unbiasedness.tex`](../../../proofs/inheritance-unbiasedness.tex).

## Why this matters

Pins down exactly when the blog's headline credit-assignment trick is unbiased and identifies the failure modes (non-additive children, tool-use chains where child A's output is child B's input). Sharpens the open question §7.1 of `02-math-deep-dive.md`.

## Code

See [`../code/101-tighter-inheritance-unbiasedness.py`](../code/101-tighter-inheritance-unbiasedness.py) — numerical verification on a finite witness.

## Cross-link to the chain

Extends chain Ch 31's baseline-doesn't-bias proof from a sequence to a tree.
