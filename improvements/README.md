# Improvements — Prototypes for `05-improvements.tex`

This folder contains the runnable prototypes that back the four proposals in
`../05-improvements.tex` (Stage 6 of the study run on the alphaXiv blog
"Reinforcing Recursive Language Models").

## Files

| File | Backs section in `05-improvements.tex` | What it does |
|------|----------------------------------------|--------------|
| `local-baseline.py`  | *Code / Implementation Improvements* | Compares vanilla shared-policy GRPO vs. GRPO + per-child local baseline on a 3-paper x 4-passage tree bandit. Reports gradient-norm variance ratio and final reward. |
| `k-scale-sweep.py`   | *Experimental Extensions* | Sweeps $k_g \in \{1, 2, 4, 8\}$ at fixed compute budget $C = G \cdot (1 + k_g) = 32$. Reports final reward and steps-to-threshold per $k_g$. |
| `requirements.txt`   | -- | numpy only. |

The two remaining sections (*Mathematical Improvements* and
*Theoretical / Conceptual Connections*) are backed by proofs in
`../proofs/inheritance-unbiasedness.tex` and `../proofs/rlm-as-options.tex`
(written separately — see Agent D's deliverables).

## Run

```bash
pip install -r requirements.txt
python3 local-baseline.py
python3 k-scale-sweep.py
```

Both scripts are CPU-runnable, deterministic (seed = 0), and finish in
under 60 seconds.

## What to look for

### `local-baseline.py`

Headline metric is `ratio = var_with_baseline / var_no_baseline`, computed
over the per-step gradient norm. A ratio < 1.0 means the local baseline
reduced the variance of the signal that SGD actually consumes. On the
seed-0 run this ratio is ~0.87 and the final post-100-step reward improves
from ~0.41 to ~0.49.

A second diagnostic, `advantage_var_*`, reports the variance of the raw
$A_{g,i}$ values. This is *expected* to grow under the local baseline
(because we add a residual term to the parent advantage), and is reported
only as a sanity check — it is not the headline metric.

### `k-scale-sweep.py`

The output table shows monotone decline in final reward as $k_g$ grows
under the fixed-compute constraint. On this toy bandit the optimal
$k_g = 1$, because the toy has low per-paper noise and the GRPO baseline
benefits more from extra parent groups than from extra children. On a
real evidence-selection task we expect a knee around $k_g \in \{2, 4\}$
matching the blog's empirical pick — a future step would be to repeat the
sweep on a noisier bandit (more passages per paper) where the
information value of additional children is non-trivial.
