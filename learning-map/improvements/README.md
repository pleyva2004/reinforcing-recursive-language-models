# Improvements concept graph

Four improvement proposals that build on top of the paper's ideas. Each links to a concept page (with a runnable code witness) and a validation file (proof or measurement).

```mermaid
graph TD
    classDef intermediate fill:#ffe9b3,stroke:#b88a00,color:#000
    classDef advanced fill:#ffc9c9,stroke:#a02525,color:#000

    p9["paper:9 advantage-inheritance"]
    p10["paper:10 1/k_g averaging"]
    p11["paper:11 recursive subtree loss"]
    p4["paper:4 child-rollout"]

    n101["101. Tighter unbiasedness theorem (PROOF)"]:::advanced
    n102["102. Per-child local-baseline (MEASUREMENT)"]:::advanced
    n103["103. k_g scale-law (MEASUREMENT)"]:::intermediate
    n104["104. RLM children as options (PROOF)"]:::advanced

    p9 --> n101
    p10 --> n101
    p9 --> n102
    p10 --> n103
    p4 --> n104
    p11 --> n104

    click n101 "concepts/101-tighter-inheritance-unbiasedness.md"
    click n102 "concepts/102-local-baseline-variance-reduction.md"
    click n103 "concepts/103-kg-scale-law.md"
    click n104 "concepts/104-rlm-as-options.md"
```

## Validation modes

- **PROOF (math-flavoured improvements):**
  - 101 → [`../../proofs/inheritance-unbiasedness.tex`](../../proofs/inheritance-unbiasedness.tex)
  - 104 → [`../../proofs/rlm-as-options.tex`](../../proofs/rlm-as-options.tex)
- **MEASUREMENT (experiment-flavoured improvements):**
  - 102 → [`../../improvements/local-baseline.py`](../../improvements/local-baseline.py) (Agent C)
  - 103 → [`../../improvements/k-scale-sweep.py`](../../improvements/k-scale-sweep.py) (Agent C)

See [`../tour.md`](../tour.md) §4 "Improvements walk" for the per-improvement narrative.
