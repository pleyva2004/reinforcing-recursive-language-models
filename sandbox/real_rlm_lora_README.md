# `real_rlm_lora.py` — real LoRA RL fine-tuning of Qwen 2.5 1.5B

This is the full-fat companion to the math-clean toy demos
(`toy_recursive_bandit.py`, `tiny_gpt_rlm.py`). The toys exist to make every
gradient inspectable; this script wires the same algorithm into a real
Hugging Face transformer using LoRA, so it can run on a single Apple
Silicon laptop (M4 Pro, 48 GB unified).

## What the script does

A near-reproduction of the alphaXiv blog "Reinforcing Recursive Language
Models", scaled down to fit a laptop:

| Aspect                | Blog (NovaSky-AI / SkyRL)         | This script                   |
|-----------------------|------------------------------------|-------------------------------|
| Base model            | Qwen3.5-4B                         | **Qwen2.5-1.5B-Instruct**     |
| Hardware              | 8x H200                            | **single M4 Pro (48 GB MPS)** |
| Group size G          | 8                                  | **4**                         |
| Children per root k_g | up to 4                            | **2**                         |
| Reward                | LLM-judge rubric (separate model)  | **char/token-set F1 + paper-pick**|
| Adapter               | full fine-tune                     | **LoRA r=16, alpha=32**       |

The training loss is *exactly* the per-root + per-child PPO-clipped
surrogate from `02-math-deep-dive.md` Section 2, with `1/k_g` averaging
and child-inherits-parent-advantage.

## Install

```bash
cd sandbox
pip install -r requirements.txt
```

The first run downloads ~3 GB for the Qwen 2.5 1.5B Instruct weights.

Optional, for the corpus:

```bash
python3 data/fetch_papers.py        # ~5 minutes, ~50 MB of PDFs
```

## Run

```bash
python3 real_rlm_lora.py --steps 5    # smoke test          (~10 min on M4 Pro)
python3 real_rlm_lora.py --train      # full run, 100-200 steps (~3-4 hr on M4 Pro)
python3 real_rlm_lora.py --eval       # load saved adapter, run eval episodes
```

The trained LoRA adapter is saved to `sandbox/lora_adapter/`.
A JSON training history (per-step reward, advantage, loss, grad norm) is
saved alongside it.

## Wall-clock template (USER fills in actual numbers)

| step count | wall-clock      | mean reward at end |
|------------|-----------------|---------------------|
| 5 (smoke)  | ___ min         | ___                 |
| 50         | ___ min         | ___                 |
| 150 (full) | ___ hr ___ min  | ___                 |

## Troubleshooting

- **`OutOfMemoryError` on MPS.** Reduce `G_ROOTS` (4 -> 2) or `K_CHILDREN`
  (2 -> 1) at the top of `real_rlm_lora.py`. As a last resort, lower
  `MAX_NEW_ROOT` / `MAX_NEW_CHILD`.
- **`RuntimeError: bf16 ... not supported`.** PyTorch < 2.4 has flaky bf16
  on MPS. Either upgrade PyTorch, or change `dtype = torch.bfloat16` to
  `torch.float32` in `load_model_and_tokenizer`.
- **`No episodes found`.** Run `python3 data/fetch_papers.py` first.
- **Loss is `nan`.** Likely a degenerate ratio in PPO. Check that
  `MAX_NEW_*` aren't producing extremely long sequences. Consider
  lowering `LR` from `1e-5` to `5e-6`.
- **`ImportError: torch / transformers / peft`.** The script is designed
  to exit gracefully with an install message in this case (so it is safe
  to run on a Linux orchestrator without GPU). Install via
  `pip install -r requirements.txt`.

## Mapping back to the math (`02-math-deep-dive.md`)

| Math object                                              | Code site                                  |
|----------------------------------------------------------|--------------------------------------------|
| `pi_theta` (parent + child)                              | `model` (single shared `PeftModel`)        |
| Old logp `pi_theta_old`                                  | `Rollout.old_logprobs`                     |
| Group-relative advantage `A_g`                           | `compute_grpo_advantages`                  |
| PPO-clipped surrogate `L_node`                           | `_ppo_clipped_loss_for_rollout`            |
| Per-root full loss with `1/k_g` averaging                | `compute_loss`                             |
| Child-inherits-parent-advantage                          | `compute_loss`: `A_g` reused for children  |

The toy demos use closed-form softmax gradients; this script uses PyTorch
autograd over LoRA parameters. The math is identical.
