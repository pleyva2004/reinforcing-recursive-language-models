#!/usr/bin/env python3
"""
real_rlm_lora.py — Real LoRA fine-tuning of Qwen 2.5 1.5B Instruct via the
shared-policy + child-inherits-parent-advantage GRPO loop from the alphaXiv
blog "Reinforcing Recursive Language Models".

This is the full-fat companion to the math-clean toy demos
(`toy_recursive_bandit.py`, `tiny_gpt_rlm.py`). The toys exist to make every
gradient inspectable; this script wires the same algorithm into a real
Hugging Face transformer, using LoRA for tractability on a single
Apple Silicon laptop (M4 Pro, 48 GB unified).

Math reference: see `02-math-deep-dive.md`, Section 2.
  - Per-rollout PPO-clipped surrogate     L_node(y, A; theta)
  - Group-relative advantage              A_g = (r_g - mu_r) / sigma_r
  - Per-root full loss with 1/k_g balance L_g^full = L_g^root + (1/k_g) * sum_i L_g,i^child

CLI
---
    python3 real_rlm_lora.py --steps 5      # smoke test  (~10 min on M4 Pro)
    python3 real_rlm_lora.py --train        # full run    (100-200 steps, 3-4 hr)
    python3 real_rlm_lora.py --eval         # load saved adapter, run eval episodes

Hardware
--------
This file is *importable* on any platform: if torch / transformers / peft
aren't installed it prints a clear install message and exits 0. Real
training requires Apple Silicon MPS (or a CUDA GPU).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 1. Soft imports + device detection
# ---------------------------------------------------------------------------

_MISSING: list[str] = []
try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised only on Linux without torch
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _MISSING.append("torch>=2.4")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _MISSING.append("transformers>=4.45")

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    _MISSING.append("peft>=0.12")


def _print_install_message_and_exit() -> None:
    msg = (
        "real_rlm_lora.py: required packages not installed: "
        + ", ".join(_MISSING)
        + "\n\n"
        "Install (Apple Silicon recommended):\n"
        "    pip install -r requirements.txt\n"
        "or minimally:\n"
        "    pip install torch>=2.4 transformers>=4.45 peft>=0.12 accelerate>=1.0\n"
    )
    print(msg)
    sys.exit(0)


def detect_device() -> str:
    """Return 'mps' on Apple Silicon, 'cuda' on NVIDIA, else 'cpu'."""
    if torch is None:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# 2. Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

PPO_EPS = 0.2
LR = 1e-5
WEIGHT_DECAY = 0.01

G_ROOTS = 4          # number of root rollouts per episode
K_CHILDREN = 2       # number of children per root (k_g)
MAX_NEW_ROOT = 64    # max tokens for root rollout (<rlm_query ...>...</rlm_query>)
MAX_NEW_CHILD = 96   # max tokens for child rollout (extracted passage)

DATA_DIR = Path(__file__).parent / "data" / "papers"
ADAPTER_DIR = Path(__file__).parent / "lora_adapter"
MANIFEST_PATH = Path(__file__).parent / "data" / "papers" / "manifest.json"


# ---------------------------------------------------------------------------
# 3. Episode dataclass + corpus loading
# ---------------------------------------------------------------------------


@dataclass
class CandidatePaper:
    paper_id: str
    title: str
    chunks_text: str  # concatenated chunks for this paper (truncated for prompt)


@dataclass
class Episode:
    query: str
    gold_paper_idx: int                 # index into candidate_papers (0..2)
    gold_passage_text: str              # the passage the child should extract
    candidate_papers: list[CandidatePaper]


def _read_manifest() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except Exception:
        return []


def _load_paper_chunks(paper_id: str, max_chars: int = 4000) -> str:
    """Concatenate all chunk_*.txt files for a paper, capped at max_chars."""
    parts: list[str] = []
    for path in sorted(DATA_DIR.glob(f"{paper_id}_chunk_*.txt")):
        try:
            parts.append(path.read_text(errors="ignore"))
        except Exception:
            continue
    text = "\n".join(parts)
    return text[:max_chars] if len(text) > max_chars else text


def _extract_gold_passage(text: str, query_keywords: list[str]) -> str:
    """Heuristic: pick the sentence in `text` most overlapping with keywords."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if 40 < len(s) < 400]
    if not sentences:
        return text[:200]
    best, best_score = sentences[0], -1.0
    kw_lower = [k.lower() for k in query_keywords]
    for s in sentences:
        sl = s.lower()
        score = sum(sl.count(k) for k in kw_lower)
        if score > best_score:
            best, best_score = s, score
    return best


def build_episodes(seed: int = 0) -> list[Episode]:
    """Construct evidence-selection episodes from the on-disk corpus.

    Each episode pairs a synthesized research question with one gold paper
    and (G_ROOTS - 1) distractors picked from the same manifest. The gold
    passage is the sentence in the gold paper most overlapping the query
    keywords; this is what the child rollout should extract.
    """
    rng = random.Random(seed)
    manifest = _read_manifest()
    if not manifest:
        return []

    papers: list[CandidatePaper] = []
    for entry in manifest:
        text = _load_paper_chunks(entry["arxiv_id"])
        if not text.strip():
            continue
        papers.append(
            CandidatePaper(
                paper_id=entry["arxiv_id"],
                title=entry.get("title", entry["arxiv_id"]),
                chunks_text=text,
            )
        )
    if len(papers) < 3:
        return []

    # Synthesize one episode per paper, using its title-tokens as the query.
    episodes: list[Episode] = []
    for gold in papers:
        title_tokens = [w for w in re.findall(r"[A-Za-z]{4,}", gold.title)][:4]
        if not title_tokens:
            title_tokens = ["method", "result"]
        query = (
            f"Find the passage that best explains the {' '.join(title_tokens).lower()} "
            f"contribution of this paper."
        )
        gold_passage = _extract_gold_passage(gold.chunks_text, title_tokens)
        # Sample 2 distractors
        others = [p for p in papers if p.paper_id != gold.paper_id]
        rng.shuffle(others)
        distractors = others[:2]
        candidates = [gold] + distractors
        rng.shuffle(candidates)
        gold_idx = candidates.index(gold)
        episodes.append(
            Episode(
                query=query,
                gold_paper_idx=gold_idx,
                gold_passage_text=gold_passage,
                candidate_papers=candidates,
            )
        )
    return episodes


# ---------------------------------------------------------------------------
# 4. Prompt templates
# ---------------------------------------------------------------------------


def _root_prompt(episode: Episode) -> str:
    """Build the parent prompt: pick which paper to query."""
    lines = [
        "You are a recursive research agent. Given a question and a list of "
        "candidate papers, decide which single paper most likely contains the "
        "answer, and emit a query targeting it.",
        "",
        f"Question: {episode.query}",
        "",
        "Candidate papers:",
    ]
    for i, p in enumerate(episode.candidate_papers):
        snippet = p.chunks_text[:300].replace("\n", " ")
        lines.append(f"  [{i}] {p.title} :: {snippet}...")
    lines += [
        "",
        "Respond with exactly one tag of the form:",
        '  <rlm_query paper="N">extract: ...</rlm_query>',
        "where N is the index of the chosen paper. Respond with the tag only.",
        "",
        "Answer: ",
    ]
    return "\n".join(lines)


def _child_prompt(episode: Episode, root_choice: int, child_query: str) -> str:
    """Build the child prompt: extract relevant passage from chosen paper."""
    paper = episode.candidate_papers[root_choice]
    return (
        "You are a recursive research sub-agent. Extract the single most "
        "relevant passage from the paper below that answers the query.\n\n"
        f"Paper title: {paper.title}\n"
        f"Paper text:\n{paper.chunks_text[:2000]}\n\n"
        f"Query: {child_query}\n\n"
        "Extracted passage (1-3 sentences, verbatim from the paper):\n"
    )


# ---------------------------------------------------------------------------
# 5. Rollout helpers
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    prompt_text: str
    full_ids: Any           # tensor [1, T_total]
    prompt_len: int         # number of prompt tokens
    gen_ids: Any            # tensor [1, T_gen]
    old_logprobs: Any       # tensor [T_gen]   (logprob under behaviour policy)
    decoded: str            # decoded generated text
    parsed: dict = field(default_factory=dict)


def _generate_one(
    model, tokenizer, prompt_text: str, max_new_tokens: int, device: str
) -> Rollout:
    """Sample one rollout and cache the per-token logprobs under the *current*
    (behaviour) policy. These are what the PPO ratio later divides by.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    full_ids = out.sequences  # [1, prompt_len + T_gen]
    gen_ids = full_ids[:, prompt_len:]
    # Compute old-logprobs by re-running a forward pass on the full sequence.
    with torch.no_grad():
        logits = model(full_ids).logits  # [1, T_total, V]
    # logit at position t predicts token t+1; gather logprobs of generated tokens
    shift_logits = logits[:, prompt_len - 1:-1, :]   # [1, T_gen, V]
    shift_targets = gen_ids                           # [1, T_gen]
    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    old_logprobs = log_probs_all.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
    decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return Rollout(
        prompt_text=prompt_text,
        full_ids=full_ids.detach(),
        prompt_len=prompt_len,
        gen_ids=gen_ids.detach(),
        old_logprobs=old_logprobs.squeeze(0).detach(),
        decoded=decoded,
    )


def _parse_root(decoded: str) -> dict:
    """Parse <rlm_query paper="N">child_query</rlm_query>. Robust to noise."""
    m = re.search(r"<rlm_query\s+paper=\"?(\d+)\"?>(.*?)</rlm_query>", decoded, re.S)
    if m:
        return {"paper_idx": int(m.group(1)), "child_query": m.group(2).strip()}
    # Fallback: try to find a bare digit at the start.
    digits = re.findall(r"\b([0-2])\b", decoded)
    return {
        "paper_idx": int(digits[0]) if digits else 0,
        "child_query": decoded.strip()[:120] or "extract relevant passage",
    }


def generate_root_rollout(model, tokenizer, episode: Episode, device: str) -> Rollout:
    prompt = _root_prompt(episode)
    rollout = _generate_one(model, tokenizer, prompt, MAX_NEW_ROOT, device)
    parsed = _parse_root(rollout.decoded)
    # Clamp to valid index range
    parsed["paper_idx"] = max(0, min(parsed["paper_idx"], len(episode.candidate_papers) - 1))
    rollout.parsed = parsed
    return rollout


def generate_child_rollout(
    model, tokenizer, episode: Episode, parent_choice: int, child_query: str, device: str
) -> Rollout:
    prompt = _child_prompt(episode, parent_choice, child_query)
    rollout = _generate_one(model, tokenizer, prompt, MAX_NEW_CHILD, device)
    rollout.parsed = {"extraction": rollout.decoded.strip()}
    return rollout


# ---------------------------------------------------------------------------
# 6. Reward + advantage
# ---------------------------------------------------------------------------


def _token_set_f1(a: str, b: str) -> float:
    """Cheap token-set F1 between two strings (lowercased word tokens)."""
    ta = {w for w in re.findall(r"[A-Za-z0-9]+", a.lower()) if len(w) > 2}
    tb = {w for w in re.findall(r"[A-Za-z0-9]+", b.lower()) if len(w) > 2}
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    p = inter / len(ta)
    r = inter / len(tb)
    return 2 * p * r / (p + r)


def _seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def compute_reward(episode: Episode, root_choice: int, child_extraction: str) -> float:
    """Reward in [0, 2]: +1 for correct paper, +1 for high-overlap extraction."""
    r = 0.0
    if root_choice == episode.gold_paper_idx:
        r += 1.0
    f1 = _token_set_f1(child_extraction, episode.gold_passage_text)
    sm = _seq_ratio(child_extraction[:400], episode.gold_passage_text[:400])
    score = max(f1, sm)
    if score > 0.3:
        r += 1.0
    elif score > 0.15:
        r += 0.5
    return r


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    """A_g = (r_g - mu) / (sigma + eps).  Standard GRPO group-relative advantage."""
    n = len(rewards)
    if n == 0:
        return []
    mu = sum(rewards) / n
    var = sum((r - mu) ** 2 for r in rewards) / max(1, n)
    sigma = math.sqrt(var)
    return [(r - mu) / (sigma + 1e-8) for r in rewards]


# ---------------------------------------------------------------------------
# 7. PPO-clipped loss
# ---------------------------------------------------------------------------


def _ppo_clipped_loss_for_rollout(
    model, rollout: Rollout, advantage: float, eps: float = PPO_EPS
):
    """L_node(y, A; theta): mean over generated tokens of the PPO-clipped surrogate.

    We re-run a forward pass with grad enabled to get current logprobs; the
    per-token ratio is exp(new_logp - old_logp); the surrogate is
        - mean( min( rho*A, clip(rho, 1-eps, 1+eps)*A ) )
    (negative because we want to maximize the surrogate).
    """
    full_ids = rollout.full_ids
    prompt_len = rollout.prompt_len
    gen_ids = rollout.gen_ids
    logits = model(full_ids).logits
    shift_logits = logits[:, prompt_len - 1:-1, :]
    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    new_logprobs = log_probs_all.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)
    old_logprobs = rollout.old_logprobs.to(new_logprobs.device)
    ratio = torch.exp(new_logprobs - old_logprobs)
    adv = torch.tensor(advantage, dtype=ratio.dtype, device=ratio.device)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
    per_token = torch.minimum(unclipped, clipped)
    return -per_token.mean()


def compute_loss(
    model,
    root_rollouts: list[Rollout],
    child_rollouts: list[list[Rollout]],
    advantages: list[float],
    k_g: int,
    eps: float = PPO_EPS,
):
    """Implements

        L = (1/G) sum_g [ L_node(y_g, A_g) + (1/k_g) sum_{i=1..k_g} L_node(y_{g,i}, A_g) ].

    See `02-math-deep-dive.md` Sections 2.2 and 2.3.
    """
    G = len(root_rollouts)
    losses = []
    for g in range(G):
        A_g = advantages[g]
        root_loss = _ppo_clipped_loss_for_rollout(model, root_rollouts[g], A_g, eps)
        child_losses = []
        for child in child_rollouts[g]:
            child_losses.append(_ppo_clipped_loss_for_rollout(model, child, A_g, eps))
        if child_losses:
            child_term = sum(child_losses) / k_g  # 1/k_g balancing
            losses.append(root_loss + child_term)
        else:
            losses.append(root_loss)
    return sum(losses) / max(1, G)


# ---------------------------------------------------------------------------
# 8. Model loading + LoRA setup
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(device: str):
    print(f"[load] device = {device}")
    print(f"[load] base model = {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True
    )
    model.to(device)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[lora] trainable params: {n_trainable:,} / {n_total:,} "
        f"({100 * n_trainable / n_total:.3f}%)"
    )
    return model, tokenizer


def _lora_grad_norm(model) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(sq)


# ---------------------------------------------------------------------------
# 9. Training loop
# ---------------------------------------------------------------------------


def train(steps: int, log_every: int = 5) -> None:
    if _MISSING:
        _print_install_message_and_exit()

    device = detect_device()
    model, tokenizer = load_model_and_tokenizer(device)

    episodes = build_episodes()
    if not episodes:
        print(
            "[error] no episodes found. Run `python3 sandbox/data/fetch_papers.py` "
            "first to populate sandbox/data/papers/."
        )
        sys.exit(1)
    print(f"[data] loaded {len(episodes)} episodes from {DATA_DIR}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    rng = random.Random(0)
    history: list[dict] = []
    t_start = time.time()
    for step in range(1, steps + 1):
        episode = rng.choice(episodes)

        # ------------------------------------------------------------------
        # 9a. Sampling phase: gradients OFF, just collect rollouts + old logp
        # ------------------------------------------------------------------
        torch.set_grad_enabled(False)
        root_rollouts: list[Rollout] = []
        child_rollouts: list[list[Rollout]] = []
        rewards: list[float] = []
        for g in range(G_ROOTS):
            root = generate_root_rollout(model, tokenizer, episode, device)
            root_rollouts.append(root)
            kids: list[Rollout] = []
            for _ in range(K_CHILDREN):
                kid = generate_child_rollout(
                    model,
                    tokenizer,
                    episode,
                    root.parsed["paper_idx"],
                    root.parsed["child_query"],
                    device,
                )
                kids.append(kid)
            child_rollouts.append(kids)
            # Reward: average over children's extractions
            extractions = [k.parsed["extraction"] for k in kids]
            r_root = max(
                compute_reward(episode, root.parsed["paper_idx"], e)
                for e in extractions
            )
            rewards.append(r_root)

        advantages = compute_grpo_advantages(rewards)

        # ------------------------------------------------------------------
        # 9b. Optimization phase: gradients ON, compute loss, step
        # ------------------------------------------------------------------
        torch.set_grad_enabled(True)
        optim.zero_grad(set_to_none=True)
        loss = compute_loss(
            model, root_rollouts, child_rollouts, advantages, k_g=K_CHILDREN
        )
        loss.backward()
        grad_norm = _lora_grad_norm(model)
        optim.step()

        rec = {
            "step": step,
            "mean_reward": sum(rewards) / len(rewards),
            "mean_abs_advantage": sum(abs(a) for a in advantages) / len(advantages),
            "loss": float(loss.detach().item()),
            "grad_norm": grad_norm,
            "elapsed_s": time.time() - t_start,
        }
        history.append(rec)
        if step % log_every == 0 or step == 1:
            print(
                f"[step {step:4d}] reward={rec['mean_reward']:.3f}  "
                f"|A|={rec['mean_abs_advantage']:.3f}  "
                f"loss={rec['loss']:+.4f}  gnorm={rec['grad_norm']:.4f}  "
                f"t={rec['elapsed_s']:.1f}s"
            )

    # Save adapter
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    print(f"[save] LoRA adapter -> {ADAPTER_DIR}")
    history_path = ADAPTER_DIR / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[save] training history -> {history_path}")

    # Quick before/after sample (post-training only; "before" is a separate run)
    print("\n[sample] post-training generation on 1 episode:")
    torch.set_grad_enabled(False)
    ep = episodes[0]
    root = generate_root_rollout(model, tokenizer, ep, device)
    print(f"  query     : {ep.query}")
    print(f"  gold paper: [{ep.gold_paper_idx}] {ep.candidate_papers[ep.gold_paper_idx].title}")
    print(f"  root pick : [{root.parsed['paper_idx']}]")
    print(f"  root text : {root.decoded[:240]!r}")


# ---------------------------------------------------------------------------
# 10. Eval
# ---------------------------------------------------------------------------


def evaluate() -> None:
    if _MISSING:
        _print_install_message_and_exit()
    device = detect_device()
    print(f"[eval] device = {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True
    )
    base.to(device)
    if (ADAPTER_DIR / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        model.to(device)
        print(f"[eval] loaded adapter from {ADAPTER_DIR}")
    else:
        model = base
        print(f"[eval] no adapter at {ADAPTER_DIR}; evaluating base model")

    episodes = build_episodes()
    if not episodes:
        print("[error] no episodes; run fetch_papers.py first.")
        sys.exit(1)

    correct = 0
    f1_sum = 0.0
    n = 0
    torch.set_grad_enabled(False)
    for ep in episodes:
        root = generate_root_rollout(model, tokenizer, ep, device)
        child = generate_child_rollout(
            model, tokenizer, ep, root.parsed["paper_idx"], root.parsed["child_query"], device
        )
        if root.parsed["paper_idx"] == ep.gold_paper_idx:
            correct += 1
        f1_sum += _token_set_f1(child.parsed["extraction"], ep.gold_passage_text)
        n += 1
    print(f"[eval] paper-pick accuracy = {correct}/{n} = {correct / n:.3f}")
    print(f"[eval] mean extraction F1  = {f1_sum / n:.3f}")


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None,
                        help="smoke-test step count (e.g. 5)")
    parser.add_argument("--train", action="store_true",
                        help="full training run (100-200 steps)")
    parser.add_argument("--eval", action="store_true",
                        help="load saved adapter and evaluate")
    parser.add_argument("--full-steps", type=int, default=150,
                        help="step count for --train (default 150)")
    args = parser.parse_args()

    if _MISSING:
        _print_install_message_and_exit()

    if args.eval:
        evaluate()
    elif args.train:
        train(steps=args.full_steps)
    elif args.steps is not None:
        train(steps=args.steps)
    else:
        parser.print_help()
        print("\nNo action specified. Try --steps 5 (smoke) or --train (full).")


if __name__ == "__main__":
    main()
