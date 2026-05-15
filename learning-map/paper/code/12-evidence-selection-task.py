"""12-evidence-selection-task.py — toy evidence-selection workload.

Generates a synthetic doc with a 'gold span', and a toy agent that picks K
spans from the doc. Shows that span-level F1 is brittle (multiple equivalent
gold spans), motivating the rubric-based reward of concept 13.

Run: python3 12-evidence-selection-task.py
"""
import random


GOLD = {"the cat sat", "a cat sat", "feline rested"}  # multiple valid answers


def f1(pred: set, gold_one: str) -> float:
    p_tokens = set(" ".join(pred).split())
    g_tokens = set(gold_one.split())
    inter = len(p_tokens & g_tokens)
    if inter == 0:
        return 0.0
    p = inter / len(p_tokens)
    r = inter / len(g_tokens)
    return 2 * p * r / (p + r)


def best_f1(pred: set) -> float:
    return max(f1(pred, g) for g in GOLD)


def toy_agent_picks(rng: random.Random, doc_chunks):
    return set(rng.sample(doc_chunks, k=2))


if __name__ == "__main__":
    rng = random.Random(0)
    doc_chunks = [
        "the cat sat",
        "the dog ran",
        "the sun rose",
        "feline rested",
        "stars shone bright",
    ]
    runs = [toy_agent_picks(rng, doc_chunks) for _ in range(8)]
    print("Span-level best-of-gold F1 (brittle, varies wildly):")
    for i, pred in enumerate(runs):
        score = best_f1(pred)
        print(f"  run {i}: pred={pred}  F1={score:.3f}")
    avg = sum(best_f1(p) for p in runs) / len(runs)
    print(f"\nMean F1 across 8 runs: {avg:.3f}")
    print("Brittleness motivates concept 13 (rubric LLM-judge reward).")
