"""13-rubric-llm-judge-reward.py — toy LLM-judge rubric reward.

A rule-based stand-in for an LLM judge: scores 'agent answer' against
'gold answer' on relevance + completeness + parsimony. Returns a value in
[0, 1] suitable for use as r_g in GRPO.

Run: python3 13-rubric-llm-judge-reward.py
"""
from typing import List


def relevance(pred: List[str], gold: List[str]) -> float:
    g = set(" ".join(gold).split())
    p = set(" ".join(pred).split())
    if not p:
        return 0.0
    return len(p & g) / len(p)


def completeness(pred: List[str], gold: List[str]) -> float:
    g = set(" ".join(gold).split())
    p = set(" ".join(pred).split())
    if not g:
        return 1.0
    return len(p & g) / len(g)


def parsimony(pred: List[str]) -> float:
    n = len(pred)
    if n <= 3:
        return 1.0
    return max(0.0, 1.0 - (n - 3) * 0.15)


def rubric_reward(pred: List[str], gold: List[str]) -> float:
    return (relevance(pred, gold) + completeness(pred, gold) + parsimony(pred)) / 3.0


if __name__ == "__main__":
    gold = ["the cat sat on the mat"]
    cases = [
        ["the cat sat"],
        ["the cat sat on the mat"],
        ["the dog ran away"],
        ["the cat sat", "the cat", "a cat", "feline", "small cat", "tabby"],
    ]
    for c in cases:
        r = rubric_reward(c, gold)
        print(f"pred={c}\n  -> reward={r:.3f}")
    print("\nKey property: smooth, multi-criteria, allows multiple valid answers.")
    print("This becomes r_g; the GRPO advantage is computed from these r_g across")
    print("the G rollouts of one prompt.")
