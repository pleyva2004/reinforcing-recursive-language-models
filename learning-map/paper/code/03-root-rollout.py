"""03-root-rollout.py — sample root rollouts and score them.

Demonstrates: G root rollouts of a toy 'find-the-target' task, only the root
final-answer earns reward (children get nothing). Prints the resulting reward
distribution.

Run: python3 03-root-rollout.py
"""
import random
from typing import List


TARGET = 7


def root_rollout(rng: random.Random) -> int:
    """Toy root: 'guess' a number in {0..9}; recursively guesses are the
    descendants but only the root's final guess scores."""
    final_guess = rng.randint(0, 9)
    return final_guess


def reward(answer: int) -> float:
    return 1.0 if answer == TARGET else 0.0


def grpo_batch(G: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    rewards = [reward(root_rollout(rng)) for _ in range(G)]
    return rewards


if __name__ == "__main__":
    rewards = grpo_batch(G=64, seed=0)
    n_correct = sum(1 for r in rewards if r > 0.5)
    print(f"Sampled G=64 root rollouts; correct guesses: {n_correct}/64")
    print(f"Mean reward: {sum(rewards) / len(rewards):.3f}")
    print("\nKey property: only the root rollout receives reward — every other")
    print("node in the eventual tree must learn from inherited signals.")
