"""04-child-rollout.py — child rollout under shared policy.

Models a child rollout that runs under the same policy pi_theta as the parent.
Child receives no scalar reward; its return value is fed back into the parent.
Demonstrates the spawn-and-resume protocol.

Run: python3 04-child-rollout.py
"""
import random


def policy_sample(prompt: str, rng: random.Random) -> str:
    """Stand-in for pi_theta(.|prompt). Returns a plausible 'answer'."""
    if "summarise" in prompt:
        return rng.choice(["foo-summary", "bar-summary"])
    return "ok"


def child_rollout(prompt: str, rng: random.Random) -> str:
    """A child runs end-to-end and returns its FINAL answer."""
    # Pretend turn 1: emit FINAL(answer)
    return policy_sample(prompt, rng)


def parent_with_one_child(rng: random.Random):
    # Parent decides to call rlm_query at some point.
    sub = child_rollout("summarise chunk 1", rng)
    parent_answer = f"answer-using-{sub}"
    return parent_answer, sub


if __name__ == "__main__":
    rng = random.Random(0)
    answer, sub = parent_with_one_child(rng)
    print(f"Parent FINAL: {answer}")
    print(f"Child FINAL  : {sub}  (no reward computed for this child)")
    print("\nKey property: child has no scalar reward of its own — its only")
    print("learning signal will be the parent's GRPO advantage, inherited.")
