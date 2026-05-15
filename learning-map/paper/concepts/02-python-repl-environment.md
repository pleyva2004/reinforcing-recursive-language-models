# Python REPL agent environment

**Level:** intro
**Prerequisites:** [01-rlm-defn](01-rlm-defn.md)
**Used by:** [03-root-rollout](03-root-rollout.md)

## Plain-English intro

The agent does not "talk" in free-form text. Each turn it emits a Python code block; the harness executes that block in a persistent REPL and feeds back the captured stdout/stderr. Three primitives are exposed: `FINAL(answer)`, `FINAL_VAR(name)`, and `rlm_query(prompt, context=None)`. The REPL state — variables, imports, partial computations — persists across turns within a node, so the agent can chain computations like a human would in a notebook.

## Formal definition

Let $\mathcal{S}$ be the set of REPL states (Python heap snapshots). At turn $t$ the model output $y_t$ is parsed into a code block $c_t$ and executed: $s_{t+1} = \mathrm{exec}(c_t, s_t)$, producing observation $o_{t+1} \subseteq \mathrm{stdout} \cup \mathrm{stderr}$. Three privileged calls extend the semantics:

- `FINAL(a)` sets a *terminal flag* and returns $a$ as the rollout's answer.
- `FINAL_VAR(name)` sets the same flag with $a := s_{t+1}[name]$.
- `rlm_query(p, c)` is intercepted; the harness spawns a child rollout with prompt $p$ and returns its `FINAL` answer to the parent's REPL.

## Why this matters for the paper

Code-execution observations are *deterministic given the code* — this gives sharp, reproducible reward signal without the noise of natural-language tool calls, and lets the harness implement `rlm_query` as a clean recursive-spawn primitive instead of a fuzzy text protocol.

## Code

See [`../code/02-python-repl-environment.py`](../code/02-python-repl-environment.py).

## Cross-link to the chain

Generalises chain Ch 29 (`pleyva2004/first-principles-to-llms` Ch 29 — *MDP foundations*) by giving the action-space concrete Python semantics rather than abstract action symbols.
