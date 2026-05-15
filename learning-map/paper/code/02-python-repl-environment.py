"""02-python-repl-environment.py — toy persistent-state REPL.

Implements a minimal Python REPL with FINAL / FINAL_VAR / rlm_query primitives.
The REPL state persists across turns. rlm_query is faked by a stub that returns
'CHILD_ANSWER' so we can demonstrate the protocol without a real LM.

Run: python3 02-python-repl-environment.py
"""
import io
import contextlib


class RLMRepl:
    def __init__(self):
        self.state = {}
        self.final = None
        self.children = []

    def _rlm_query(self, prompt, context=None):
        # Stub: in real usage this spawns a child rollout under pi_theta.
        self.children.append(("rlm_query", prompt))
        return f"CHILD_ANSWER({prompt!r})"

    def _final(self, ans):
        self.final = ans
        return ans

    def _final_var(self, name):
        self.final = self.state.get(name, None)
        return self.final

    def step(self, code):
        env = dict(self.state)
        env["FINAL"] = self._final
        env["FINAL_VAR"] = self._final_var
        env["rlm_query"] = self._rlm_query
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, env)
        # persist any new vars (skip privileged callables)
        for k, v in env.items():
            if k not in ("FINAL", "FINAL_VAR", "rlm_query", "__builtins__"):
                self.state[k] = v
        return buf.getvalue(), self.final


if __name__ == "__main__":
    r = RLMRepl()
    out, _ = r.step("x = 1 + 1\nprint('x =', x)")
    print("turn 1 stdout:", out.strip())
    out, _ = r.step("y = rlm_query('summarise chunk 1')\nprint('child answered:', y)")
    print("turn 2 stdout:", out.strip())
    print("turn 2 spawned children:", r.children)
    out, ans = r.step("FINAL_VAR('x')")
    print("turn 3 returned answer:", ans)
    print("\nKey property: REPL state persists; FINAL_VAR resolves variables")
    print("from that persistent state; rlm_query is the recursion primitive.")
