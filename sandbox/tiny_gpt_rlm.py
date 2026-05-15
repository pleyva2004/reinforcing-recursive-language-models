"""
tiny_gpt_rlm.py
===============

Adapt the Ch.27 tiny-GPT (numpy, hand-derived backprop) into a SHARED policy
for a synthetic RLM-style "decompose then extract" task. The same network plays
both decomposer (parent) and sub-agent (child) — exactly the architectural
choice the blog calls out: a single Qwen-3-4B model trained jointly.

Synthetic task
--------------
Vocab is character-level (28 chars). The prompt is:
        "find:X "
where X is one of {a..z}. The model then emits up to two sub-queries delimited
by "<" ... ">" — when the harness sees "<", it samples a stub passage from the
corpus, swaps in the next char, and feeds it back to the model. After up to two
such expansions the model emits a final answer "=A". Reward = +1 if A == X
else 0 (positive-only reward keeps GRPO stable on toy scale).

We treat the "<...>" calls as CHILD rollouts; tokens between calls are root
tokens. Every token's PG term gets the SAME group-relative advantage A_g, with
the 1/k_g averaging on the child block (matches L_g^full from 02-math-deep-
dive.md).

Simplifications (matching Ch.31 GRPO recipe)
- 1 transformer layer, vocab=28, d=16, T=24, dff=32
- transformer block frozen; only token-embedding E + position-embedding P are
  trained (E doubles as the unembedding via weight tying, so updating E
  updates the policy head). Cuts param count to ~600 trainable scalars.
- 30 GRPO steps, G=4 root rollouts per step, max 2 child calls per root.
- PPO clip eps=0.2; AdamW lr=0.05.

Run:  python3 tiny_gpt_rlm.py     (~30-120s on CPU)
"""

import math
import time
import copy
import numpy as np

np.random.seed(0)

# ----------------------------------------------------------------------------
# Vocab (char-level): a-z + the structural tokens "<", ">", "=", "?", " "
# ----------------------------------------------------------------------------
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
SPECIALS = list("<>= ")
CHARS = LETTERS + SPECIALS
V = len(CHARS)
stoi = {c: i for i, c in enumerate(CHARS)}
itos = {i: c for i, c in enumerate(CHARS)}
PAD = stoi[" "]
LT = stoi["<"]
GT = stoi[">"]
EQ = stoi["="]


def encode(s):
    return [stoi[c] for c in s if c in stoi]


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


# ----------------------------------------------------------------------------
# Tiny GPT (1 layer, hand-coded forward + backward — adapted from Ch.27)
# ----------------------------------------------------------------------------
d, H, dff, L = 16, 1, 32, 1
T = 24

def init_linear(fan_in, fan_out, scale=1.0):
    return (np.random.randn(fan_in, fan_out) * (scale / math.sqrt(fan_in))).astype(np.float64)


params = {}
params["E"] = (np.random.randn(V, d) * 0.05).astype(np.float64)  # tied head
params["P"] = (np.random.randn(T, d) * 0.05).astype(np.float64)
for l in range(L):
    params[f"ln1_g{l}"] = np.ones(d)
    params[f"ln1_b{l}"] = np.zeros(d)
    params[f"Wq{l}"] = init_linear(d, d, 0.5)
    params[f"Wk{l}"] = init_linear(d, d, 0.5)
    params[f"Wv{l}"] = init_linear(d, d, 0.5)
    params[f"Wo{l}"] = init_linear(d, d, 0.5)
    params[f"ln2_g{l}"] = np.ones(d)
    params[f"ln2_b{l}"] = np.zeros(d)
    params[f"W1{l}"] = init_linear(d, dff, 0.5)
    params[f"b1{l}"] = np.zeros(dff)
    params[f"W2{l}"] = init_linear(dff, d, 0.5)
    params[f"b2{l}"] = np.zeros(d)
params["lng"] = np.ones(d)
params["lnb"] = np.zeros(d)

# Freeze everything except E and P (the policy head + token table).
TRAINABLE = {"E", "P"}


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    return xhat * g + b


CAUSAL = (np.triu(np.ones((T, T)), k=1) * -1e9)


def forward_logits(p, x):
    """x: (B, T) int -> logits (B, T, V). No cache (we only need logits +
    ability to compute d logits / d E,P analytically below)."""
    B, Tlen = x.shape
    h = p["E"][x] + p["P"][None, :Tlen]
    for l in range(L):
        h_in = h
        h_n = layernorm(h, p[f"ln1_g{l}"], p[f"ln1_b{l}"])
        Q = h_n @ p[f"Wq{l}"]
        K = h_n @ p[f"Wk{l}"]
        Vv = h_n @ p[f"Wv{l}"]
        scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(d) + CAUSAL[None, :Tlen, :Tlen]
        attn = softmax(scores, axis=-1)
        ctx = attn @ Vv
        attn_out = ctx @ p[f"Wo{l}"]
        h2 = h_in + attn_out
        h2_n = layernorm(h2, p[f"ln2_g{l}"], p[f"ln2_b{l}"])
        z1 = h2_n @ p[f"W1{l}"] + p[f"b1{l}"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ p[f"W2{l}"] + p[f"b2{l}"]
        h = h2 + z2
    h_final = layernorm(h, p["lng"], p["lnb"])
    logits = h_final @ p["E"].T
    return logits, h_final


# ----------------------------------------------------------------------------
# Gradient of CE-style policy loss w.r.t. trainable params (E, P only).
# We compute d Loss / d logits, then chain through:
#   logits = h_final @ E^T   ->   dE += dlogits^T @ h_final + (token-row scatter from input embedding side)
#   The h_final dependence on E,P is through the input token+position embedding.
#   Since we freeze the transformer block, the dominant gradient is the head
#   path (dE +=  dlogits @ h_final). We additionally backprop the "input
#   embedding" gradient by re-running a finite-step approximation through the
#   top LayerNorm only (cheap and adequate for this toy scale).
# ----------------------------------------------------------------------------
def head_grad_E(h_final, dlogits):
    """logits = h_final @ E.T  =>  dE += sum_{b,t} dlogits[b,t,:].T outer h_final[b,t,:] / ... wait
    dlogits has shape (B, T, V); h_final (B, T, d).
    dE has shape (V, d). dE[v, :] += sum_{b,t} dlogits[b,t,v] * h_final[b,t,:].
    """
    B, Tlen, _ = dlogits.shape
    return dlogits.reshape(-1, V).T @ h_final.reshape(-1, d)


def input_grad_E_P(p, x, dlogits, h_final):
    """Approximate gradient through the input embedding via the residual
    bypass. Because we freeze the transformer block, we treat the input
    embedding as if it flowed straight through the residual path: i.e. the
    contribution of E[x_t] + P[t] to h_final[t] is approximately the identity
    plus the LayerNorm linearisation. We use the simple straight-through
    approximation dE[x_t,:] += g * (dlogits @ E)[t] where g=1/(LayerNorm
    scale) ~ 1.

    This is an intentional simplification (matches Ch.31's "freeze block,
    train head" recipe). The head_grad_E term above is the dominant signal.
    """
    B, Tlen, _ = dlogits.shape
    # gradient flowing back from logits into h_final
    dh_final = dlogits @ p["E"]            # (B, T, d)
    # straight-through into the input embedding
    dh_in = dh_final
    dE = np.zeros_like(p["E"])
    # Scatter into E rows by token id (the input-embedding contribution)
    np.add.at(dE, x, dh_in)
    dP = dh_in.sum(axis=0)
    return dE, dP


# ----------------------------------------------------------------------------
# AdamW (only over TRAINABLE params)
# ----------------------------------------------------------------------------
class AdamW:
    def __init__(self, params, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.0):
        self.m = {k: np.zeros_like(v) for k, v in params.items() if k in TRAINABLE}
        self.v = {k: np.zeros_like(v) for k, v in params.items() if k in TRAINABLE}
        self.b1, self.b2, self.eps, self.wd = beta1, beta2, eps, wd
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k in TRAINABLE:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            update = (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)
            if self.wd:
                update = update + self.wd * params[k]
            params[k] -= lr * update


# ----------------------------------------------------------------------------
# Sampling: roll out a sequence one token at a time. We collect:
#   - the (T,) input window the model sees at each step
#   - the chosen token
#   - the log-prob under theta_old (cached at sample time)
#   - whether this token belongs to a "child" block (between < and >)
# ----------------------------------------------------------------------------
def pad_left(ids, length=T):
    if len(ids) >= length:
        return ids[-length:]
    return [PAD] * (length - len(ids)) + list(ids)


def sample_token(p, prefix_ids, rng, temperature=1.0):
    x = np.array([pad_left(prefix_ids)], dtype=np.int64)
    logits, _ = forward_logits(p, x)
    last = logits[0, -1] / temperature
    pr = softmax(last)
    a = int(rng.choice(V, p=pr))
    return a, float(np.log(pr[a] + 1e-12))


def stub_passage(rng, target):
    """The harness response to a child query. With probability 0.6 it
    contains the target letter (shaped to make the task learnable)."""
    if rng.random() < 0.6:
        return target
    return rng.choice(LETTERS)


def rollout(p, rng, target_letter, max_tokens=20, max_calls=2):
    """Generate one root rollout. Returns prompt+continuation tokens, plus
    per-token (input-window, chosen-token, logp_old, is_child) records."""
    prompt = encode(f"find:{target_letter} ")
    seq = list(prompt)
    records = []
    in_call = False
    n_calls = 0
    final_answer = None

    for _ in range(max_tokens):
        if len(seq) >= T:
            break
        a, lp = sample_token(p, seq, rng)
        records.append(dict(window=pad_left(seq), token=a,
                            logp_old=lp, is_child=in_call))
        seq.append(a)

        ch = itos[a]
        if ch == "<" and not in_call and n_calls < max_calls:
            in_call = True
        elif ch == ">" and in_call:
            in_call = False
            n_calls += 1
            # Inject a stub passage character (the "REPL response").
            stub_char = stub_passage(rng, target_letter)
            stub_id = stoi[stub_char]
            seq.append(stub_id)  # not a sampled token: no PG term
            # NOTE: we treat the entire <...> block above as one child
            # rollout. The stub injection is environmental (not under the
            # policy), so it does not get a PG term.
        elif ch == "=":
            # Take the next sampled letter as the answer.
            a2, lp2 = sample_token(p, seq, rng)
            records.append(dict(window=pad_left(seq), token=a2,
                                logp_old=lp2, is_child=False))
            seq.append(a2)
            final_answer = itos[a2]
            break

    if final_answer is None:
        # Force an = + answer at the end.
        for _ in range(2):
            if len(seq) >= T:
                break
            a, lp = sample_token(p, seq, rng)
            records.append(dict(window=pad_left(seq), token=a,
                                logp_old=lp, is_child=False))
            seq.append(a)
        # Treat the last sampled letter as the answer.
        for c in reversed(seq):
            if itos[c] in LETTERS:
                final_answer = itos[c]
                break

    # Shaped reward (denser signal, in [0, 1]):
    #   +0.5 if the model emitted "=" (used the answer protocol)
    #   +0.5 if the answer letter matches the target
    #   +0.2 partial: any occurrence of the target letter anywhere after the prompt
    used_eq = EQ in seq[len(encode(f'find:{target_letter} ')):]
    target_id = stoi[target_letter]
    saw_target = target_id in seq[len(encode(f'find:{target_letter} ')):]
    reward = 0.0
    if used_eq:
        reward += 0.5
    if final_answer == target_letter:
        reward += 0.5
    elif saw_target:
        reward += 0.2
    return dict(seq=seq, records=records, reward=reward,
                k_g=max(n_calls, 1), n_calls=n_calls)


# ----------------------------------------------------------------------------
# Full RLM-GRPO loss for one batch of G rollouts.
#   For each rollout g, every sampled token contributes a PPO term with
#   advantage A_g. Child-block tokens get a 1/k_g averaging factor.
# ----------------------------------------------------------------------------
EPS = 0.2


def compute_grad_and_loss(p, rollouts, advantages):
    grads = {k: np.zeros_like(v) for k, v in p.items() if k in TRAINABLE}
    total_loss = 0.0
    total_kl = 0.0
    n_tokens = 0

    # Bucket per-rollout records into a flat batch for one forward pass.
    windows, tokens, logp_olds, weights = [], [], [], []
    for r, A in zip(rollouts, advantages):
        kg = max(r["k_g"], 1)
        for rec in r["records"]:
            windows.append(rec["window"])
            tokens.append(rec["token"])
            logp_olds.append(rec["logp_old"])
            w = A
            if rec["is_child"]:
                w = w / kg     # 1/k_g balancing trick
            weights.append(w)

    if not windows:
        return grads, 0.0, 0.0

    x = np.array(windows, dtype=np.int64)            # (N, T)
    tokens = np.array(tokens, dtype=np.int64)        # (N,)
    logp_olds = np.array(logp_olds, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)

    logits, h_final = forward_logits(p, x)            # (N, T, V), (N, T, d)
    last_logits = logits[:, -1, :]                    # (N, V)
    last_h = h_final[:, -1, :]                        # (N, d)
    last_pr = softmax(last_logits, axis=-1)
    logp_new = np.log(last_pr[np.arange(len(tokens)), tokens] + 1e-12)

    ratio = np.exp(logp_new - logp_olds)
    clipped = np.clip(ratio, 1.0 - EPS, 1.0 + EPS)

    surrogate_unclipped = ratio * weights
    surrogate_clipped = clipped * weights
    surrogate = np.minimum(surrogate_unclipped, surrogate_clipped)
    loss_per_tok = -surrogate
    total_loss = float(loss_per_tok.mean())

    # Gradient of -surrogate w.r.t. last_logits.
    # For tokens where unclipped is the min and we are inside the clip
    # interval, d(ratio*w)/d logits = ratio * w * (e_a - softmax(logits)).
    # When clipped branch wins, gradient is zero (standard PPO).
    inside = (1.0 - EPS < ratio) & (ratio < 1.0 + EPS)
    use_grad = inside | (surrogate_unclipped <= surrogate_clipped)
    coeffs = np.where(use_grad, ratio * weights, 0.0)

    # d log pi(a) / d logits = (e_a - p)
    dlogits_last = -last_pr.copy()
    dlogits_last[np.arange(len(tokens)), tokens] += 1.0
    dlogits_last *= coeffs[:, None]
    # We are minimising -surrogate, so the sign on the gradient is -coeffs *
    # (e_a - p). Equivalently dL/d logits_last = -dlogits_last as built.
    dlogits_last = -dlogits_last
    dlogits_last /= len(tokens)  # mean over tokens

    # Re-shape into a full (N, T, V) gradient that is zero except at t=T-1.
    dlogits_full = np.zeros_like(logits)
    dlogits_full[:, -1, :] = dlogits_last

    # Head path: logits = h_final @ E^T  =>  dE += dlogits^T @ h_final
    grads["E"] = head_grad_E(h_final, dlogits_full)
    # Input-embedding path (straight-through approximation, dominant for
    # the position embedding P and a secondary contribution to E).
    dE_in, dP_in = input_grad_E_P(p, x, dlogits_full, h_final)
    grads["E"] = grads["E"] + dE_in
    grads["P"] = dP_in

    n_tokens = len(tokens)
    # KL(theta_old || theta_new) approximation per token: (logp_old - logp_new)
    # is +ve when new policy assigns less mass; we use mean absolute as a soft
    # KL proxy.
    total_kl = float(np.mean(logp_olds - logp_new))
    return grads, total_loss, (total_kl, n_tokens)


# ----------------------------------------------------------------------------
# KL-to-initial diagnostic (full categorical KL averaged over a probe set).
# ----------------------------------------------------------------------------
def categorical_kl_to(p_old, p_new, probe_ids):
    """Average KL(p_old || p_new) over a probe of input windows."""
    x = np.array(probe_ids, dtype=np.int64)
    l_old, _ = forward_logits(p_old, x)
    l_new, _ = forward_logits(p_new, x)
    p1 = softmax(l_old[:, -1, :], axis=-1)
    p2 = softmax(l_new[:, -1, :], axis=-1)
    return float((p1 * (np.log(p1 + 1e-12) - np.log(p2 + 1e-12))).sum(axis=-1).mean())


# ----------------------------------------------------------------------------
# Eval helper
# ----------------------------------------------------------------------------
def eval_reward(p, rng, n=32):
    rewards = []
    for _ in range(n):
        target = rng.choice(LETTERS)
        r = rollout(p, rng, target)
        rewards.append(r["reward"])
    return float(np.mean(rewards))


def sample_completion(p, target_letter, seed=0):
    rng = np.random.default_rng(seed)
    r = rollout(p, rng, target_letter)
    return decode(r["seq"]), r["reward"], r["n_calls"]


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("tiny_gpt_rlm.py — RLM-GRPO on a tiny char-level GPT (shared policy)")
    print("=" * 64)
    print(f"V={V}  d={d}  L={L}  T={T}  trainable params={sum(params[k].size for k in TRAINABLE)}")

    # Snapshot initial policy (for KL diagnostic)
    init_params = copy.deepcopy(params)

    rng = np.random.default_rng(0)

    # Build a small probe set (random prefixes) for KL eval.
    probe_windows = []
    for _ in range(16):
        plen = int(rng.integers(2, T))
        probe_windows.append(pad_left(list(rng.integers(0, V, size=plen).tolist())))

    # Initial reward
    init_reward = eval_reward(init_params, np.random.default_rng(99), n=32)
    print(f"\n--- Initial samples (pre-training) ---")
    for tgt in ["a", "g", "z"]:
        s, r, k = sample_completion(init_params, tgt, seed=42)
        print(f"  target={tgt!r}  reward={r:.0f}  k_calls={k}  rollout={s!r}")
    print(f"\ninitial reward (32 trials): {init_reward:.3f}")

    # GRPO loop
    opt = AdamW(params, wd=0.0)
    G = 8
    STEPS = 50
    LR = 0.1
    history = []
    t0 = time.time()
    for step in range(STEPS):
        # 1. Sample G rollouts with theta_old = current params (snapshot logp_old at sample time)
        rollouts = []
        for _ in range(G):
            target = rng.choice(LETTERS)
            r = rollout(params, rng, target)
            r["target"] = target
            rollouts.append(r)
        rewards = np.array([r["reward"] for r in rollouts])
        mu = rewards.mean()
        sd = rewards.std() + 1e-8
        advantages = (rewards - mu) / sd

        # 2. Compute full-loss gradient and AdamW step
        grads, loss, (kl_step, ntok) = compute_grad_and_loss(params, rollouts, advantages)
        # global-norm clip on trainable grads
        sq = sum((grads[k] ** 2).sum() for k in TRAINABLE)
        gn = float(np.sqrt(sq))
        clip_c = 1.0
        if gn > clip_c:
            for k in TRAINABLE:
                grads[k] *= clip_c / (gn + 1e-12)
        opt.step(params, grads, LR)

        history.append(dict(step=step, reward=float(rewards.mean()),
                            loss=loss, gn=gn, kl=kl_step, ntok=ntok))
        if step % 5 == 0 or step == STEPS - 1:
            print(f"step {step:3d}  reward {rewards.mean():.3f}  "
                  f"loss {loss:+.4f}  gnorm {gn:.3f}  kl_step {kl_step:+.4f}  "
                  f"tokens {ntok}")

    elapsed = time.time() - t0
    print(f"\ntraining done in {elapsed:.1f}s  ({STEPS} GRPO steps, G={G})")

    # Final reward (averaged over a fresh batch)
    final_reward = eval_reward(params, np.random.default_rng(101), n=32)

    # KL to initial policy
    kl_init = categorical_kl_to(init_params, params, probe_windows)

    print(f"\n--- Final samples (post-training) ---")
    for tgt in ["a", "g", "z"]:
        s, r, k = sample_completion(params, tgt, seed=42)
        print(f"  target={tgt!r}  reward={r:.0f}  k_calls={k}  rollout={s!r}")

    print(f"\n=== Validation ===")
    print(f"initial reward (32 trials)            : {init_reward:.3f}")
    print(f"final reward   (32 trials)            : {final_reward:.3f}")
    print(f"KL(initial_policy || final_policy)    : {kl_init:.4f}")
    print(f"runtime                               : {elapsed:.1f}s")

    # Reward curve (matplotlib if available, else table)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [h["step"] for h in history]
        rs = [h["reward"] for h in history]
        # smoothed
        win = 5
        sm = [np.mean(rs[max(0, i - win): i + 1]) for i in range(len(rs))]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(steps, rs, alpha=0.3, label="raw reward")
        ax.plot(steps, sm, label=f"smoothed (w={win})")
        ax.set_xlabel("GRPO step")
        ax.set_ylabel("reward")
        ax.set_title("tiny GPT RLM training")
        ax.legend()
        out = "/tmp/tiny_gpt_rlm_reward.png"
        fig.tight_layout()
        fig.savefig(out)
        print(f"wrote reward curve to {out}")
    except Exception as e:
        print(f"matplotlib unavailable ({e}); reward table:")
        for i in range(0, len(history), 5):
            print(f"  step {history[i]['step']:3d}  reward {history[i]['reward']:.3f}")

    print(f"\nRESULT init_reward={init_reward:.4f} "
          f"final_reward={final_reward:.4f} kl_to_init={kl_init:.4f} "
          f"runtime_s={elapsed:.2f}")
    return init_reward, final_reward, kl_init, elapsed


if __name__ == "__main__":
    main()
