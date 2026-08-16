"""Where does the gate's RUN-LEVEL variance come from?

Re-scoring one checkpoint gave -4.37 then -6.97. Per-cell scatter matched binomial theory
(sd 5.10 vs 5.77 predicted), but the 21 cell differences were one-sided (13 negative, the big
ones -11.3/-10.7/-10.7/-8.0 against +7.3 as the largest positive), mean -2.60pt where
independence predicts SE 1.11pt. Something moves every cell together, and that kind of error
does NOT shrink when you add games per cell.

The scoring path has exactly three places it can enter, so test them in order and stop at the
first that fails -- each stage conditions on the previous one being clean:

  1. PROMPT     is the rendered prompt identical for the same state?
                (rl_config.PROMPT_FMT["deck_shuffle"] randomises the DECK[] segment)
  2. LOGITS     given byte-identical prompts, does the model return identical scores?
                (bf16 / nondeterministic kernels / a fresh load each run)
  3. ARGMAX     given identical logits, is the chosen candidate identical?
                (tie-breaking order when two candidates score the same)

Run:  CUDA_VISIBLE_DEVICES=0 python /root/diag_run_variance.py
"""
import gzip
import json
import os
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

N_DECISIONS = 200
ROLLOUT = "/root/out/rlDL/A_r1.jsonl.gz"


def stage1_prompt():
    print("=" * 72)
    print("STAGE 1 -- is the rendered prompt stable for a fixed state?")
    print("=" * 72)
    import library
    import rl_config
    from lm.serialize import serialize_stateless
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent

    fmt = dict(rl_config.PROMPT_FMT)
    print("PROMPT_FMT:", fmt)
    dl = library.read_deck("alakazam")
    ol = library.read_deck("dragapult")
    agent = make_lm_agent("alakazam", None, None)
    obs, _ = battle_start(dl, ol)
    try:
        for _ in range(30):
            cur = obs.get("current")
            if cur is None or cur.get("result", -1) != -1 or obs.get("select") is None:
                break
            if cur["turn"] >= 2 and len(obs["select"].get("option") or []) >= 3:
                break
            obs = battle_select(agent(obs))
        renders = [serialize_stateless(obs, deck_ids=dl, deck_name="alakazam", **fmt)
                   for _ in range(8)]
    finally:
        battle_finish()
    uniq = len(set(renders))
    print("  8 renders of ONE state -> %d distinct prompts" % uniq)
    if uniq > 1:
        a, b = renders[0], next(r for r in renders if r != renders[0])
        # show the first line that differs, trimmed
        for la, lb in zip(a.split("\n"), b.split("\n")):
            if la != lb:
                print("  first differing line:")
                print("    A: %s" % la[:110])
                print("    B: %s" % lb[:110])
                break
        print("  VERDICT: prompt rendering is NON-DETERMINISTIC for a fixed state.")
        print("           Every eval run therefore feeds the policy different inputs.")
    else:
        print("  VERDICT: prompt rendering is deterministic. Move to stage 2.")
    return uniq == 1


def _load_pairs(n):
    out = []
    for line in gzip.open(ROLLOUT, "rt"):
        d = json.loads(line)
        if len(d.get("cands") or []) >= 2:
            out.append((d["prompt"], d["cands"]))
        if len(out) >= n:
            break
    return out


def _score_all(model_dir, pairs, maxlen=1024):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    model.eval()
    out = []
    with torch.no_grad():
        for prompt, cands in pairs:
            enc = tok([[prompt, c] for c in cands], padding=True, truncation="only_first",
                      max_length=maxlen, return_tensors="pt").to("cuda")
            s = model(**enc).logits.squeeze(-1).float().tolist()
            out.append(s if isinstance(s, list) else [s])
    del model
    torch.cuda.empty_cache()
    return out


def stage23(model_dir):
    print()
    print("=" * 72)
    print("STAGE 2/3 -- identical prompts through two FRESH model loads")
    print("=" * 72)
    pairs = _load_pairs(N_DECISIONS)
    print("decisions replayed: %d (prompts taken verbatim from %s)" % (len(pairs), ROLLOUT))
    a = _score_all(model_dir, pairs)
    b = _score_all(model_dir, pairs)
    exact = sum(1 for x, y in zip(a, b) if x == y)
    maxdiff = 0.0
    flips = 0
    ties = 0
    for x, y in zip(a, b):
        for u, v in zip(x, y):
            maxdiff = max(maxdiff, abs(u - v))
        if max(range(len(x)), key=lambda i: x[i]) != max(range(len(y)), key=lambda i: y[i]):
            flips += 1
        top = sorted(x, reverse=True)
        if len(top) > 1 and abs(top[0] - top[1]) < 1e-6:
            ties += 1
    print("  logits bitwise identical : %d / %d decisions" % (exact, len(pairs)))
    print("  largest logit difference : %.3e" % maxdiff)
    print("  ARGMAX FLIPS             : %d / %d  (%.1f%%)"
          % (flips, len(pairs), 100.0 * flips / max(1, len(pairs))))
    print("  exact top-2 ties         : %d" % ties)
    print()
    if flips == 0 and exact == len(pairs):
        print("  VERDICT: the scoring path is deterministic. The run-level shift is NOT here;")
        print("           look at game-side state (engine RNG interaction, cell scheduling).")
    elif flips == 0:
        print("  VERDICT: logits wobble but the argmax never flips -- cannot explain the gate.")
    else:
        print("  VERDICT: %.1f%% of decisions flip between two loads of the SAME weights."
              % (100.0 * flips / max(1, len(pairs))))
        print("           At ~70 decisions/game that is a different policy every run, which")
        print("           is exactly a run-level error and will NOT shrink with more games.")


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/root/out/rlDL/A_r6_policy"
    ok = stage1_prompt()
    if not ok:
        print()
        print("Stage 1 already explains a run-level shift; stages 2/3 run anyway for the record.")
    stage23(ckpt)


if __name__ == "__main__":
    main()
