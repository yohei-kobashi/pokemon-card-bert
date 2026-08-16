"""Rough CPU inference (PLAYING) cost for the SFT LM agent. Reference only:
   - PyTorch float32 on this SERVER CPU (NOT Kaggle HW; Kaggle ~4 vCPU).
   - PRE-quantization: ship model is GGUF k-quant (llama.cpp) => faster than this.
   Self-contained scoring (gather-before-lm_head, same math as ScoringModel/_batched_score,
   but correct `core` for a MERGED non-PEFT model). BENCH_THREADS sets torch threads."""
import os, sys, time, statistics
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = os.path.expanduser("~/ptcg/repo")
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    sys.path.insert(0, p)
import json
import torch
THREADS = int(os.environ.get("BENCH_THREADS", "4"))
torch.set_num_threads(THREADS)
from transformers import AutoTokenizer, AutoModelForCausalLM
import library, arena
from lm.agent import make_lm_agent
from battle_log import load_agent

BASE = os.path.join(ROOT, "out", "rl", "sft_merged")
print(f"loading {BASE} on CPU float32 (threads={THREADS}) ...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()
print(f"loaded in {time.time()-t0:.1f}s", flush=True)
MAXLEN = 1024
PAD = tok.pad_token_id


def raw_score(prompt, cands, obs=None):
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = [tok(c, add_special_tokens=False)["input_ids"] or [tok.eos_token_id] for c in cands]
    cap = MAXLEN - max(len(c) for c in c_ids)
    if cap > 0 and len(p_ids) > cap:
        p_ids = p_ids[-cap:]
    seqs = [p_ids + c for c in c_ids]
    L = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), L), PAD, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        inp[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    start = len(p_ids)
    bpos, ppos = [], []
    for i, c in enumerate(c_ids):
        for t in range(start, start + len(c)):
            bpos.append(i); ppos.append(t - 1)
    with torch.no_grad():
        h = model.model(input_ids=inp, attention_mask=att).last_hidden_state   # merged: .model = backbone
        bt = torch.tensor(bpos); pt = torch.tensor(ppos)
        lp = torch.log_softmax(model.lm_head(h[bt, pt]).float(), -1)
        tgt = inp[bt, pt + 1]
        tok_lp = lp[torch.arange(len(bpos)), tgt]
    out, k = [], 0
    for c in c_ids:
        out.append(float(tok_lp[k:k + len(c)].sum()) / max(1, len(c))); k += len(c)
    return out


timings, ncands = [], []
stats = {"calls": 0, "ok": 0, "err": 0, "first_err": None}
TARGET = 80


class Done(BaseException):
    pass


def timed(prompt, cands, obs=None):
    stats["calls"] += 1
    try:
        t = time.perf_counter()
        r = raw_score(prompt, cands, obs)
        dt = time.perf_counter() - t
    except Exception as e:
        stats["err"] += 1
        if stats["first_err"] is None:
            stats["first_err"] = repr(e)
        raise
    stats["ok"] += 1
    timings.append(dt); ncands.append(len(cands))
    if len(timings) >= TARGET:
        raise Done
    return r


class Shim:
    pass


sm = Shim(); sm.score = timed

tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
PILOT, OPP = "dragapult", "alakazam"
dl = library.read_deck(PILOT); ol = library.read_deck(OPP)
lm_agent = make_lm_agent(dl, profile=tun.get(PILOT, {}), model=sm)
oa = load_agent(OPP)

print(f"playing {PILOT} (LM/CPU) vs {OPP} (heuristic), collecting {TARGET} scored decisions ...", flush=True)
tg = time.time()
g = 0
try:
    while len(timings) < TARGET and g < 20:
        g += 1
        if g % 2 == 1:
            arena.play(lm_agent, oa, dl, ol)
        else:
            arena.play(oa, lm_agent, ol, dl)
except Done:
    pass
wall = time.time() - tg

print(f"\nscore() calls={stats['calls']} ok={stats['ok']} err={stats['err']} "
      f"first_err={stats['first_err']}  games_played={g}")
if not timings:
    print("no successful score() timings"); sys.exit(0)
warm = min(3, len(timings) - 1)
ms = [x * 1000 for x in timings[warm:]]
c = ncands[warm:]
print("\n==== CPU PLAYING COST (PyTorch f32, pre-quant, NOT Kaggle HW) ====")
print(f"threads               : {THREADS}")
print(f"scored decisions      : {len(ms)} over {g} game(s) (dropped {warm} warmup)")
print(f"candidates/decision   : mean {statistics.mean(c):.1f}  min {min(c)}  max {max(c)}")
print(f"ms / decision (score) : mean {statistics.mean(ms):.0f}  median {statistics.median(ms):.0f}"
      f"  p90 {sorted(ms)[max(0,int(0.9*len(ms))-1)]:.0f}  max {max(ms):.0f}")
print(f"ms / candidate        : mean {statistics.mean([m/n for m, n in zip(ms, c)]):.0f}")
print(f"scored decisions/game : ~{stats['ok']/max(1,g):.0f}")
print(f"EST LM score time/game: ~{statistics.mean(timings[warm:]) * (stats['ok']/max(1,g)):.1f}s")
