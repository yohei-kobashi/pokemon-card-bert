"""Batched multi-adapter rollout (the countermeasure to the Stage-C 2-adapter batching
problem). cg keeps ONE global battle per process (see tools/arena.py), so we CANNOT run
N games concurrently in one process. Architecture instead:

    W CPU worker processes  (each plays ONE game at a time via cg, no torch/CUDA)
        └── request (adapter, prompt, candidates) ──►  1 GPU SERVER (main process)
        ◄── per-candidate scores ───────────────────  batches requests, GROUPS BY ADAPTER,
                                                       one set_adapter + one padded forward
                                                       per group, dispatches results back.

So at most 2 adapters ("pilot" learning + "opp" frozen) => 2 batched forwards per wave;
set_adapter just re-points the LoRA matrices on the shared base (no reload). Only PILOT
decisions are logged (the policy gradient trains the pilot only).

spawn (not fork): the server owns the CUDA context; workers must start fresh without it.
"""
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

_ACT = "[ACT]\n"


# ---------------------------------------------------------------------------- #
#  GPU server: batched, adapter-grouped candidate scoring                       #
# ---------------------------------------------------------------------------- #
def _batched_score(model, tok, device, maxlen, items):
    """items = [(prompt, [cand,...]), ...] (all for ONE adapter). Returns [[score/cand]].
    Flattens every (prompt+cand) sequence, one padded forward, slices length-normalized
    candidate logprobs, regroups per decision."""
    import torch
    seqs, meta = [], []                       # meta = (decision_idx, prompt_len, cand_len)
    for di, (prompt, cands) in enumerate(items):
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = [tok(c, add_special_tokens=False)["input_ids"] or [tok.eos_token_id] for c in cands]
        cap = maxlen - max(len(c) for c in c_ids)
        if cap > 0 and len(p_ids) > cap:
            p_ids = p_ids[-cap:]
        for c in c_ids:
            seqs.append(p_ids + c); meta.append((di, len(p_ids), len(c)))
    # EFFICIENCY (RL-cycle critical): a plain model(...).logits runs lm_head+softmax over
    # EVERY position x the 260k domain vocab (B x L x 260k) when only the ~clen candidate
    # positions per sequence matter. Run the backbone once, gather ONLY the scoring
    # positions, and apply lm_head to those -- same trick as training's _batch_loss. Cuts
    # the head/softmax ~L/clen-fold (~80x here) and kills the giant logits tensor.
    #
    # MICRO-BATCH the forward (OOM fix 2026-07-23): a full wave = up to batch_cap decisions x
    # ~dozens of candidates each => ~1000 (prompt+cand) sequences x L tokens in ONE forward,
    # which OOMs at high worker counts (a 256-core box => 64 workers => 7.55 GiB single alloc).
    # Chunk into <= MICRO sequences per forward so GPU memory is bounded by MICRO, NOT by the
    # worker/wave size. Waves still group by adapter (one set_adapter/group); only the forward
    # is chunked. Tune via RL_MICRO_BATCH (default 64: ~fits Npos x 260k lm_head on 24 GB).
    pad = tok.pad_token_id
    core = model.base_model.model if hasattr(model, "base_model") else model
    micro = max(1, int(os.environ.get("RL_MICRO_BATCH", "64")))
    seq_score = [0.0] * len(seqs)
    for c0 in range(0, len(seqs), micro):
        chunk = seqs[c0:c0 + micro]; cmeta = meta[c0:c0 + micro]
        L = max(len(s) for s in chunk)
        inp = torch.full((len(chunk), L), pad, dtype=torch.long, device=device)
        att = torch.zeros((len(chunk), L), dtype=torch.long, device=device)
        for i, s in enumerate(chunk):
            inp[i, :len(s)] = torch.tensor(s, device=device); att[i, :len(s)] = 1
        bpos, ppos = [], []
        for i, (_di, plen, clen) in enumerate(cmeta):
            for t in range(clen):
                bpos.append(i); ppos.append(plen + t - 1)   # position that predicts cand token t
        with torch.no_grad():
            h = core.model(input_ids=inp, attention_mask=att).last_hidden_state   # b x L x H
            bt = torch.tensor(bpos, device=device); pt = torch.tensor(ppos, device=device)
            lp = torch.log_softmax(core.lm_head(h[bt, pt]).float(), -1)            # Npos x vocab
            tgt = inp[bt, pt + 1]                                                  # predicted cand tokens
            tok_lp = lp[torch.arange(len(bpos), device=device), tgt]              # Npos
        k = 0
        for i, (_di, _plen, clen) in enumerate(cmeta):
            seq_score[c0 + i] = float(tok_lp[k:k + clen].sum()) / max(1, clen); k += clen
    out = [[] for _ in items]
    for i, (di, _plen, _clen) in enumerate(meta):
        out[di].append(seq_score[i])
    return out


def server_loop(base, adapters, maxlen, req_q, res_qs, n_workers, batch_cap, wait_ms):
    """Runs in the MAIN (GPU) process. Loads base + named adapters; drains requests,
    groups by adapter, batched-forwards, dispatches. Stops after every worker sends DONE."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
    names = list(adapters)
    model = PeftModel.from_pretrained(model, adapters[names[0]], adapter_name=names[0])
    for nm in names[1:]:
        model.load_adapter(adapters[nm], adapter_name=nm)
    model.eval()
    device = next(model.parameters()).device

    done = 0
    while done < n_workers:
        pending = {nm: [] for nm in names}          # adapter -> [(wid, rid, prompt, cands)]
        t0 = time.time()
        # collect a wave: up to batch_cap requests or wait_ms, whichever first
        while sum(len(v) for v in pending.values()) < batch_cap and (time.time() - t0) * 1000 < wait_ms:
            try:
                msg = req_q.get(timeout=max(0.001, wait_ms / 1000 - (time.time() - t0)))
            except Exception:
                break
            if msg[0] == "DONE":
                done += 1
                if done >= n_workers and not any(pending.values()):
                    return
                continue
            wid, rid, adapter, prompt, cands = msg
            pending[adapter].append((wid, rid, prompt, cands))
        for adapter, batch in pending.items():
            if not batch:
                continue
            model.set_adapter(adapter)
            scores = _batched_score(model, tok, device, maxlen, [(p, c) for _, _, p, c in batch])
            for (wid, rid, _, _), sc in zip(batch, scores):
                res_qs[wid].put((rid, sc))


# ---------------------------------------------------------------------------- #
#  CPU worker: plays games, requests scores over the queue                      #
# ---------------------------------------------------------------------------- #
def _rpc(req_q, res_q, wid, rid, adapter, prompt, cands):
    req_q.put((wid, rid[0], adapter, prompt, cands))
    rid[0] += 1
    got = res_q.get()                      # (rid, scores) — one worker => one in-flight
    return got[1]


def _softmax_sample(scores, temp, rng):
    if temp <= 1e-6:
        i = max(range(len(scores)), key=lambda k: scores[k]); return i, 0.0
    m = max(scores); ex = [math.exp((s - m) / temp) for s in scores]; Z = sum(ex)
    p = [e / Z for e in ex]; r = rng.random(); acc = 0.0
    for i, pr in enumerate(p):
        acc += pr
        if r <= acc:
            return i, math.log(max(pr, 1e-12))
    return len(p) - 1, math.log(max(p[-1], 1e-12))


def worker_main(wid, pairs, gpm, temp, heur, seed, profiles_json, req_q, res_q, results_q):
    import library                                    # cg is single-battle per process = fine
    from lm.serialize import serialize_stateless, multipick_substate, STOP
    from lm.actions import encode_option
    from cg.game import battle_start, battle_select, battle_finish
    from lm.agent import make_lm_agent
    profiles = json.loads(profiles_json)
    rng = random.Random(seed)
    rid = [0]
    dpath = lambda n: os.path.join(ROOT, "decks", n + ".csv")
    forced_cache = {}
    cur = {"pilot": None, "opp": None}   # current game's deck ids -> v2 STABLE glossary prefix

    def forced(obs, deck):
        if deck not in forced_cache:
            forced_cache[deck] = make_lm_agent(deck, profiles.get(deck), None)
        return forced_cache[deck](obs)

    def score(adapter, prompt, cands):
        return _rpc(req_q, res_q, wid, rid, adapter, prompt, cands)

    def pilot_pick(obs, records, matchup):
        sel = obs["select"]; opts = sel.get("option") or []
        lo = sel.get("minCount", 1) or 0; hi = sel.get("maxCount", 1) or 1
        if lo == 1 and hi == 1:
            prompt = _ACT + serialize_stateless(obs, deck_ids=cur["pilot"]); cands = [encode_option(o, obs) for o in opts]
            sc = score("pilot", prompt, cands)
            j, _ = _softmax_sample(sc, temp, rng)
            records.append(dict(matchup=matchup, prompt=prompt, cands=cands, chosen=j, scores=sc))
            return [j]
        picked = []
        while len(picked) < hi:
            sub, remaining, allow_stop = multipick_substate(obs, picked)
            if not remaining:
                break
            prompt = _ACT + serialize_stateless(sub, deck_ids=cur["pilot"])
            cands = [encode_option(opts[i], obs) for i in remaining] + ([STOP] if allow_stop else [])
            sc = score("pilot", prompt, cands)
            j, _ = _softmax_sample(sc, temp, rng)
            records.append(dict(matchup=matchup, prompt=prompt, cands=cands, chosen=j, scores=sc))
            if allow_stop and j == len(cands) - 1:
                break
            picked.append(remaining[j])
        return picked if len(picked) >= lo else None

    def opp_pick(obs):                                  # opponent = argmax under "opp" adapter
        sel = obs["select"]; opts = sel.get("option") or []
        if len(opts) < 2:
            return None
        lo = sel.get("minCount", 1) or 0; hi = sel.get("maxCount", 1) or 1
        if lo == 1 and hi == 1:
            sc = score("opp", _ACT + serialize_stateless(obs, deck_ids=cur["opp"]), [encode_option(o, obs) for o in opts])
            return [max(range(len(sc)), key=lambda k: sc[k])]
        picked = []
        while len(picked) < hi:
            sub, remaining, allow_stop = multipick_substate(obs, picked)
            if not remaining:
                break
            cands = [encode_option(opts[i], obs) for i in remaining] + ([STOP] if allow_stop else [])
            sc = score("opp", _ACT + serialize_stateless(sub, deck_ids=cur["opp"]), cands)
            j = max(range(len(sc)), key=lambda k: sc[k])
            if allow_stop and j == len(cands) - 1:
                break
            picked.append(remaining[j])
        return picked if len(picked) >= lo else None

    all_records, all_rewards = [], []
    # RL_GAME_CAP bounds per-game decisions: stall/mill/control mirrors can run toward the
    # 4000-decision ceiling, and the whole rollout blocks on the slowest worker's slowest
    # game (results_q.get x W + join). A game not resolved by the cap is silently dropped
    # (recs never committed) -- fine for a broad-climb round. Default 4000 = no change.
    game_cap = int(os.environ.get("RL_GAME_CAP", "4000"))
    npairs = len(pairs)
    for _pi, (pilot, opp) in enumerate(pairs):
        print(f"[w{wid}] pair {_pi + 1}/{npairs} {pilot} vs {opp}", file=sys.stderr, flush=True)
        d_pilot = [int(x) for x in open(dpath(pilot))]
        d_opp = [int(x) for x in open(dpath(opp))]
        cur["pilot"], cur["opp"] = d_pilot, d_opp     # v2 glossary prefix for this matchup
        for g in range(gpm):
            use_heur = rng.random() < heur
            opp_agent = make_lm_agent(opp, profiles.get(opp), None) if use_heur else None
            first = (g % 2 == 0); pilot_i = 0 if first else 1
            d0, d1 = (d_pilot, d_opp) if first else (d_opp, d_pilot)
            matchup = f"{pilot}__vs__{opp}"
            recs = []
            obs, _ = battle_start(d0, d1)
            if obs is None:
                battle_finish(); continue
            try:
                for _ in range(game_cap):
                    cur = obs.get("current")
                    if cur is None:
                        break
                    if cur.get("result", -1) != -1:
                        win = 1 if cur["result"] == pilot_i else -1
                        all_rewards.append(dict(matchup=matchup, reward=win, n_decisions=len(recs),
                                                opp_kind="heuristic" if use_heur else "lm"))
                        all_records.extend(recs); recs = None
                        break
                    sel = obs.get("select")
                    if sel is None:
                        break
                    yi = cur["yourIndex"]
                    if yi == pilot_i:
                        ch = pilot_pick(obs, recs, matchup) if (sel and len(sel.get("option") or []) >= 2) else forced(obs, pilot)
                        if ch is None:
                            ch = forced(obs, pilot)
                    else:
                        ch = (opp_agent(obs) if use_heur else opp_pick(obs))
                        if ch is None:
                            ch = forced(obs, opp)
                    obs = battle_select(ch)
            finally:
                battle_finish()
    req_q.put(("DONE",))
    results_q.put((all_records, all_rewards))


# ---------------------------------------------------------------------------- #
#  orchestrator                                                                 #
# ---------------------------------------------------------------------------- #
def run_batched_rollout(base, pilot_adapter, opp_adapter, pairs, gpm, temp, heur,
                        workers, seed, batch_cap=0, wait_ms=15):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    adapters = {"pilot": pilot_adapter, "opp": opp_adapter or pilot_adapter}
    profiles_json = open(os.path.join(ROOT, "agents", "tuning.json")).read()
    W = min(workers, max(1, len(pairs)))
    batch_cap = batch_cap or W                          # <= W in-flight (one per worker)

    req_q = ctx.Queue()
    res_qs = [ctx.Queue() for _ in range(W)]
    results_q = ctx.Queue()
    # shard pairs across workers
    shards = [pairs[i::W] for i in range(W)]
    procs = []
    for wid in range(W):
        p = ctx.Process(target=worker_main, args=(
            wid, shards[wid], gpm, temp, heur, seed + wid, profiles_json,
            req_q, res_qs[wid], results_q))
        p.start(); procs.append(p)
    # main process = GPU server
    server_loop(base, adapters, 1024, req_q, res_qs, W, batch_cap, wait_ms)
    records, rewards = [], []
    for _ in range(W):
        r, w = results_q.get(); records.extend(r); rewards.extend(w)
    for p in procs:
        p.join()
    return records, rewards
