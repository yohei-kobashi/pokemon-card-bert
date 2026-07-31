"""Learned value function: predict P(me wins) from a board state.

Why: the hand-written evaluator (tools/eval_state.py) discriminates winner-from-loser
at 51.4 / 65.1 / 76.8 / 90.0 % at 25/50/75/100% of game progress. The early number is
NOT a tuning failure -- 92% of states at the 25% mark are prize-TIED, so the prize term
(which carries the score) is silent and only the clamped tie-breakers speak. Peeking K
steps into the realised future recovers it (25% mark: 51.5 -> 69.6% at +40 steps ~= 6.8
turns), which says the information is THERE but not in the static features.

Search at inference is not affordable (10 min/game), so the play is: lookahead/outcome
as the TRAINING TARGET, a learned function as the fast approximation.

Design notes:
  * Label = did this player win. Directly optimises the metric we report.
  * Every state is emitted from BOTH perspectives with the label flipped, which forces
    the model to be antisymmetric instead of learning "player 0 wins more".
  * The handcrafted evaluate() score is itself a FEATURE, so the model starts from the
    heuristic and can only add to it.
  * Split is by GAME, never by state -- states within a game are near-duplicates and a
    state-level split leaks the outcome.
  * Excluded on purpose (user call): deck count, discard contents, retreat cost -- their
    value swings too much per deck to generalise across a 60-deck fleet.

Usage:
    python tools/value_net.py --tar <raw.tar> --matchups 400 --games 20 --out out/value
"""
import argparse, gzip, json, os, random, sys, tarfile, time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                    # noqa: E402
from cg.api import to_observation_class               # noqa: E402
from agents.engine_v2 import SideView, _prize_value, _CARDS   # noqa: E402
from eval_state import evaluate, _side, _weak_mult, WIN       # noqa: E402

_TAR = None          # per-worker handle (tarfile objects do not survive fork cleanly)


def _sv_feats(sv, agg, ps):
    """Per-side features. All universal game state -- nothing deck-specific."""
    a = sv.active
    act_pv = _prize_value(a.pk) if a else 0
    return [
        agg["prizes_left"], agg["n_pk"], agg["energy"], agg["hand"],
        agg["board_hp"] / 100.0, agg["dmg_frac"], agg["loaded"], agg["stage"],
        agg["prize_on_board"],
        len(sv.bench), 1.0 if a else 0.0,
        (a.hp / a.max_hp) if (a and a.max_hp) else 0.0,
        (a.hp / 100.0) if a else 0.0,
        act_pv,
        (a.best_ready_dmg / 100.0) if a else 0.0,
        (a.best_potential_dmg / 100.0) if a else 0.0,
        1.0 if getattr(ps, "asleep", False) else 0.0,
        1.0 if getattr(ps, "paralyzed", False) else 0.0,
        1.0 if getattr(ps, "confused", False) else 0.0,
        1.0 if getattr(ps, "poisoned", False) else 0.0,
        1.0 if getattr(ps, "burned", False) else 0.0,
    ]


_SIDE_NAMES = ["prizes_left", "n_pk", "energy", "hand", "board_hp", "dmg_frac",
               "loaded", "stage", "prize_on_board", "n_bench", "has_active",
               "act_hp_frac", "act_hp", "act_prize_value", "act_ready_dmg",
               "act_potential_dmg", "asleep", "paralyzed", "confused",
               "poisoned", "burned"]


def feature_names():
    return (["turn", "is_first", "to_move", "eval_raw", "eval_prize_lead"]
            + [f"me_{n}" for n in _SIDE_NAMES]
            + [f"op_{n}" for n in _SIDE_NAMES]
            + [f"d_{n}" for n in _SIDE_NAMES])


def featurize(st, me):
    """Feature vector for `st` from player `me`'s view, or None if unusable."""
    try:
        me_sv = SideView(st.players[me], {}, True)
        op_sv = SideView(st.players[1 - me], {}, False)
        me_a, op_a = _side(me_sv), _side(op_sv)
        mf = _sv_feats(me_sv, me_a, st.players[me])
        of = _sv_feats(op_sv, op_a, st.players[1 - me])
        ev = evaluate(st, me)
        if abs(ev) >= WIN:
            return None                     # terminal: the override already answers it
        row = [float(getattr(st, "turn", 0) or 0),
               1.0 if getattr(st, "firstPlayer", -1) == me else 0.0,
               1.0 if getattr(st, "yourIndex", -1) == me else 0.0,
               ev / 1000.0,
               float(op_a["prizes_left"] - me_a["prizes_left"])]
        row += mf + of + [a - b for a, b in zip(mf, of)]
        return row
    except Exception:
        return None


def _open_member(tar_path, name):
    """Kaggle may hand us the raw tar OR an already-extracted tree; support both."""
    if tar_path is None:
        return gzip.open(name, "rt")
    global _TAR
    if _TAR is None:
        _TAR = tarfile.open(tar_path)
    return gzip.open(_TAR.extractfile(name), "rt")


def _list_members(data_path):
    """-> (tar_path_or_None, [member names or file paths])"""
    if os.path.isdir(data_path):
        out = []
        for root, _dirs, files in os.walk(data_path):
            out += [os.path.join(root, f) for f in files if f.endswith(".jsonl.gz")]
        return None, sorted(out)
    tf = tarfile.open(data_path)
    names = [m.name for m in tf.getmembers() if m.name.endswith(".jsonl.gz")]
    tf.close()
    return data_path, sorted(names)


def _extract(args):
    """One tar member -> (X, y, progress, game_key) for a sample of its games."""
    name, tar_path, n_games, per_game, seed = args
    rng = random.Random(seed)
    X, y, prog, gid_out = [], [], [], []
    header, games = None, {}
    try:
        for line in _open_member(tar_path, name):
            rec = json.loads(line)
            if rec.get("kind") == "game":
                header = rec
                continue
            if header is None or rec.get("kind") != "step":
                continue
            g = rec["game_id"]
            if g not in games and len(games) >= n_games:
                continue
            games.setdefault(g, {"w": header["winner"], "s": []})["s"].append(rec["obs"])
    except Exception:
        return [], [], [], []
    for g, d in games.items():
        steps, w = d["s"], d["w"]
        N = len(steps)
        if N < 8 or w not in (0, 1):
            continue
        idxs = sorted(rng.sample(range(N), min(per_game, N)))
        for i in idxs:
            try:
                st = to_observation_class(steps[i]).current
            except Exception:
                continue
            if not st or len(st.players or []) != 2:
                continue
            # BOTH perspectives or NEITHER: the discrimination metric pairs the winner
            # row with the loser row by position, so dropping one side of a state would
            # silently desynchronise every pair after it.
            f0, f1 = featurize(st, 0), featurize(st, 1)
            if f0 is None or f1 is None:
                continue
            for me, f in ((0, f0), (1, f1)):
                X.append(f); y.append(1 if me == w else 0)
                prog.append(i / max(N - 1, 1)); gid_out.append(f"{name}:{g}")
    return X, y, prog, gid_out


def build(data_path, matchups, n_games, per_game, workers, seed=0, deadline=0):
    tar_path, names = _list_members(data_path)
    random.Random(1).shuffle(names)
    names = names[:matchups]
    jobs = [(n, tar_path, n_games, per_game, seed + i) for i, n in enumerate(names)]
    X, y, prog, gid = [], [], [], []
    t0 = time.time()
    with Pool(workers) as pool:
        it = pool.imap_unordered(_extract, jobs)
        for k, (a, b, c, d) in enumerate(it, 1):
            X += a; y += b; prog += c; gid += d
            if k % 25 == 0:
                print(f"  [{k}/{len(jobs)}] rows={len(X)} {time.time()-t0:.0f}s", flush=True)
            if deadline and time.time() - t0 > deadline:
                # Stop EXTRACTING, not the run -- the kernel still has to train and
                # report, and a 12h wall-clock kill loses everything.
                print(f"  extract deadline hit at {k}/{len(jobs)} members", flush=True)
                pool.terminate()
                break
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8),
            np.asarray(prog, dtype=np.float32), np.asarray(gid))


def _bucket_report(score_w, score_l, prog, label):
    """Winner-vs-loser discrimination in the SAME buckets eval_state.py reports."""
    print(f"\n{label}: winner-score > loser-score, by game progress")
    out = {}
    for lo, hi, tag in ((0.0, 0.15, "0-15%"), (0.15, 0.35, "25%"),
                        (0.35, 0.65, "50%"), (0.65, 0.85, "75%"), (0.85, 1.01, "100%")):
        m = (prog >= lo) & (prog < hi)
        n = int(m.sum())
        if not n:
            continue
        acc = float((score_w[m] > score_l[m]).mean()) * 100
        out[tag] = acc
        print(f"   {tag:>6} : {acc:5.1f}%   (n={n})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", default="data/kaggle_out/v24_full/selfplay_v24_full_raw.tar",
                    help="raw .tar OR a directory tree of *.jsonl.gz")
    ap.add_argument("--extract-deadline", type=float, default=0,
                    help="seconds; stop extracting and go train (0 = no limit)")
    ap.add_argument("--matchups", type=int, default=400)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--per-game", type=int, default=14)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "value"))
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--from-npz", default="", help="skip extraction; refit from value_data.npz")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.from_npz:
        d = np.load(args.from_npz, allow_pickle=False)
        X, y, prog, gid = d["X"], d["y"], d["prog"], d["gidx"].astype(str)
        print(f"refit from {args.from_npz}: rows {len(X)} features {X.shape[1]}")
    else:
        X, y, prog, gid = None, None, None, None
    if X is None:
      print(f"extracting: {args.matchups} matchups x {args.games} games x "
          f"{args.per_game} states, {args.workers} workers", flush=True)
      X, y, prog, gid = build(args.tar, args.matchups, args.games, args.per_game,
                              args.workers, deadline=args.extract_deadline)
    print(f"rows {len(X)}  features {X.shape[1]}  win-rate {y.mean():.3f}")
    if len(X) < 5000:
        print("too few rows; aborting"); return

    # split by GAME -- a state-level split leaks the outcome through near-duplicates
    ug = np.unique(gid)
    rng = np.random.default_rng(0); rng.shuffle(ug)
    test_g = set(ug[:max(1, int(len(ug) * args.holdout))].tolist())
    te = np.array([g in test_g for g in gid])
    tr = ~te
    print(f"games {len(ug)}  train rows {int(tr.sum())}  test rows {int(te.sum())}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=0)
    t0 = time.time()
    clf.fit(X[tr], y[tr])
    print(f"fit {time.time()-t0:.0f}s  iters {clf.n_iter_}")
    p = clf.predict_proba(X[te])[:, 1]
    print(f"holdout AUC {roc_auc_score(y[te], p):.4f}  "
          f"acc {(p.round() == y[te]).mean():.4f}")

    # Discrimination: each state contributes one winner-row and one loser-row (the two
    # perspectives are adjacent in emission order). Rebuild the pairs from the labels.
    Xt, yt, pt = X[te], y[te], prog[te]
    ev = Xt[:, 3]                                     # handcrafted eval, same rows
    sc = clf.predict_proba(Xt)[:, 1]
    # Rows are emitted strictly as (state, me=0), (state, me=1) and the split is by game,
    # so adjacency survives masking: reshape to pairs and read off which side won.
    assert len(yt) % 2 == 0, "rows must come in perspective pairs"
    idx = np.arange(len(yt)).reshape(-1, 2)
    lab = yt[idx]
    assert (lab.sum(axis=1) == 1).all(), "each pair must hold exactly one winner"
    wi = np.where(lab[:, 0] == 1, idx[:, 0], idx[:, 1])
    li = np.where(lab[:, 0] == 1, idx[:, 1], idx[:, 0])
    base = _bucket_report(ev[wi], ev[li], pt[wi], "HANDCRAFTED eval_state")
    learn = _bucket_report(sc[wi], sc[li], pt[wi], "LEARNED value function")
    print("\ndelta (learned - handcrafted):")
    for k in learn:
        if k in base:
            print(f"   {k:>6} : {learn[k]-base[k]:+5.1f} pt")

    # The FEATURE MATRIX is the portable artifact, not the pickle: a sklearn/numpy
    # pickle written on the Kaggle image (numpy 2.x) will not load under numpy 1.26,
    # and fitting takes under a minute -- so ship the data and refit in place.
    _, gidx = np.unique(gid, return_inverse=True)
    np.savez_compressed(os.path.join(args.out, "value_data.npz"),
                        X=X, y=y, prog=prog, gidx=gidx.astype(np.int32),
                        features=np.array(feature_names()))
    import pickle
    with open(os.path.join(args.out, "value_model.pkl"), "wb") as f:
        pickle.dump({"model": clf, "features": feature_names()}, f)
    json.dump({"auc": float(roc_auc_score(y[te], p)), "rows": int(len(X)),
               "handcrafted": base, "learned": learn},
              open(os.path.join(args.out, "value_metrics.json"), "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
