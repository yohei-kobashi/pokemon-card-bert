"""Export our live (Kaggle) match history into the local logs/ replay format.

Each Kaggle episode replay already carries, at steps[0][0]["visualize"], the exact
heroz visualize array that cg.game.visualize_data() produces locally -- i.e. the
same thing battle_log.save_battle() writes into logs/. So converting a live game to
a local replay is lossless: pull that array out and write it under the local
filename convention {ts}_{mode}_{p0agent}-{p0deck}_vs_{p1agent}-{p1deck}.json so it
replays in visualizer.html exactly like a local AI-vs-AI game.

Our side's (agent, deck) is recovered from the submission fileName "<agent>-<deck>.tar.gz".
The opponent is a foreign submission, so its deck is classified from the replay's
decklist (leaderboard_distribution.classify) and labelled "<teamSlug>-<archetype>".

    PYTHONPATH=cg-lib python tools/export_live_logs.py [--team 16372630] \
        [--out logs] [--limit N] [--sub SUBID ...]

Idempotent: skips an episode whose output file already exists. Replays are cached
under scratchpad_replays/ by scout_decks._download_episode.
"""
import sys, os, re, json, argparse
sys.path.insert(0, "tools")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kaggle
from scout_decks import _download_episode, _game_decklists
from leaderboard_distribution import _load_our_decks, classify, _pokemon_ids

COMPETITION = "pokemon-tcg-ai-battle"


def _slug(s):
    return re.sub(r"[^A-Za-z0-9_]+", "-", str(s)).strip("-_") or "x"


def _sub_labels(api):
    """submissionId -> (agent, deck) parsed from our own submissions' fileNames."""
    out = {}
    for s in api.competition_submissions(COMPETITION):
        d = s.to_dict() if hasattr(s, "to_dict") else {}
        fn = d.get("fileName") or ""
        ref = d.get("ref")
        m = re.match(r"(.+?)-(.+?)\.tar\.gz$", fn)
        if ref is not None and m:
            out[int(ref)] = (m.group(1), m.group(2))
    return out


# Element keys cg.game.visualize_data() emits locally. A live replay element is a
# superset (adds action/obs/ver); projecting to these makes the file byte-schema
# identical to a locally-produced logs/ replay (inner "current" schema already matches).
_LOCAL_KEYS = ("current", "logs", "select", "selected")


def _visualize(rep, raw=False):
    """The full-game heroz visualize array (== local save_battle payload), or None.

    By default each element is projected to the local visualize_data() key set so the
    output matches locally-produced logs/ replays exactly; raw=True keeps the live
    superset (extra action/obs/ver keys).
    """
    steps = rep.get("steps") or []
    if not steps or not steps[0]:
        return None
    v = steps[0][0].get("visualize")
    if not v:
        return None
    if isinstance(v, str):
        v = json.loads(v)
    if raw:
        return v
    return [{k: el[k] for k in _LOCAL_KEYS if k in el} if isinstance(el, dict) else el
            for el in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", type=int, default=16372630)
    ap.add_argument("--out", default="logs")
    ap.add_argument("--limit", type=int, default=0, help="max episodes (0 = all)")
    ap.add_argument("--sub", type=int, nargs="*", help="only these submission ids")
    ap.add_argument("--raw", action="store_true",
                    help="keep the live visualize superset (extra action/obs/ver keys)")
    args = ap.parse_args()

    api = kaggle.api
    our = _load_our_decks()
    sub_labels = _sub_labels(api)
    os.makedirs(args.out, exist_ok=True)

    sub_ids = args.sub or sorted(sub_labels, reverse=True)
    seen_eps, written, skipped, failed = set(), 0, 0, 0

    for sid in sub_ids:
        try:
            eps = list(api.competition_list_episodes(sid))
        except Exception as e:
            print(f"  sub {sid}: list failed: {e}", file=sys.stderr)
            continue
        for e in eps:
            if e.id in seen_eps:
                continue
            seen_eps.add(e.id)
            ed = e.to_dict()
            agents = ed.get("agents") or []
            if not any(a.get("teamId") == args.team for a in agents):
                continue
            # order agents by their board index (missing index == 0)
            ordered = sorted(agents, key=lambda a: a.get("index", 0))
            ts = (ed.get("endTime") or ed.get("createTime") or "")
            ts = re.sub(r"[^0-9]", "", ts)[:14] or f"ep{e.id}"

            try:
                rep = _download_episode(api, e.id)
            except Exception as ex:
                print(f"  ep {e.id}: download failed: {ex}", file=sys.stderr)
                failed += 1
                continue
            vis = _visualize(rep, raw=args.raw)
            if vis is None:
                print(f"  ep {e.id}: no visualize array", file=sys.stderr)
                failed += 1
                continue

            names = rep.get("info", {}).get("TeamNames") or []
            dls = _game_decklists(rep)
            labels = []
            for a in ordered:
                sub = a.get("submissionId")
                tn = a.get("teamName", "?")
                if a.get("teamId") == args.team and sub in sub_labels:
                    ag, dk = sub_labels[sub]
                    labels.append(f"{_slug(ag)}-{_slug(dk)}")
                else:
                    cnt = dls.get(tn, {})
                    arch, score = classify(cnt, our) if cnt else ("unknown", 0)
                    if score < 0.5:
                        pk = _pokemon_ids(cnt)
                        arch = pk[0][2] if pk else "unknown"
                    labels.append(f"{_slug(tn)}-{_slug(arch)}")

            fname = f"{ts}_AIvAI_{labels[0]}_vs_{labels[1]}.json"
            path = os.path.join(args.out, fname)
            if os.path.exists(path):
                skipped += 1
                continue
            with open(path, "w") as f:
                f.write(vis if isinstance(vis, str) else json.dumps(vis))
            written += 1
            if args.limit and written >= args.limit:
                print(f"reached limit {args.limit}")
                break
        if args.limit and written >= args.limit:
            break

    print(f"done: {written} written, {skipped} already present, {failed} failed; "
          f"{len(seen_eps)} episodes scanned -> {args.out}/")


if __name__ == "__main__":
    main()
