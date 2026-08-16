#!/usr/bin/env python3
"""What does an instance2-hosted opponent actually cost instance1, end to end?

    PYTHONPATH=cg-lib:tools:. python3 tools/bench_score_link.py \
        --url http://127.0.0.1:8077 --adapter lora_marnie_grimmsnarl_r1 --threads 20

Measures the WHOLE path -- tunnel round trip, one shared GPU, adapter switching -- because that
is what sets how many games a round can afford, and none of it is visible in a server-side
timing. The number to plan with is `decisions/s`, against the arithmetic printed at the end:
a gate of 3 arms x 8 opponents x 150 games is ~180k opponent decisions, so 10/s is five hours.

--threads should match the gate's worker count (field_chain runs 20). GPU work is serialised on
the server by design (batching is a measured loss on this workload), so concurrency here buys
only the hiding of round-trip latency -- which is exactly what needs measuring over a link
between two rented boxes.
"""
import argparse
import os
import statistics
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lm.remote_scorer import RemoteScorer          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8077")
    ap.add_argument("--adapter", default="", help="empty = the server's current one")
    ap.add_argument("--adapters", default="", help="comma list: round-robin, as 8 opponent "
                                                   "decks in one gate would")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--per-thread", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=800, help="prompt length; real ones are ~800")
    ap.add_argument("--cands", type=int, default=6, help="real decisions average 5.9")
    a = ap.parse_args()

    names = [x for x in a.adapters.split(",") if x] or [a.adapter]
    filler = " ".join("c%d" % (i % 900 + 1) for i in range(a.tokens))
    cands = ["opt%d" % i for i in range(a.cands)]
    lat, lock, errs = [], threading.Lock(), []

    probe = RemoteScorer(a.url, names[0] or None)
    print("server: %s" % probe.health(), flush=True)

    def work(tid):
        sc = RemoteScorer(a.url, names[tid % len(names)] or None)
        for i in range(a.per_thread):
            p = "[ACT]\nDECK win[%s] t%d.%d || SEL MAIN n1-1 :: 0=end" % (filler, tid, i)
            t0 = time.time()
            try:
                sc.score(p, cands)
            except Exception as e:                                       # noqa: BLE001
                with lock:
                    errs.append(str(e))
                continue
            with lock:
                lat.append(time.time() - t0)

    t0 = time.time()
    ths = [threading.Thread(target=work, args=(i,)) for i in range(a.threads)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0

    if not lat:
        raise SystemExit("every request failed: %s" % (errs[:3] or "?"))
    lat.sort()
    n = len(lat)
    print("\nadapters   %s" % ", ".join(names))
    print("threads    %d x %d = %d decisions (%d failed)"
          % (a.threads, a.per_thread, n, len(errs)))
    print("latency    mean %.0f ms  p50 %.0f  p90 %.0f  max %.0f"
          % (1000 * statistics.mean(lat), 1000 * lat[n // 2],
             1000 * lat[min(n - 1, int(0.9 * n))], 1000 * lat[-1]))
    rate = n / wall
    print("THROUGHPUT %.1f decisions/s  (%.0f ms/decision of GPU)" % (rate, 1000 * wall / n))
    print("\nwhat that buys, at ~50 opponent decisions per game:")
    for tag, games in (("collect 100 games", 100), ("gate 150 x 8 opp", 150 * 8),
                       ("gate 3 arms x 150 x 8", 3 * 150 * 8)):
        secs = games * 50 / rate
        print("  %-24s %6d games  %5.1f h" % (tag, games, secs / 3600.0))
    if errs:
        print("\nfailures (%d): %s" % (len(errs), errs[0]))


if __name__ == "__main__":
    main()
