import gzip, json, math, random, statistics as st
rows = [json.loads(x) for x in gzip.open("/root/rl/plan_r2.jsonl.gz", "rt")]
random.Random(0).shuffle(rows)


def H(wc):
    s = sum(wc)
    p = [x / s for x in wc if x > 0]
    return -sum(q * math.log(q) for q in p)


print("FLOOR (target entropy): probe300 %.4f | all %.4f | rows %d"
      % (st.mean([H(r["wc"]) for r in rows[:300]]), st.mean([H(r["wc"]) for r in rows]), len(rows)))
