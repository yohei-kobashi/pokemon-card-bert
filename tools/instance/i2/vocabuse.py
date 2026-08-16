"""How much of the 160,922-token vocabulary does this task actually touch?

The embedding is 160,922 x 2560 = 412M parameters -- larger than a tenth of the model -- and it
is trainable, so every step computes a dense gradient over all of it and the optimizer walks all
of it. The output side is worse: the lm_head projects to all 160,922 columns for every token,
though the model may only ever legally emit ~6,282 of them.

Our prompts are machine-generated from a fixed card pool, so the token set is CLOSED -- the same
property that let the reranker's vocabulary go 53,339 -> 3,254 (`rerank-deploy-quantization-and-
speed`). This counts what is actually reachable, which bounds how far the same trick goes here.
"""
import gzip, json, sys, collections
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib"); sys.path.insert(0, "tools/instance")
from transformers import AutoTokenizer
from lm.vocab import special_tokens

tk = AutoTokenizer.from_pretrained("unsloth/Qwen3-4B-Base")
n_base = len(tk)
act = json.load(open("data/action_vocab_v39.json"))["tokens"]
tk.add_tokens(list(special_tokens()) + act)

used = collections.Counter()
n = 0
with gzip.open("data/sft/v39_dag005.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        if not d.get("target"):
            continue
        n += 1
        for i in tk(d["prompt"], add_special_tokens=False)["input_ids"]:
            used[i] += 1
        if n >= 60000:
            break

n_added = len(tk) - n_base
base_used = sum(1 for i in used if i < n_base)
print("prompts scanned        %d" % n)
print("full vocabulary        %d  (base %d + added %d)" % (len(tk), n_base, n_added))
print("base tokens ever used  %d  (%.2f%% of the base vocabulary)"
      % (base_used, 100.0 * base_used / n_base))
for thr in (5, 100):
    print("   used >= %-4d          %d" % (thr, sum(1 for i, c in used.items() if i < n_base and c >= thr)))
keep = base_used + n_added + 64      # + specials/headroom
print("\nprunable to             %d tokens (%.1fx smaller)" % (keep, len(tk) / keep))
print("  embedding params      %.0fM -> %.0fM" % (len(tk) * 2560 / 1e6, keep * 2560 / 1e6))
print("  lm_head work per token %.2fx" % (keep / len(tk)))
print("\nNOTE: the prompts are generated from a fixed card pool, so this set is closed; anything")
print("outside it can never appear. That is the same argument that made the reranker's prune safe.")
