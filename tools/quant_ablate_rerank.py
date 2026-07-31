"""Ablate INT8 dynamic-quantization settings on an already-exported fp32 reranker ONNX.

CONTEXT: the fp32 ONNX export is numerically EXACT (argmax 100%, |diff| max 1e-4 vs PyTorch),
so the accuracy collapse (argmax 40%, mean |logit diff| 9.5) comes purely from quantization.
Prime suspect is the word-embedding `Gather` (53339 x 768 = 41.0M params): ONNX Runtime's
per-channel dynamic quantization of Gather data is fragile, whereas MatMul quantization is the
well-trodden BERT path. `reduce_range` is the other classic x86 accuracy fix (avoids INT32
accumulator saturation on CPUs without VNNI).

Each variant is scored against the SAME PyTorch reference on REAL decisions, and its .tar.gz
contribution is measured (the 197.65625 MiB cap applies to the compressed tarball).

Usage:
  python tools/quant_ablate_rerank.py --fp32 /root/onnx/rerank_diag/model.onnx \
     --model /root/out/rerank_gte_mp --data <rerank.jsonl.gz> --n 40 --work /root/onnx/ablate
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

TAR_CAP = 207257600

# (label, op_types, per_channel, reduce_range, exclude_regex)
# Round 1 established: Gather quantization must be OFF, per_channel ON, reduce_range ON
# (argmax 40.0 -> 67.5%). Round 2 layers selective fp32 exclusion on top of that winner --
# weights have no outliers (max |w| 4.4), so the residual damage is activation scaling and
# 22-layer error accumulation, which exclusion targets directly.
VARIANTS = [
    ("matmul+gather_perch", ["MatMul", "Gather"], True, False, None),   # round-1, broken
    ("matmul+gather_pertensor", ["MatMul", "Gather"], False, False, None),
    ("matmul_only_perch", ["MatMul"], True, False, None),
    ("matmul_only_pertensor", ["MatMul"], False, False, None),
    ("matmul_only_perch_rr", ["MatMul"], True, True, None),             # round-1 winner
    # ---- round 2: exclusion on top of the round-1 winner --------------------------------
    ("rr_ex_head", ["MatMul"], True, True, r"(classifier|/head/|pooler|/dense/)"),
    ("rr_ex_last2", ["MatMul"], True, True, r"layers\.(20|21)\D"),
    ("rr_ex_first2", ["MatMul"], True, True, r"layers\.[01]\D"),
    ("rr_ex_first2_last2_head", ["MatMul"], True, True,
     r"(layers\.[01]\D|layers\.(20|21)\D|classifier|/head/|pooler|/dense/)"),
    ("rr_ex_Wo", ["MatMul"], True, True, r"/Wo/"),   # largest weight max (4.44)
    # size-friendly: keep Gather quantized but with the round-1 winner's other settings
    ("matmul+gather_perch_rr", ["MatMul", "Gather"], True, True, None),
]


def _mb(n):
    return f"{n / 1024 / 1024:.1f} MiB"


def _gz(path):
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=True) as tf:
        subprocess.run(["tar", "czf", tf.name, "-C", os.path.dirname(path),
                        os.path.basename(path)], check=True)
        return os.path.getsize(tf.name)


def _records(data, n, stride=977):
    recs = []
    with gzip.open(data, "rt") as fh:
        for i, line in enumerate(fh):
            if i % stride == 0:
                recs.append(json.loads(line))
                if len(recs) >= n:
                    break
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", required=True, help="exported fp32 model.onnx")
    ap.add_argument("--model", required=True, help="trained reranker dir (tokenizer + weights)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="", help="comma-filter variant labels")
    args = ap.parse_args()

    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from onnxruntime.quantization import quantize_dynamic, QuantType

    os.makedirs(args.work, exist_ok=True)
    recs = _records(args.data, args.n)
    tok = AutoTokenizer.from_pretrained(args.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager")
    if getattr(ref_model.config, "reference_compile", None):
        ref_model.config.reference_compile = False
    ref_model = ref_model.eval().to(dev)

    # PyTorch reference computed ONCE
    batches, refs = [], []
    for r in recs:
        enc = tok([[r["state"], c] for c in r["candidates"]], padding=True,
                  truncation="only_first", max_length=args.max_len, return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        att = enc["attention_mask"].astype(np.int64)
        with torch.no_grad():
            out = ref_model(input_ids=torch.tensor(ids).to(dev),
                            attention_mask=torch.tensor(att).to(dev)
                            ).logits.squeeze(-1).float().cpu().numpy()
        batches.append((ids, att))
        refs.append(out)
    print(f"reference ready: {len(recs)} decisions, "
          f"{sum(b[0].shape[0] for b in batches)} pairs\n", flush=True)

    keep = set(x for x in args.only.split(",") if x)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4

    # node names for exclusion regexes (ONNX node names look like
    # "/model/layers.21/mlp/Wo/MatMul"), loaded lazily and only once
    node_names = None
    rows = []
    for label, ops, perch, rr, exre in VARIANTS:
        if keep and label not in keep:
            continue
        excl = []
        if exre:
            import re
            import onnx
            if node_names is None:
                g = onnx.load(args.fp32, load_external_data=False).graph
                node_names = [(n.name, n.op_type) for n in g.node]
            pat = re.compile(exre)
            excl = [n for n, op in node_names if op in ops and pat.search(n)]
            print(f"{label}: excluding {len(excl)} nodes matching {exre}", flush=True)
            if not excl:
                print(f"  WARNING: regex matched NOTHING -- variant is a duplicate of the "
                      f"unexcluded baseline, check node naming", flush=True)
        out_path = os.path.join(args.work, f"{label}.onnx")
        if not os.path.exists(out_path):
            quantize_dynamic(args.fp32, out_path, weight_type=QuantType.QInt8,
                             op_types_to_quantize=ops, per_channel=perch,
                             reduce_range=rr, nodes_to_exclude=excl,
                             extra_options={"MatMulConstBOnly": True})
        sess = ort.InferenceSession(out_path, so, providers=["CPUExecutionProvider"])
        top = 0
        dsum = dmax = 0.0
        for (ids, att), ref in zip(batches, refs):
            got = sess.run(["logits"], {"input_ids": ids, "attention_mask": att})[0].reshape(-1)
            d = float(np.abs(ref - got).max())
            dsum += d
            dmax = max(dmax, d)
            top += int(int(np.argmax(ref)) == int(np.argmax(got)))
        raw = os.path.getsize(out_path)
        gz = _gz(out_path)
        row = dict(label=label, ops="+".join(ops), per_channel=perch, reduce_range=rr,
                   exclude_regex=exre, n_excluded=len(excl),
                   argmax=top / len(refs), mean_diff=dsum / len(refs), max_diff=dmax,
                   raw_bytes=raw, gz_bytes=gz)
        rows.append(row)
        print(f"{label:26s} argmax {row['argmax']:6.1%}  |diff| mean {row['mean_diff']:8.4f} "
              f"max {row['max_diff']:8.4f}  raw {_mb(raw):>10s}  tar.gz {_mb(gz):>10s}",
              flush=True)
        del sess

    json.dump(rows, open(os.path.join(args.work, "ablation.json"), "w"), indent=1)
    print("\nBudget note: cap applies to the tar.gz; onnxruntime(stripped) ~17.5 MiB + "
          "tokenizer ~1.5 MiB + repo ~1 MiB must also fit under "
          f"{_mb(TAR_CAP)}.", flush=True)


if __name__ == "__main__":
    main()
