"""Prune the reranker's word embedding to the tokens that actually occur, then export +
weight-only-INT8-quantize the ONNX deploy model.

WHY: the embedding is 53,339 x 768 = 41.0M params -- 37% of the model -- and weight-only
quantization leaves it alone (it is a Gather, not a MatMul), so it sits at fp32 156 MiB and
dominates the submission budget. A full sweep of all 1.52M training records + the card DB +
every single-char token found only 3,252 distinct ids (6.1%), so 94% of that is dead weight.

The TOKENIZER IS NOT TOUCHED. Dropping BPE merges would silently change how an unseen card
name tokenizes; instead the tokenizer keeps emitting original ids and a 53,339-entry int32
lookup (213 KB) maps them onto the pruned rows at inference. Any id outside the kept set --
which the sweep says cannot happen, but might if the card pool changes -- maps to [UNK]
rather than crashing on an out-of-range Gather.

Quantization is weight-only INT8, block 128, accuracy_level=1 (fp32 compute). Measured on
real decisions: 97.5% argmax agreement vs 55-82% for dynamic INT8 (which also quantizes
ACTIVATIONS) and 25-35% for weight-only int4. 149M params is too small to survive 4 bits.

Usage:
  python tools/prune_vocab_rerank.py --model /root/out/rerank_gte_mp \
      --keep /root/onnx/keep_ids.json --data <rerank.jsonl.gz> --work /root/onnx/pruned
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

TAR_CAP = 207257600
TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


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


def prune(model_dir, keep_path, out_dir):
    """Slice the embedding rows down to `keep`, write the pruned model + the id remap."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    keep_ids = json.load(open(keep_path))
    # specials must survive even if the sweep somehow missed one -- CLS/SEP/PAD are emitted
    # by the tokenizer on every single pair
    keep = sorted(set(int(i) for i in keep_ids) | set(int(i) for i in tok.all_special_ids))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager")
    if getattr(model.config, "reference_compile", None):
        model.config.reference_compile = False

    emb = model.get_input_embeddings()
    old_vocab, dim = emb.weight.shape
    with torch.no_grad():
        new_w = emb.weight[torch.tensor(keep)].clone()
    new_emb = torch.nn.Embedding(len(keep), dim, padding_idx=None)
    with torch.no_grad():
        new_emb.weight.copy_(new_w)
    model.set_input_embeddings(new_emb)
    model.config.vocab_size = len(keep)

    unk_new = keep.index(int(tok.unk_token_id))
    remap = np.full(old_vocab, unk_new, dtype=np.int32)
    remap[np.array(keep, dtype=np.int64)] = np.arange(len(keep), dtype=np.int32)

    # the config's special-token ids are old-vocab and now out of range; the ONNX path never
    # reads them (ids come in pre-remapped) but a PyTorch reload of this dir would warn or
    # index out of bounds
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "cls_token_id",
                 "sep_token_id", "unk_token_id", "mask_token_id"):
        old = getattr(model.config, attr, None)
        if isinstance(old, int) and 0 <= old < old_vocab:
            setattr(model.config, attr, int(remap[old]))

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    for f in TOK_FILES:
        src = os.path.join(model_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, f))
    np.save(os.path.join(out_dir, "vocab_remap.npy"), remap)
    print(f"[1/4] pruned vocab {old_vocab} -> {len(keep)} "
          f"({len(keep) / old_vocab:.1%}), embedding {old_vocab * dim * 4 / 2**20:.1f} -> "
          f"{len(keep) * dim * 4 / 2**20:.1f} MiB fp32; remap "
          f"{_mb(os.path.getsize(os.path.join(out_dir, 'vocab_remap.npy')))}", flush=True)
    return model, remap, keep


def export_fp32(model, out_path, max_len):
    import torch
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model = model.eval()
    seq = max_len // 2
    ids = torch.randint(0, model.config.vocab_size, (2, seq), dtype=torch.long)
    att = torch.ones((2, seq), dtype=torch.long)
    torch.onnx.export(
        model, (ids, att), out_path, input_names=["input_ids", "attention_mask"],
        output_names=["logits"], opset_version=17, do_constant_folding=True,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}})
    print(f"[2/4] fp32 onnx {_mb(os.path.getsize(out_path))}", flush=True)
    return out_path


def quantize_wonly(fp32_path, out_path, bits=8, block=128):
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    m = onnx.load(fp32_path)
    q = MatMulNBitsQuantizer(m, bits=bits, block_size=block, is_symmetric=True,
                             accuracy_level=1)   # 1 = fp32 compute; int8 compute is what broke
    q.process()
    q.model.save_model_to_file(out_path, use_external_data_format=False)
    print(f"[3/4] weight-only int{bits} blk{block}: raw {_mb(os.path.getsize(out_path))}  "
          f"tar.gz {_mb(_gz(out_path))}", flush=True)
    return out_path


def verify(ref_dir, onnx_paths, remap, recs, max_len):
    """Reference = the ORIGINAL full-vocab PyTorch model on raw ids; candidates = the pruned
    ONNX on remapped ids. Pruning itself should be numerically exact."""
    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ref_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref_model = AutoModelForSequenceClassification.from_pretrained(
        ref_dir, dtype=torch.float32, attn_implementation="eager")
    if getattr(ref_model.config, "reference_compile", None):
        ref_model.config.reference_compile = False
    ref_model = ref_model.eval().to(dev)

    batches, refs, n_oov = [], [], 0
    for r in recs:
        enc = tok([[r["state"], c] for c in r["candidates"]], padding=True,
                  truncation="only_first", max_length=max_len, return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        att = enc["attention_mask"].astype(np.int64)
        with torch.no_grad():
            out = ref_model(input_ids=torch.tensor(ids).to(dev),
                            attention_mask=torch.tensor(att).to(dev)
                            ).logits.squeeze(-1).float().cpu().numpy()
        n_oov += int((remap[ids] == remap[tok.unk_token_id]).sum()
                     - (ids == tok.unk_token_id).sum())
        batches.append((remap[ids].astype(np.int64), att))
        refs.append(out)
    del ref_model
    torch.cuda.empty_cache()
    print(f"[4/4] reference: {len(recs)} decisions, {n_oov} ids fell outside the kept set "
          f"(expect 0)", flush=True)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    rows = []
    for label, path in onnx_paths.items():
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        top = 0
        dsum = dmax = 0.0
        for (ids, att), ref in zip(batches, refs):
            got = sess.run(["logits"], {"input_ids": ids, "attention_mask": att})[0].reshape(-1)
            d = float(np.abs(ref - got).max())
            dsum += d
            dmax = max(dmax, d)
            top += int(int(np.argmax(ref)) == int(np.argmax(got)))
        raw, gz = os.path.getsize(path), _gz(path)
        rows.append(dict(label=label, argmax=top / len(refs), mean_diff=dsum / len(refs),
                         max_diff=dmax, raw_bytes=raw, gz_bytes=gz))
        print(f"  {label:22s} argmax {top / len(refs):6.1%}  |diff| mean "
              f"{dsum / len(refs):8.4f} max {dmax:8.4f}  raw {_mb(raw):>10s}  "
              f"tar.gz {_mb(gz):>10s}", flush=True)
        del sess
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--keep", required=True, help="keep_ids.json from the token sweep")
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--keep-fp32", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    pruned_dir = os.path.join(args.work, "model")
    model, remap, keep = prune(args.model, args.keep, pruned_dir)

    fp32 = export_fp32(model, os.path.join(args.work, "fp32", "model.onnx"), args.max_len)
    del model
    int8 = quantize_wonly(fp32, os.path.join(args.work, "model_wonly_int8.onnx"))

    recs = _records(args.data, args.n)
    paths = {"pruned_fp32": fp32, "pruned_wonly_int8": int8}
    rows = verify(args.model, paths, remap, recs, args.max_len)

    # what the real submission tarball costs: quantized graph + tokenizer + remap
    tok_bytes = sum(os.path.getsize(os.path.join(pruned_dir, f))
                    for f in TOK_FILES if os.path.exists(os.path.join(pruned_dir, f)))
    q_gz = next(r["gz_bytes"] for r in rows if r["label"] == "pruned_wonly_int8")
    print(f"\nBUDGET  model.tar.gz {_mb(q_gz)} + tokenizer {_mb(tok_bytes)} + remap 213 KB "
          f"+ onnxruntime(stripped) ~17.5 MiB + repo ~1 MiB  vs cap {_mb(TAR_CAP)}", flush=True)
    json.dump(rows, open(os.path.join(args.work, "pruned.json"), "w"), indent=1)
    if not args.keep_fp32:
        shutil.rmtree(os.path.dirname(fp32), ignore_errors=True)


if __name__ == "__main__":
    main()
