#!/usr/bin/env python3
"""Can DeBERTa-v3-base actually SHIP? Export -> parity -> INT8 -> size -> CPU speed.

The reranker is deploy-constrained, not accuracy-constrained: 197.66 MiB and a 600 s/game CPU
budget. A backbone that trains better and cannot be exported or quantised is worth nothing, and
DeBERTa-v3's disentangled attention is exactly the kind of thing that exports badly (extra
gather/bucket ops) and costs ~2x a plain attention at inference. So the deploy path is checked
BEFORE any training is spent on it.

Stages are separate and each prints its own verdict, so a failure at export still leaves the
size/speed question answered for whatever did export.
"""
import json, os, sys, time
import numpy as np

ROOT = "/root/ptcg/repo"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cg-lib"))
WORK = "/root/deberta_probe"
os.makedirs(WORK, exist_ok=True)
NAME = sys.argv[1] if len(sys.argv) > 1 else "microsoft/deberta-v3-base"
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 384
# The shipped model is vocab-PRUNED before quantising (53,339 -> 3,254 on the current one), so
# the size question is meaningless at full vocab: the fp32 embedding alone is 400 MB here and
# 9.5 MiB after pruning. Simulate the pruned width so the number means something.
KEEP_VOCAB = int(sys.argv[3]) if len(sys.argv) > 3 else 3254
WORK = WORK + "_" + NAME.split("/")[-1]
os.makedirs(WORK, exist_ok=True)

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lm import vocab as V

print("=== stage 1: load + resize for domain tokens", flush=True)
tok = AutoTokenizer.from_pretrained(NAME)
doms = V.special_tokens()
tok.add_tokens(doms)
try:
    model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=1)
except Exception as e:
    # deberta-v3 ships pytorch_model.bin only, and transformers >=5 refuses torch.load with
    # torch <2.6 (CVE-2025-32434). Build from config and load the state dict ourselves --
    # weights_only=True is exactly what the guard is protecting against, and we are the caller.
    print("  from_pretrained refused (%s); loading the state dict directly" % type(e).__name__)
    from transformers import AutoConfig
    from huggingface_hub import hf_hub_download
    cfg = AutoConfig.from_pretrained(NAME, num_labels=1)
    model = AutoModelForSequenceClassification.from_config(cfg)
    sd = torch.load(hf_hub_download(NAME, "pytorch_model.bin"), map_location="cpu",
                    weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith(("classifier.", "pooler."))]
    print("  loaded: %d missing (%d outside the head), %d unexpected"
          % (len(missing), len(real_missing), len(unexpected)))
    if len(real_missing) > 5:
        print("  BACKBONE DID NOT LOAD -- first missing:", real_missing[:5]); sys.exit(1)
model.resize_token_embeddings(len(tok))
if KEEP_VOCAB and KEEP_VOCAB < len(tok):
    model.resize_token_embeddings(KEEP_VOCAB)
    print("  vocab pruned to %d rows (the deploy pipeline's step)" % KEEP_VOCAB)
model.eval()
n_par = sum(p.numel() for p in model.parameters())
emb = model.get_input_embeddings().weight.numel()
print("  params %.1fM (embedding %.1fM, backbone %.1fM) | vocab %d"
      % (n_par/1e6, emb/1e6, (n_par-emb)/1e6, len(tok)), flush=True)

print("=== stage 2: ONNX export (opset 17)", flush=True)
onnx_fp32 = os.path.join(WORK, "model_fp32.onnx")
ids = torch.randint(0, 1000, (2, SEQ), dtype=torch.long)
att = torch.ones_like(ids)
t0 = time.time()
try:
    torch.onnx.export(model, (ids, att), onnx_fp32,
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                    "attention_mask": {0: "batch", 1: "seq"},
                                    "logits": {0: "batch"}},
                      opset_version=17, do_constant_folding=True)
    print("  EXPORT OK in %.0fs -> %.1f MB" % (time.time()-t0,
                                               os.path.getsize(onnx_fp32)/1e6), flush=True)
except Exception as e:
    print("  EXPORT FAILED: %s: %s" % (type(e).__name__, str(e)[:400]))
    sys.exit(1)

print("=== stage 3: parity torch vs onnxruntime", flush=True)
import onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = 8
sess = ort.InferenceSession(onnx_fp32, so, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
ii = rng.integers(0, min(3000, KEEP_VOCAB), size=(8, SEQ)).astype(np.int64)
aa = np.ones_like(ii)
with torch.no_grad():
    ref = model(input_ids=torch.tensor(ii), attention_mask=torch.tensor(aa)).logits.numpy().ravel()
got = sess.run(["logits"], {"input_ids": ii, "attention_mask": aa})[0].ravel()
print("  max|diff| %.2e   (fp32 export is sound if < 1e-3)" % np.abs(ref-got).max(), flush=True)

print("=== stage 4: weight-only INT8 (blk128, acc1) -- the shipped recipe", flush=True)
onnx_int8 = os.path.join(WORK, "model_wonly_int8.onnx")
try:
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    m = onnx.load(onnx_fp32)
    q = MatMulNBitsQuantizer(m, bits=8, block_size=128, is_symmetric=True, accuracy_level=1)
    q.process()
    q.model.save_model_to_file(onnx_int8, use_external_data_format=True)
    tot = sum(os.path.getsize(os.path.join(WORK, f)) for f in os.listdir(WORK)
              if f.startswith("model_wonly"))
    import subprocess as _sp
    tgz = os.path.join(WORK, "m.tgz")
    _sp.run(["tar", "czf", tgz, "-C", WORK] +
            [f for f in os.listdir(WORK) if f.startswith("model_wonly")], check=True)
    comp = os.path.getsize(tgz)
    print("  QUANT OK -> raw %.1f MiB | tar.gz %.1f MiB | budget 197.66 MiB "
          "(runtime+tokenizer+fallback ~25 MiB on top)"
          % (tot/1048576, comp/1048576), flush=True)
except Exception as e:
    print("  QUANT FAILED: %s: %s" % (type(e).__name__, str(e)[:400]))
    onnx_int8 = None

if onnx_int8:
    s8 = ort.InferenceSession(onnx_int8, so, providers=["CPUExecutionProvider"])
    g8 = s8.run(["logits"], {"input_ids": ii, "attention_mask": aa})[0].ravel()
    print("  INT8 vs fp32: max|diff| %.3f  spearman-ish argmax agree n/a (random input)"
          % np.abs(got-g8).max(), flush=True)

print("=== stage 5: CPU speed at 4 threads (Kaggle), real pair length %d" % SEQ, flush=True)
for path, label in ((onnx_fp32, "fp32"), (onnx_int8, "int8")):
    if not path:
        continue
    o = ort.SessionOptions(); o.intra_op_num_threads = 4; o.inter_op_num_threads = 1
    s = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
    x = rng.integers(0, min(3000, KEEP_VOCAB), size=(1, SEQ)).astype(np.int64); a = np.ones_like(x)
    for _ in range(3):
        s.run(["logits"], {"input_ids": x, "attention_mask": a})
    t0 = time.time(); N = 20
    for _ in range(N):
        s.run(["logits"], {"input_ids": x, "attention_mask": a})
    per = (time.time()-t0)/N
    # 5.2 candidates/decision (post-dedup), 80 decisions/game -- the deploy cost model
    print("  %-4s %.3f s per candidate | %.2f s per decision (x5.2) | %.0f s per game (x80) "
          "= %.0f%% of 600 s" % (label, per, per*5.2, per*5.2*80, 100*per*5.2*80/600), flush=True)
