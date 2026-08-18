"""PyTorch scorer for the cross-encoder reranker -- the研究 path (local CPU / Colab GPU).

Same contract as ``lm.rerank_scorer.OnnxRerankerScorer`` and ``lm.scorer.LlamaScorer``:
``score(prompt, candidates, obs=None) -> list[float]``, one logit per candidate, consumed by
``lm.agent.make_lm_agent`` which takes the argmax -- so the move is always legal.

Why a second scorer instead of reusing the ONNX one: a model that has just been fine-tuned on
Colab exists only as HuggingFace weights. Exporting and quantising it to ONNX before it can be
played would put two more steps (and two more ways to be wrong) between "training finished" and
"see whether it is stronger". This class loads the checkpoint directly, on whatever device is
available.

The tokenizer settings are NOT defaults: ``truncation_side='left'`` and
``truncation='only_first'`` drop the HEAD of an over-long state, never the board and the option
menu at the tail. Right-truncation once deleted the menu in 99% of decisions and the model
played blind. Keep this identical to lm/rerank_scorer.py and tools/dusk_plan_train.py.
"""
import os
import time

_ACT = "[ACT]\n"


class HfRerankerScorer:
    def __init__(self, model_dir, device="auto", max_len=512, batch=32, dtype="auto",
                 time_budget=0.0):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            # fp16 on GPU is ~2x faster and matches argmax closely; on CPU it is EMULATED and
            # several times slower than fp32, so the same flag must not mean the same thing.
            td = torch.float16 if device == "cuda" else torch.float32
        else:
            td = getattr(torch, dtype)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.truncation_side = "left"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, dtype=td).to(device).eval()
        self.device, self.max_len, self.batch = device, max_len, batch
        self.time_budget = time_budget      # 0 = no bank (local play has no Kaggle clock)
        self.spent = 0.0
        self.n_decisions = 0

    def reset_bank(self):
        self.spent = 0.0
        self.n_decisions = 0

    def score(self, prompt, candidates, obs=None):
        if self.time_budget and self.spent >= self.time_budget:
            raise RuntimeError("time budget spent -> engine fallback")
        torch = self.torch
        t0 = time.time()
        state = prompt[len(_ACT):] if prompt.startswith(_ACT) else prompt
        out = []
        for i in range(0, len(candidates), self.batch):
            chunk = candidates[i:i + self.batch]
            enc = self.tok([[state, c] for c in chunk], padding=True, truncation="only_first",
                           max_length=self.max_len, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.squeeze(-1)
            out.extend(float(x) for x in logits.float().reshape(-1))
        self.spent += time.time() - t0
        self.n_decisions += 1
        return out


def resolve_model(path_or_repo):
    """A local checkpoint directory, or a HuggingFace repo id to pull.

    Accepts a Drive path straight from Colab ("/content/drive/MyDrive/PTCG/models/r1") as
    well as "yoheikobashi/ptcg-dusknoir-deberta-reranker", so the same flag serves the
    published baseline and whatever the研究 just trained."""
    path_or_repo = os.path.expanduser(path_or_repo)
    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo,
                             allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
