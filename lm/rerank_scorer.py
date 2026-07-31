"""Deploy scorer for the cross-encoder reranker: ONNX Runtime on CPU.

``OnnxRerankerScorer.score(prompt, candidates, obs=None) -> list[float]`` returns one
scalar logit per candidate (higher = better). It is the ``model`` consumed by
``lm.agent.make_lm_agent``, which takes the argmax -- so the output is always a legal move.
Same contract as ``lm.scorer.LlamaScorer``; the two are interchangeable.

Unlike the decoder, a cross-encoder CANNOT reuse a KV cache: state and candidate attend to
each other, so the state's representation differs per candidate and the whole
[state ; candidate] pair is re-encoded N times per decision. That is where the accuracy
comes from, and it is also the entire cost model -- latency scales with
(candidates x state length), not with candidate length.

TIME BANK: Kaggle gives a 600s CUMULATIVE thinking bank per game and running out is a
forfeit LOSS. Measured on the real decision distribution at 4 vCPU (the competition
runtime), a full game projects to ~433s -- inside the bank, but with a p90 of 13.3s and a
max of 34.7s per decision the tail is not safe. So ``score`` accumulates inference time and
RAISES once ``time_budget`` is spent; ``make_lm_agent`` catches that and falls back to
engine_v2 for the rest of the game. Pure safety -- it never routes on difficulty.

VOCAB REMAP: with a vocab-pruned model the tokenizer is UNCHANGED (dropping BPE merges
would silently change how an unseen card name tokenizes) and emits original ids; the
int32 ``vocab_remap.npy`` maps them onto the surviving embedding rows. See
tools/prune_vocab_rerank.py.
"""
import os
import time

_ACT = "[ACT]\n"


class OnnxRerankerScorer:
    def __init__(self, onnx_path, tok_dir, max_len=1024, threads=4, batch=48,
                 remap=None, time_budget=480.0):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.np = np
        self.remap = np.load(remap) if remap else None
        self.tok = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
        # direction="left" drops the HEAD (card-rules glossary) when a state overflows,
        # never the tail. The tail is the board and the option menu -- the decision itself.
        # With the default right-truncation the SEL menu survived in 1% of decisions.
        self.tok.enable_truncation(max_length=max_len, strategy="only_first",
                                   direction="left")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads       # cap BOTH: ORT otherwise oversubscribes on a
        so.inter_op_num_threads = 1             # many-core host and gets SLOWER, not faster
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.max_len = max_len
        self.batch = batch
        self.time_budget = time_budget
        self.spent = 0.0            # cumulative inference seconds THIS GAME (the time bank)
        self.n_decisions = 0

    def reset_bank(self):
        """New game -> refill the bank. ``make_lm_agent`` calls this at deck selection, so a
        persistent agent process cannot carry one game's spend into the next (that bug once
        tripped the fallback permanently after a few games)."""
        self.spent = 0.0
        self.n_decisions = 0

    def score(self, prompt, candidates, obs=None):
        if self.spent >= self.time_budget:
            raise RuntimeError("time budget spent -> engine fallback")
        np = self.np
        t0 = time.time()
        state = prompt[len(_ACT):] if prompt.startswith(_ACT) else prompt
        out = []
        for i in range(0, len(candidates), self.batch):
            chunk = candidates[i:i + self.batch]
            encs = self.tok.encode_batch([(state, c) for c in chunk])
            L = max(len(e.ids) for e in encs)
            ids = np.zeros((len(encs), L), dtype=np.int64)
            att = np.zeros((len(encs), L), dtype=np.int64)
            for j, e in enumerate(encs):
                ids[j, :len(e.ids)] = e.ids
                att[j, :len(e.ids)] = 1
            if self.remap is not None:
                ids = self.remap[ids].astype(np.int64)
            logits = self.sess.run(["logits"], {"input_ids": ids, "attention_mask": att})[0]
            out.extend(float(x) for x in logits.reshape(-1))
        self.spent += time.time() - t0
        self.n_decisions += 1
        return out
