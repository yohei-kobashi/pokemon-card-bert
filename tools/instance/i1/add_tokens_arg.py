import sys, os
sys.path.insert(0, "/root/ptcg/repo")
sys.path.insert(0, "/root/ptcg/repo/cg-lib")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm import vocab

SRC = sys.argv[1]
DST = sys.argv[2]

toks = vocab.special_tokens()
print("domain tokens to add:", len(toks), flush=True)
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(SRC, trust_remote_code=True, torch_dtype=torch.float16)
before = len(tok)
n_added = tok.add_tokens(toks)
model.resize_token_embeddings(len(tok))
print("vocab before=%d added=%d after=%d hidden=%s layers=%s tie=%s" % (
    before, n_added, len(tok), model.config.hidden_size,
    getattr(model.config, "num_hidden_layers", "?"),
    getattr(model.config, "tie_word_embeddings", "?")), flush=True)
os.makedirs(DST, exist_ok=True)
tok.save_pretrained(DST)
model.save_pretrained(DST)
print("SAVED_EXT", DST, flush=True)
