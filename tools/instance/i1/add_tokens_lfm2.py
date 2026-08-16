import sys, os
sys.path.insert(0, "/root/ptcg/repo")
sys.path.insert(0, "/root/ptcg/repo/cg-lib")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm import vocab

SRC = "/root/lfm2_230m"
DST = "/root/lfm2_230m_ext"

toks = vocab.special_tokens()
print("domain tokens to add:", len(toks), flush=True)

tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(SRC, trust_remote_code=True, torch_dtype=torch.float16)
before = len(tok)
n_added = tok.add_tokens(toks)
model.resize_token_embeddings(len(tok))
after = len(tok)
print("vocab before=%d added=%d after=%d" % (before, n_added, after), flush=True)
try:
    print("hidden_size=%s tie_word_embeddings=%s" % (
        model.config.hidden_size, getattr(model.config, "tie_word_embeddings", "?")), flush=True)
except Exception as e:
    print("cfg", e, flush=True)

os.makedirs(DST, exist_ok=True)
tok.save_pretrained(DST)
model.save_pretrained(DST)
print("SAVED_EXT", flush=True)
