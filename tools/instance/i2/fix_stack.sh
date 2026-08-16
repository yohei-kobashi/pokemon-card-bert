set -eux
pip install --no-cache-dir torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python3 -c "import torch;print(\"torch\",torch.__version__,\"avail\",torch.cuda.is_available())"
pip install --no-cache-dir causal-conv1d
python3 -c "
import torch, causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())
print(\"FAST PATH AVAILABLE:\", M.is_fast_path_available)
"
echo FIX_DONE
