set -x
pip install --no-cache-dir ninja packaging setuptools wheel
# nvcc in the image is 12.1 while torch is cu128; prefer the pip-installed 12.8 toolkit if present
NVCC_DIR=$(python3 -c "
import glob,os
c=glob.glob(\"/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvcc\")
print(c[0] if c else \"\")
" 2>/dev/null)
if [ -n "$NVCC_DIR" ] && [ -x "$NVCC_DIR/bin/nvcc" ]; then export CUDA_HOME="$NVCC_DIR"; fi
echo "CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}"
${CUDA_HOME:-/usr/local/cuda}/bin/nvcc --version | tail -1 || true
# --no-build-isolation so the build sees the INSTALLED torch 2.11.0+cu128, not a cu13 one that
# pip would fetch into an isolated overlay (that overlay torch failed CUDA init entirely).
MAX_JOBS=16 TORCH_CUDA_ARCH_LIST="8.9" CAUSAL_CONV1D_FORCE_BUILD=TRUE \
  pip install --no-cache-dir --no-build-isolation causal-conv1d
python3 -c "
import torch, causal_conv1d
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
print(\"torch\", torch.__version__, \"cuda_ok\", torch.cuda.is_available())
print(\"FAST PATH AVAILABLE:\", M.is_fast_path_available)
"
echo CCV_DONE
