# Faster VGGT with Block-Sparse Global Attention

[📄 Arxiv](https://arxiv.org/abs/2509.07120) | [🌐 Project Page](https://brianwang00001.github.io/sparse-vggt/)

## Quick Start
Setup the environment:
```bash
# Clone the repository
git clone --recursive https://github.com/brianwang00001/sparse-vggt
cd sparse-vggt

# Install dependencies
uv sync

# Compile SpargeAttn
# Needs cuda installed (we used cuda 12.8)
uv pip install -e external/SpargeAttn/ --no-build-isolation
```

Try the sparse VGGT model:
```python
import torch
from vggt.models.vggt import VGGT
from sparse_vggt.models.vggt import sparse_aggregator_from_vggt

# Load the original VGGT model
model = VGGT.from_pretrained("facebook/VGGT-1B")

# Replace the aggregator with the sparse aggregator
# Note: `aux_output_store` is a dictionary of auxiliary outputs from the global attention
# You can use it to get the sparsity of the global attention
sparse_aggregator, aux_output_store = sparse_aggregator_from_vggt(
    model.aggregator,
    sparse_ratio=0.1,  # example config
    cdf_threshold=0.97,  # example config
)
model.aggregator = sparse_aggregator

# Use the sparse model as usual
model.cuda()
model.eval()
images = torch.randn(10, 3, 518, 378).cuda()
with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(images)
```

Similar for Pi3:
```python
import torch
from pi3.models.pi3 import Pi3
from sparse_vggt.models.pi3 import sparse_model_from_pi3

model = Pi3.from_pretrained("yyfz233/Pi3")
model, aux_output_store = sparse_model_from_pi3(model, sparse_ratio=0.1, cdf_threshold=0.97)
```

