"""Read checkpoint bias precision without loading the full model."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

root = Path("/model")
index = root / "model.safetensors.index.json"
mapping = json.loads(index.read_text())["weight_map"]
rows = []
for name, filename in mapping.items():
    if name.endswith("dt_bias"):
        with safe_open(str(root / filename), framework="pt", device="cpu") as f:
            value = f.get_tensor(name)
        rounded = value.to(torch.bfloat16).float()
        rows.append({
            "name": name, "checkpoint_dtype": str(value.dtype),
            "shape": list(value.shape),
            "bf16_round_maxabs": float((value.float() - rounded).abs().max()),
            "bf16_round_nonzero": int((value.float() != rounded).sum()),
        })
report = {
    "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
    "runtime": {"torch": torch.__version__}, "bias_parameters": rows,
}
path = Path("/results/model-bias-probe.json")
with path.open("x") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
