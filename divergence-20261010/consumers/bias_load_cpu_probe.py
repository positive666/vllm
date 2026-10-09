"""Diagnose actual Qwen GDN bias loading without executing a GPU kernel."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import cast


PINNED_HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
PINNED_MODULE_SHA = "d4056cb0105ca7c13fb3fa41be8f8e2ba3a2ea4f1ac81115abe3f3f4e6b3b45d"
PINNED_LOADER_SHA = "15eeb78befb71c75240f5b20c762381905a8f54f83f252de914330b1dddb7dfe"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-copy", action="store_true",
        help="Verify exact constructor/loader file SHAs without a Git checkout",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    expected_module = (
        repo / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    ).resolve()
    expected_loader = (repo / "vllm/model_executor/model_loader/weight_utils.py").resolve()
    if sha256(expected_module) != PINNED_MODULE_SHA:
        raise RuntimeError("Qwen constructor bytes do not match the reviewed d8 source")
    if sha256(expected_loader) != PINNED_LOADER_SHA:
        raise RuntimeError("Weight-loader bytes do not match the reviewed d8 source")
    head = None
    if not args.source_copy:
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != PINNED_HEAD:
            raise RuntimeError("Source head does not match the reviewed d8 source")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VLLM_GDN_DECODE_KERNEL"] = "triton"
    sys.path.insert(0, str(repo))

    import pytest
    import torch

    from vllm.config import KernelConfig, ModelConfig, VllmConfig
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
    from vllm.model_executor.model_loader import weight_utils
    from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    if Path(qwen_gdn_linear_attn.__file__).resolve() != expected_module:
        raise RuntimeError("Imported Qwen constructor is not the pinned source")
    if Path(weight_utils.__file__).resolve() != expected_loader:
        raise RuntimeError("Imported weight-loader module is not the pinned source")

    config_args = dict(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_key_heads=4,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    fp32_checkpoint = 1.0 + torch.arange(16, dtype=torch.float32) * 1e-4
    rows = []
    by_dtype = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            parallel_state, "_TP", SimpleNamespace(world_size=2, rank_in_group=1)
        )
        platform = qwen_gdn_linear_attn.current_platform
        monkeypatch.setattr(platform, "is_cpu", lambda: False)
        monkeypatch.setattr(platform, "is_cuda", lambda: True)
        monkeypatch.setattr(platform, "has_device_capability", lambda _: True)
        monkeypatch.setattr(platform, "current_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(
            qwen_gdn_linear_attn, "_get_flashinfer_gdn_decode", lambda: lambda: None
        )
        for checkpoint_dtype in (torch.bfloat16, torch.float32):
            checkpoint = fp32_checkpoint.to(checkpoint_dtype)
            observed = {}
            for backend in ("auto", "triton", "flashinfer"):
                hf_config = Qwen3NextConfig(**config_args)
                config = VllmConfig(
                    kernel_config=KernelConfig(gdn_decode_backend=backend)
                )
                config.model_config = cast(
                    ModelConfig,
                    SimpleNamespace(dtype=torch.bfloat16, hf_text_config=hf_config),
                )
                config.cache_config.mamba_ssm_cache_dtype = "float32"
                config.additional_config = {"gdn_prefill_backend": "triton"}
                with set_current_vllm_config(config), set_default_torch_dtype(
                    torch.bfloat16
                ):
                    layer = qwen_gdn_linear_attn.QwenGatedDeltaNetAttention(
                        hf_config, config, prefix="model.layers.0.linear_attn"
                    )
                # This is the real constructor-installed TP shard loader.
                layer.dt_bias.weight_loader(layer.dt_bias, checkpoint)
                actual = layer.dt_bias.detach().float().cpu()
                expected_dtype = (
                    torch.float32 if backend == "flashinfer" else torch.bfloat16
                )
                expected = checkpoint[8:].to(expected_dtype).float()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                assert layer.dt_bias.device.type == "cpu"
                assert not torch.cuda.is_initialized()
                observed[backend] = actual
                rows.append(
                    {
                        "checkpoint_dtype": str(checkpoint_dtype),
                        "backend": backend,
                        "parameter_dtype": str(layer.dt_bias.dtype),
                        "source_rank1_values": checkpoint[8:].float().tolist(),
                        "loaded_rank1_values": actual.tolist(),
                        "actual_loader_matches_selected_parameter_dtype": True,
                        "cuda_initialized": False,
                    }
                )
            torch.testing.assert_close(
                observed["auto"], observed["triton"], atol=0, rtol=0
            )
            equal = torch.equal(observed["triton"], observed["flashinfer"])
            assert equal == (checkpoint_dtype == torch.bfloat16)
            by_dtype[str(checkpoint_dtype)] = {
                "baseline_and_flashinfer_values_equal": equal,
                "max_abs_difference": float(
                    (observed["triton"] - observed["flashinfer"]).abs().max()
                ),
            }
    report = {
        "status": "CPU_LOAD_SEMANTICS_REPRODUCED",
        "source_head_binding": PINNED_HEAD,
        "git_head_verified": head,
        "source_binding": "Exact SHA256 checks of constructor and weight-loader files",
        "source_module_sha256": sha256(expected_module),
        "source_weight_loader_sha256": sha256(
            repo / "vllm/model_executor/model_loader/weight_utils.py"
        ),
        "probe_sha256": sha256(Path(__file__)),
        "torch_version": torch.__version__,
        "scope": "Real constructor and real sharded loader, mocked CUDA admission only",
        "gpu_kernels_executed": False,
        "model_output_or_accuracy_evaluated": False,
        "by_checkpoint_dtype": by_dtype,
        "rows": rows,
        "interpretation": (
            "BF16 checkpoint values are exact-promoted for FlashInfer. Non-BF16-"
            "representable FP32 checkpoint values are retained by FlashInfer but "
            "rounded to the baseline BF16 parameter dtype by auto/Triton. This "
            "does not explain divergence for the actual BF16 checkpoint."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "by_dtype": by_dtype}))


if __name__ == "__main__":
    main()
