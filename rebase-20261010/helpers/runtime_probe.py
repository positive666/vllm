"""Record the imported runtime without changing packages or launching kernels."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata as md
import inspect
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def file_info(path):
    path = Path(path)
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def wheel_record(name, targets):
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        return {"installed": False}
    result = {"installed": True, "version": dist.version, "verified_files": []}
    record = dist.read_text("RECORD")
    if record is None:
        result["record_available"] = False
        return result
    result["record_available"] = True
    for relative, expected, size in csv.reader(io.StringIO(record)):
        path = Path(dist.locate_file(relative))
        if path.resolve() not in targets:
            continue
        entry = {"relative_path": relative, **file_info(path)}
        entry["record_hash"] = expected
        if expected:
            algorithm, wanted = expected.split("=", 1)
            actual = (
                base64.urlsafe_b64encode(
                    hashlib.new(algorithm, path.read_bytes()).digest()
                )
                .decode()
                .rstrip("=")
            )
            entry["record_matches"] = actual == wanted
        else:
            entry["record_matches"] = None
        entry["record_size_matches"] = not size or int(size) == entry["bytes"]
        result["verified_files"].append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("/results/runtime-probe.json")
    )
    parser.add_argument("--source", type=Path, default=Path("/source"))
    parser.add_argument(
        "--source-head", help="Declared base; independent git head is recorded."
    )
    args = parser.parse_args()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "probe": file_info(__file__),
        "python": {"executable": sys.executable, "version": sys.version},
        "declared_source_head": args.source_head,
        "source": str(args.source.resolve()),
        "modules": {},
        "errors": {},
        "cache_configuration": {
            name: os.environ.get(name)
            for name in (
                "PYTHONPATH",
                "VLLM_CACHE_ROOT",
                "TRITON_CACHE_DIR",
                "CUDA_CACHE_PATH",
                "XDG_CACHE_HOME",
                "FLASHINFER_WORKSPACE_BASE",
            )
        },
    }
    try:
        git = subprocess.run(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        report["observed_git_head"] = (
            git.stdout.strip() if git.returncode == 0 else None
        )
        report["git_probe_exit"] = git.returncode
    except OSError as exc:
        report["observed_git_head"] = None
        report["git_probe_exit"] = 127
        report["git_probe_error"] = f"{type(exc).__name__}: {exc}"
    imported = {}
    required = (
        "torch",
        "vllm._custom_ops",
        "flashinfer.gdn_decode",
        "flashinfer.jit.env",
        "cutlass",
        "cutlass.cute",
    )
    for name in (*required, "vllm._C", "vllm._C_stable_libtorch"):
        try:
            module = importlib.import_module(name)
            imported[name] = module
            report["modules"][name] = file_info(module.__file__)
            if hasattr(module, "__version__"):
                report["modules"][name]["imported_version"] = str(module.__version__)
        except Exception as exc:
            report["errors"][name] = f"{type(exc).__name__}: {exc}"
    torch = imported.get("torch")
    if torch is not None:
        try:
            report["torch"] = {
                "version": str(torch.__version__),
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
            }
            if torch.cuda.is_available():
                report["torch"]["devices"] = [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": list(torch.cuda.get_device_capability(i)),
                        "properties": str(torch.cuda.get_device_properties(i)),
                    }
                    for i in range(torch.cuda.device_count())
                ]
            present = hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")
            report["native_mtp"] = {
                "registered": present,
                "gpu_execution_verified": False,
            }
            if present:
                schemas = torch.ops._C.fused_gdn_decode_post_conv_mtp._schemas
                report["native_mtp"]["schemas"] = {
                    key: str(value) for key, value in schemas.items()
                }
                report["native_mtp"]["cuda_dispatch_registered"] = (
                    torch._C._dispatch_has_kernel_for_dispatch_key(
                        "_C::fused_gdn_decode_post_conv_mtp", "CUDA"
                    )
                )
        except Exception as exc:
            report["errors"]["torch_inventory"] = f"{type(exc).__name__}: {exc}"
    gdn = imported.get("flashinfer.gdn_decode")
    if gdn is not None:
        try:
            report["flashinfer_api"] = str(
                inspect.signature(gdn.gated_delta_rule_decode_pretranspose)
            )
            report["flashinfer_features"] = {
                name: getattr(gdn, name, None)
                for name in (
                    "_PRETRANSPOSE_AVAILABLE",
                    "_GDN_DECODE_BF16_STATE_AVAILABLE",
                )
            }
        except Exception as exc:
            report["errors"]["flashinfer_api"] = f"{type(exc).__name__}: {exc}"
    fi_env = imported.get("flashinfer.jit.env")
    if fi_env is not None:
        report["flashinfer_cache"] = {
            name: str(getattr(fi_env, name))
            for name in (
                "FLASHINFER_WORKSPACE_BASE",
                "FLASHINFER_CACHE_DIR",
                "FLASHINFER_JIT_DIR",
            )
            if hasattr(fi_env, name)
        }
    targets = {Path(value["resolved_path"]) for value in report["modules"].values()}
    try:
        fi_dist = md.distribution("flashinfer-python")
        for relative in (
            "flashinfer/gdn_decode.py",
            "flashinfer/jit/env.py",
            "flashinfer/gdn_kernels/gdn_decode_pretranspose.py",
            "flashinfer/gdn_kernels/gdn_decode_bf16_state.py",
        ):
            targets.add(Path(fi_dist.locate_file(relative)).resolve())
    except md.PackageNotFoundError:
        pass
    report["distributions"] = {}
    for name in (
        "torch",
        "vllm",
        "flashinfer-python",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-core",
        "nvidia-cutlass-dsl-libs-cu13",
        "nvidia-cutlass-dsl-libs-cu12",
        "apache-tvm-ffi",
    ):
        try:
            report["distributions"][name] = wheel_record(name, targets)
        except Exception as exc:
            report["errors"][f"RECORD:{name}"] = f"{type(exc).__name__}: {exc}"
    checked = [
        entry
        for dist in report["distributions"].values()
        for entry in dist.get("verified_files", [])
    ]
    report["record_integrity_failures"] = [
        entry
        for entry in checked
        if entry["record_matches"] is False or not entry["record_size_matches"]
    ]
    report["unverified_imported_files"] = sorted(
        targets - {Path(entry["resolved_path"]) for entry in checked}, key=str
    )
    report["unverified_imported_files"] = [
        str(path) for path in report["unverified_imported_files"]
    ]
    report["source_files"] = {}
    for relative in (
        "vllm/config/kernel.py",
        "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "vllm/v1/attention/backends/gdn_attn.py",
    ):
        if (args.source / relative).is_file():
            report["source_files"][relative] = file_info(args.source / relative)
    report["required_imports_succeeded"] = all(name in imported for name in required)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "required_imports_succeeded": report["required_imports_succeeded"],
                "native_mtp": report.get("native_mtp"),
                "record_integrity_failures": len(report["record_integrity_failures"]),
            }
        ),
        flush=True,
    )
    return int(
        not report["required_imports_succeeded"]
        or bool(report["record_integrity_failures"])
        or "torch_inventory" in report["errors"]
        or "flashinfer_api" in report["errors"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
