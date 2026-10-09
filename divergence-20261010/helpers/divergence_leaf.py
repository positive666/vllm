"""Select one diagnostic leaf without changing constructors or tensor values."""

from __future__ import annotations

import functools

_INSTALLED = False
_CONSTRUCTED = 0
_ORIGINALS = {}


def arm_contract(arm):
    """Return configured and actual leaf backends for the three cells."""
    choices = {
        "A": ("triton", "triton", "torch.bfloat16"),
        "B": ("flashinfer", "triton", "torch.float32"),
        "C": ("flashinfer", "flashinfer", "torch.float32"),
    }
    if arm not in choices:
        raise ValueError("Unknown diagnostic arm")
    configured, actual, bias_dtype = choices[arm]
    return {
        "arm": arm,
        "configured_backend": configured,
        "actual_leaf_backend": actual,
        "dt_bias_dtype": bias_dtype,
    }


def validate_arguments(contract, op, arguments, join):
    """Reject unsupported pages and changed parameter dtypes before dispatch."""
    if op.backend != contract["configured_backend"]:
        raise RuntimeError("GDN constructor backend differs from diagnostic arm")
    if str(arguments["dt_bias"].dtype) != contract["dt_bias_dtype"]:
        raise RuntimeError("Actual dt_bias dtype differs from constructor contract")
    if str(arguments["A_log"].dtype) != "torch.float32":
        raise RuntimeError("Actual A_log is not FP32")
    if str(arguments["initial_state"].dtype) != "torch.float32":
        raise RuntimeError("Actual recurrent state is not FP32")
    if not join or not join.get("pure_decode") or not join.get("actual_pages"):
        raise RuntimeError("Leaf lacks a verified pure-decode page join")
    pages = join["actual_pages"]
    if len(pages) != arguments["mixed_qkv"].shape[0]:
        raise RuntimeError("Leaf page count differs from packed decode rows")
    if any(not isinstance(page, int) or page <= 0 for page in pages):
        raise RuntimeError("Diagnostic leaf rejects padding and null state pages")
    if join.get("configured_backend") != contract["configured_backend"]:
        raise RuntimeError("Metadata page selection differs from constructor")
    if join.get("actual_leaf_backend") != contract["actual_leaf_backend"]:
        raise RuntimeError("Actual kernel selection differs from the arm")


def patch_class(cls, arm, observer):
    """Patch before construction and delegate once with identical arguments.

    Args:
        cls: Actual GDNDecode class.
        arm: A, B or C.
        observer: before/after callback; before returns a verified runtime join.
    """
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError("Diagnostic leaf installed twice in one process")
    if _CONSTRUCTED:
        raise RuntimeError("Diagnostic leaf installed after construction")
    contract = arm_contract(arm)
    originals = {
        "init": cls.__init__,
        "native": cls.forward_native,
        "cuda": cls.forward_cuda,
    }
    _ORIGINALS.update(originals)

    def dispatch(self, mixed_qkv, a, b, A_log, dt_bias, initial_state,
                 ssm_state_indices, out):
        arguments = {
            "mixed_qkv": mixed_qkv,
            "a": a,
            "b": b,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "initial_state": initial_state,
            "ssm_state_indices": ssm_state_indices,
            "out": out,
        }
        join = observer("before", self, arguments, contract, None)
        if join and join.get("excluded_profiling"):
            # Profiling has no real request/page join and is never evidence.
            # Keep its constructor backend, including FI for diagnostic B.
            original = originals[
                "native" if contract["configured_backend"] == "triton" else "cuda"
            ]
            result = original(self, **arguments)
            observer("after", self, arguments, contract, join)
            return result
        validate_arguments(contract, self, arguments, join)
        # B deliberately supplies FP32 bias to the original native function.
        # It never casts bias or changes the FlashInfer constructor/layout.
        original = originals[
            "native" if contract["actual_leaf_backend"] == "triton" else "cuda"
        ]
        result = original(self, **arguments)
        observer("after", self, arguments, contract, join)
        return result

    native = functools.wraps(originals["native"])(dispatch)
    cuda = functools.wraps(originals["cuda"])(dispatch)

    @functools.wraps(originals["init"])
    def initialize(self, *args, **kwargs):
        global _CONSTRUCTED
        originals["init"](self, *args, **kwargs)
        _CONSTRUCTED += 1
        if self.backend != contract["configured_backend"]:
            raise RuntimeError("Constructed backend differs from diagnostic arm")
        expected = native if arm == "A" else cuda
        if getattr(self._forward_method, "__func__", None) is not expected:
            raise RuntimeError("Constructor captured an unexpected leaf method")
        self._divergence_leaf_contract = dict(contract)

    cls.forward_native = native
    cls.forward_cuda = cuda
    cls.__init__ = initialize
    _INSTALLED = True
    return contract
