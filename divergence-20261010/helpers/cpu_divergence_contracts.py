"""Check diagnostic dispatch and logical state contracts without GPU imports."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from divergence_contract import (
    initial_gate, logical_slots, prefix_contract, validate_kv_layout,
)


def leaf_module():
    path = Path(__file__).with_name("divergence_leaf.py")
    spec = importlib.util.spec_from_file_location("test_leaf_fresh", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, dtype, shape=(8, 5)):
        self.dtype = dtype
        self.shape = shape


def dummy_class(backend, calls):
    class Dummy:
        def __init__(self):
            self.backend = backend
            self._forward_method = (
                self.forward_native if backend == "triton" else self.forward_cuda
            )

        def forward_native(self, **arguments):
            calls.append(("triton", arguments))

        def forward_cuda(self, **arguments):
            calls.append(("flashinfer", arguments))

    return Dummy


class DiagnosticContracts(unittest.TestCase):
    def test_observer_owner_does_not_register_a_parent_module(self):
        import torch
        from divergence_capture import set_owner
        parent, child = torch.nn.Module(), torch.nn.Module()
        parent.add_module("decode", child)
        set_owner(child, parent)
        self.assertIs(child._divergence_owner_ref(), parent)
        self.assertEqual(list(child.children()), [])
        self.assertEqual([name for name, _ in parent.named_modules()], ["", "decode"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_constructor_and_typed_arguments_are_preserved_per_arm(self):
        for arm, configured, actual, dtype in (
            ("A", "triton", "triton", "torch.bfloat16"),
            ("B", "flashinfer", "triton", "torch.float32"),
            ("C", "flashinfer", "flashinfer", "torch.float32"),
        ):
            with self.subTest(arm=arm):
                leaf, calls, observations = leaf_module(), [], []
                cls = dummy_class(configured, calls)
                def observe(stage, op, arguments, contract, join):
                    observations.append(stage)
                    return {
                        **contract, "pure_decode": True,
                        "actual_pages": list(range(1, 9)),
                    }
                leaf.patch_class(cls, arm, observe)
                op = cls()
                arguments = {
                    "mixed_qkv": Tensor("torch.bfloat16"),
                    "a": Tensor("torch.bfloat16"),
                    "b": Tensor("torch.bfloat16"),
                    "A_log": Tensor("torch.float32"),
                    "dt_bias": Tensor(dtype),
                    "initial_state": Tensor("torch.float32"),
                    "ssm_state_indices": Tensor("torch.int32"),
                    "out": Tensor("torch.bfloat16"),
                }
                op._forward_method(**arguments)
                self.assertEqual(op.backend, configured)
                self.assertEqual([item[0] for item in calls], [actual])
                for key, tensor in arguments.items():
                    self.assertIs(calls[0][1][key], tensor)
                self.assertEqual(observations, ["before", "after"])
                with self.assertRaises(RuntimeError):
                    leaf.patch_class(cls, arm, observe)

    def test_leaf_rejects_unreviewed_pages_and_bias_dtype(self):
        leaf, calls = leaf_module(), []
        cls = dummy_class("flashinfer", calls)
        def observe(stage, op, arguments, contract, join):
            return {
                **contract, "pure_decode": True,
                "actual_pages": [-1] + list(range(2, 9)),
            }
        leaf.patch_class(cls, "B", observe)
        op = cls()
        args = {
            "mixed_qkv": Tensor("torch.bfloat16"),
            "a": Tensor("torch.bfloat16"), "b": Tensor("torch.bfloat16"),
            "A_log": Tensor("torch.float32"),
            "dt_bias": Tensor("torch.float32"),
            "initial_state": Tensor("torch.float32"),
            "ssm_state_indices": Tensor("torch.int32"),
            "out": Tensor("torch.bfloat16"),
        }
        with self.assertRaises(RuntimeError):
            op._forward_method(**args)
        self.assertEqual(calls, [])
        args["dt_bias"] = Tensor("torch.bfloat16")
        with self.assertRaises(RuntimeError):
            op._forward_method(**args)

    def test_prefix_rejects_history_or_position_drift_before_step19(self):
        prompt, reference = [10, 11], list(range(100, 130))
        prefix_contract(prompt, reference, prompt + reference[:19], 19, 118, 20)
        for history, token, pos in (
            (prompt + reference[:18] + [999], 118, 20),
            (prompt + reference[:19], 119, 20),
            (prompt + reference[:19], 118, 21),
        ):
            with self.assertRaises(ValueError):
                prefix_contract(prompt, reference, history, 19, token, pos)

    def test_logical_kv_excludes_tail_and_is_physical_remap_invariant(self):
        pool = [
            [[100 * page + offset, -(100 * page + offset)] for offset in range(4)]
            for page in range(5)
        ]
        def logical(table, storage):
            return [storage[p][o] for p, o in logical_slots(table, 6, 4, 5)]
        before = logical([3, 1, 4], pool)
        moved = list(pool)
        moved[0], moved[2] = pool[3], pool[1]
        self.assertEqual(before, logical([0, 2, 4], moved))
        tail_changed = [[list(item) for item in page] for page in pool]
        tail_changed[1][2] = [9999, 9999]
        tail_changed[4][0] = [8888, 8888]
        self.assertEqual(before, logical([3, 1, 4], tail_changed))
        tail_changed[3][0][1] += 1
        self.assertNotEqual(before, logical([3, 1, 4], tail_changed))
        with self.assertRaises(ValueError):
            logical_slots([3], 6, 4, 5)
        with self.assertRaises(ValueError):
            logical_slots([3, 3], 6, 4, 5)
        validate_kv_layout(
            "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl",
            [5, 2, 4, 512], 256, "torch.bfloat16",
        )
        with self.assertRaises(ValueError):
            validate_kv_layout("MLA", [5, 2, 4, 512], 256, "torch.bfloat16")

    def test_initial_state_missing_is_nogo(self):
        expected = [
            {"kind": kind, "rank": rank, "layer": layer, "index": 255}
            for kind, layer in (("conv", 0), ("recurrent", 0), ("kv", 3))
            for rank in (0, 1)
        ]
        self.assertEqual(
            initial_gate(expected, [255], [0], [3])["status"], "INITIAL_COMPLETE"
        )
        self.assertEqual(
            initial_gate(expected[:-1], [255], [0], [3])["status"], "NOGO"
        )
        with self.assertRaises(ValueError):
            initial_gate(expected + [expected[0]], [255], [0], [3])


if __name__ == "__main__":
    unittest.main()
