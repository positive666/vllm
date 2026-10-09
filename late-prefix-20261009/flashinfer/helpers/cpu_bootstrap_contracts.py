"""Exercise actual artifact installation and deferred hook composition on CPU.

Stub model classes stand in for CUDA modules; real install/loaders/patch methods
run, while numerical sampler and kernel execution are deliberately not invoked.
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from late_plan import validate_inputs


class CombinedObserverBootstrap(unittest.TestCase):
    def test_actual_install_and_deferred_patch_keep_native_and_capture_hooks(self):
        import force_decode as common
        import force_decode_v2 as native
        import late_bootstrap
        import late_capture
        import launch_capture as capture
        package = Path(__file__).resolve().parent.parent
        _, examples = validate_inputs(package)
        with tempfile.TemporaryDirectory(prefix="gdn-late-bootstrap-") as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "sample").mkdir(parents=True)
            for name in ("model_runner.py", "input_batch.py", "states.py",
                         "sample/sampler.py", "sample/trace_replay.py", "gdn.py"):
                (source / name).write_text("CPU stub source: " + name, encoding="utf-8")
            def hash_file(name):
                return hashlib.sha256((source / name).read_bytes()).hexdigest()
            fixture = root / "fixture.json"
            fixture.write_text(json.dumps({"examples": examples}), encoding="utf-8")
            force_config = root / "force-config.json"
            force_config.write_text(json.dumps({"output_dir": str(root / "trace"),
                "fixture_path": str(fixture), "target_indices": [209, 255], "limit": 3500,
                "snapshot_target": 255, "snapshot_steps": [1, 18, 19, 1750, 3497, 3498, 3499],
                "expected_runner_sha256": hash_file("model_runner.py"),
                "expected_input_batch_sha256": hash_file("input_batch.py")}), encoding="utf-8")
            capture_config = root / "capture-config.json"
            capture_config.write_text(json.dumps({"output_dir": str(root / "snapshots"),
                "start_marker": str(root / "START"), "early_calls": [], "sample_calls": [],
                "sample_layers": [], "initial_batch_size": 8,
                "expected_module_sha256": hash_file("gdn.py")}), encoding="utf-8")
            environment = {"GDN_LATE_ACTIVE": "1", "GDN_FORCE_ACTIVE": "1",
                           "GDN_FORCE_CONFIG": str(force_config), "GDN_CAPTURE_ACTIVE": "1",
                           "GDN_CAPTURE_CONFIG": str(capture_config)}
            class Runner:
                def __init__(self):
                    self.scheduler_config = types.SimpleNamespace(async_scheduling=False)
                    self.speculative_config = self.speculator = self.pcp_manager = None
                    self.model_config = types.SimpleNamespace(enforce_eager=True,
                                                              enable_trace_replay=True)
                    self.is_last_pp_rank = True
                    self.sampler = object()
                def add_requests(self, scheduler_output):
                    return "native add returned"
                def sample(self, hidden_states, input_batch, grammar_output):
                    if not isinstance(self.sampler, native._SamplerObserver):
                        raise AssertionError("Native sampler observer was not installed")
                    return "native sample returned"
                def prepare_inputs(self):
                    return "native prepare"
                def execute_model(self):
                    return "native execute"
            class Decode:
                backend = "triton"
                def forward_native(self, **kwargs):
                    return "native GDN returned"
                def forward_cuda(self, **kwargs):
                    return "CUDA GDN returned"
            class Attention:
                def __init__(self, prefix):
                    self.gdn_decode = Decode()
                    self.tp_rank = 0
            class Loader:
                def exec_module(self, module):
                    module.original_loader_called = True
            gpu_module = types.SimpleNamespace(__file__=str(source / "model_runner.py"),
                                               GPUModelRunner=Runner)
            gdn_module = types.SimpleNamespace(__file__=str(source / "gdn.py"),
                QwenGatedDeltaNetAttention=Attention, GDNDecode=Decode)
            with patch.dict(os.environ, environment):
                runpy.run_path(str(package / "helpers/late_site/sitecustomize.py"))
                self.assertEqual(os.environ["GDN_CAPTURE_ACTIVE"], "1")
                self.assertIsNotNone(common._CONFIG)
                self.assertIsNotNone(capture._CONFIG)
                self.assertEqual(sum(isinstance(item, common._ForceFinder)
                                     for item in sys.meta_path), 1)
                self.assertEqual(sum(isinstance(item, capture._CaptureFinder)
                                     for item in sys.meta_path), 1)
                common._ForceLoader(Loader()).exec_module(gpu_module)
                capture._CaptureLoader(Loader()).exec_module(gdn_module)
                self.assertTrue(gpu_module.original_loader_called)
                self.assertTrue(gdn_module.original_loader_called)
                self.assertTrue(common._PATCHED and late_capture._PATCHED and capture._PATCHED)
                runner = Runner()
                sampler = runner.sampler
                self.assertEqual(runner.add_requests(types.SimpleNamespace(scheduled_new_reqs=[])),
                                 "native add returned")
                self.assertEqual(runner.sample(None, None, None), "native sample returned")
                self.assertIs(runner.sampler, sampler)
                self.assertEqual(runner.execute_model(), "native execute")
                attention = Attention("model.layers.0.linear_attn")
                self.assertEqual(attention.gdn_decode._gdn_capture_context["layer_idx"], 0)
                fake = types.SimpleNamespace(shape=(1, 1))
                self.assertEqual(attention.gdn_decode.forward_native(fake, fake, fake, fake,
                                                                     fake, fake, fake, fake),
                                 "native GDN returned")
                self.assertTrue((root / "trace" / ("patched-pid-" + str(os.getpid()) + ".json")).is_file())
                self.assertTrue((root / "snapshots" / ("patched-pid-" + str(os.getpid()) + ".json")).is_file())
                with self.assertRaisesRegex(RuntimeError, "installed twice"):
                    common._patch(gpu_module)
                with self.assertRaisesRegex(RuntimeError, "must precede actual runner import"):
                    late_bootstrap.install()
                self.assertEqual(os.environ["GDN_CAPTURE_ACTIVE"], "1")


if __name__ == "__main__":
    unittest.main()
