"""Offline contracts for image preservation and complete registry blob verification."""

import copy
import gzip
import io
import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import repack
import verify_manifest as verify


class ReleaseContracts(unittest.TestCase):
    def test_linux_legacy_args_escaped_remains_in_expected_metadata(self):
        config = {"ArgsEscaped": True, "Entrypoint": ["/entry", "serve"]}
        self.assertIn('ENTRYPOINT ["/entry", "serve"]',
                      repack.runtime_lines(config, os_name="linux"))
        self.assertIs(config["ArgsEscaped"], True)
        with self.assertRaises(ValueError):
            repack.runtime_lines(config, os_name="windows")

    def fixture_policy(self):
        return {
            "active_site_packages": "opt/app/.venv/lib/python3.12/site-packages",
            "system_site_packages": "usr/local/lib/python3.12/dist-packages",
            "remove_shadowed": [
                {
                    "path": "usr/local/lib/python3.12/dist-packages/flashinfer_cubin",
                    "legacy_name": "flashinfer-cubin",
                    "legacy_version": "0.7.2",
                    "legacy_metadata": (
                        "usr/local/lib/python3.12/dist-packages/"
                        "flashinfer_cubin-0.7.2.dist-info/METADATA"
                    ),
                    "replacement_metadata": (
                        "opt/app/.venv/lib/python3.12/site-packages/"
                        "flashinfer_cubin-0.7.3.dist-info/METADATA"
                    ),
                    "replacement_version": "0.7.3",
                }
            ],
            "remove_caches": [],
        }

    def fixture_snapshot(self):
        policy = self.fixture_policy()
        rule = policy["remove_shadowed"][0]
        return {
            "entries": {
                policy["active_site_packages"]
                + "/flashinfer_cubin/cubins/keep.cubin": {"type": "file"},
                rule["legacy_metadata"]: {
                    "type": "file",
                    "package": {"name": "flashinfer-cubin", "version": "0.7.2"},
                },
                rule["replacement_metadata"]: {
                    "type": "file",
                    "package": {"name": "flashinfer-cubin", "version": "0.7.3"},
                },
                rule["path"] + "/old.cubin": {"type": "file"},
                policy["active_site_packages"]
                + "/flashinfer_jit_cache/providers/sm103a/kernel.so": {"type": "file"},
                "usr/local/cuda": {
                    "type": "symlink",
                    "target": "/usr/local/lib/python3.12/dist-packages/nvidia/cu13",
                },
            }
        }

    def test_only_guarded_legacy_payload_is_removed(self):
        policy, snapshot = self.fixture_policy(), self.fixture_snapshot()
        self.assertEqual(
            repack.removal_paths(snapshot, policy),
            [policy["remove_shadowed"][0]["path"]],
        )
        bad = copy.deepcopy(snapshot)
        bad["entries"][policy["remove_shadowed"][0]["legacy_metadata"]]["package"][
            "version"
        ] = "0.7.3"
        with self.assertRaises(ValueError):
            repack.removal_paths(bad, policy)
        bad = copy.deepcopy(snapshot)
        del bad["entries"][
            policy["active_site_packages"] + "/flashinfer_cubin/cubins/keep.cubin"
        ]
        with self.assertRaises(ValueError):
            repack.removal_paths(bad, policy)
        policy["remove_caches"] = ["usr/local/cuda/include"]
        with self.assertRaises(ValueError):
            repack.removal_paths(snapshot, policy)

    def test_retained_symlink_cannot_depend_on_removed_cache(self):
        policy, snapshot = self.fixture_policy(), self.fixture_snapshot()
        policy["remove_caches"] = ["root/.cache/uv"]
        snapshot["entries"]["opt/app/compiler"] = {
            "type": "symlink",
            "target": "/root/.cache/uv/bin/compiler",
        }
        with self.assertRaises(ValueError):
            repack.removal_paths(snapshot, policy)

    def test_runtime_cmd_and_stop_signal_survive_generation(self):
        config = {
            "Env": ["LITERAL=$value"],
            "Labels": {"revision": "new"},
            "User": "1000:1000",
            "Entrypoint": ["/entry"],
            "Cmd": ["serve", "--port", "8000"],
            "StopSignal": "SIGTERM",
            "WorkingDir": "/app",
            "ExposedPorts": {"8000/tcp": {}},
        }
        lines = repack.runtime_lines(config)
        self.assertIn('CMD ["serve", "--port", "8000"]', lines)
        self.assertIn("USER 1000:1000", lines)
        self.assertIn("STOPSIGNAL SIGTERM", lines)
        self.assertIn("WORKDIR /app", lines)
        self.assertIn('ENV LITERAL="\\$value"', lines)
        with self.assertRaises(ValueError):
            repack.runtime_lines({**config, "Volumes": {"/data": {}}})

    def test_export_inventory_preserves_cuda_links_and_header_bytes(self):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            item = tarfile.TarInfo("usr/local/cuda/include/keep.h")
            item.size = 6
            item.mode = 0o644
            archive.addfile(item, io.BytesIO(b"header"))
            item = tarfile.TarInfo("usr/local/cuda/lib64/stubs/libcuda.so")
            item.type, item.linkname = tarfile.SYMTYPE, "../../stubs/libcuda.so.1"
            archive.addfile(item)
        entries = repack.inventory(io.BytesIO(output.getvalue()))
        self.assertEqual(
            entries["usr/local/cuda/include/keep.h"]["sha256"],
            verify.digest_bytes(b"header").split(":")[1],
        )
        self.assertEqual(
            entries["usr/local/cuda/lib64/stubs/libcuda.so"]["target"],
            "../../stubs/libcuda.so.1",
        )

    def fixture_graph(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            member = tarfile.TarInfo("usr/lib/fixture.so")
            member.size = 7
            archive.addfile(member, io.BytesIO(b"fixture"))
        layer_data = gzip.compress(stream.getvalue(), mtime=0)
        diff = verify.digest_bytes(stream.getvalue())
        config = json.dumps(
            {
                "architecture": "amd64",
                "os": "linux",
                "config": {},
                "rootfs": {"type": "layers", "diff_ids": [diff]},
            }
        ).encode()
        descriptor = lambda data, media: {
            "digest": verify.digest_bytes(data),
            "size": len(data),
            "mediaType": media,
        }
        config_desc = descriptor(config, "application/vnd.oci.image.config.v1+json")
        layer_desc = descriptor(
            layer_data, "application/vnd.oci.image.layer.v1.tar+gzip"
        )
        raw = json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "config": config_desc,
                "layers": [layer_desc],
            }
        ).encode()
        return raw, {config_desc["digest"]: config, layer_desc["digest"]: layer_data}

    def test_all_blob_bytes_and_uncompressed_diffid_are_checked(self):
        raw, blobs = self.fixture_graph()
        calls = []

        def fetch(item):
            calls.append(item["digest"])
            return io.BytesIO(blobs[item["digest"]])

        result = verify.verify_graph(raw, verify.digest_bytes(raw), fetch)
        self.assertEqual(len(calls), 2)
        self.assertTrue(result["config"]["full_readback"])
        self.assertTrue(result["layers"][0]["full_readback"])
        layer = json.loads(raw)["layers"][0]
        blobs[layer["digest"]] = blobs[layer["digest"]][:-1] + b"x"
        with self.assertRaises((ValueError, OSError, EOFError)):
            verify.verify_graph(raw, verify.digest_bytes(raw), fetch)

    def test_cross_host_redirect_drops_authentication(self):
        request = Request(
            "https://ghcr.io/v2/example/image/blobs/sha256:abc",
            headers={"Authorization": "Bearer fixture-secret"},
        )
        redirected = verify.SafeRedirect().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://cdn.example/blob?signature=fixture",
        )
        self.assertFalse(any(k.lower() == "authorization" for k in redirected.headers))
        with self.assertRaises(ValueError):
            verify.SafeRedirect().redirect_request(
                request, None, 302, "Found", {}, "http://cdn.example/blob"
            )

    def test_duplicate_payload_in_flattened_layers_is_rejected(self):
        raw, blobs = self.fixture_graph()
        manifest = json.loads(raw)
        config = json.loads(blobs[manifest["config"]["digest"]])
        config["rootfs"]["diff_ids"] *= 2
        config_raw = json.dumps(config).encode()
        manifest["config"].update(
            digest=verify.digest_bytes(config_raw), size=len(config_raw)
        )
        blobs[manifest["config"]["digest"]] = config_raw
        manifest["layers"] *= 2
        raw = json.dumps(manifest).encode()
        with self.assertRaises(ValueError):
            verify.verify_graph(
                raw,
                verify.digest_bytes(raw),
                lambda item: io.BytesIO(blobs[item["digest"]]),
            )

    def test_file_or_cmd_loss_fails_the_repack_gate(self):
        config = {"Cmd": ["serve"], "Labels": {"revision": "new"}}
        before = {
            "image_id": "sha256:" + "a" * 64,
            "config": config,
            "entries": {"sdk/keep.h": {"type": "file", "sha256": "correct"}},
        }
        after = {**copy.deepcopy(before), "image_id": "sha256:" + "b" * 64}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(
                before=root / "before",
                after=root / "after",
                plan=root / "plan",
                output=root / "result",
            )
            repack.save(args.before, before)
            repack.save(
                args.plan,
                {
                    "source_image_id": before["image_id"],
                    "removed_paths": [],
                    "expected_runtime_config": config,
                },
            )
            repack.save(args.after, after)
            with redirect_stdout(io.StringIO()):
                repack.compare(args)
            after["entries"]["sdk/keep.h"]["sha256"] = "corrupt"
            repack.save(args.after, after)
            with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                repack.compare(args)
            after["entries"] = before["entries"]
            after["config"] = {}
            repack.save(args.after, after)
            with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                repack.compare(args)

    def test_wrong_compressed_digest_cannot_pass_length_only(self):
        descriptor = {"digest": verify.digest_bytes(b"good"), "size": 4}
        with self.assertRaises(ValueError):
            verify.read_blob(io.BytesIO(b"evil"), descriptor)

    def test_offline_plan_binds_actual_versions_and_retains_cmd(self):
        policy, source = self.fixture_policy(), self.fixture_snapshot()
        source.update(
            image_id="sha256:" + "a" * 64,
            architecture="amd64",
            os="linux",
            config={
                "Cmd": ["serve"],
                "Labels": {"com.nvidia.fixture": "keep", "old.release": "drop"},
            },
        )
        active = policy["active_site_packages"]
        for name, version in (("vllm", "0.1+g1234567"), ("lmcache", "0.1+gabcdef0")):
            source["entries"][active + f"/{name}.dist-info/METADATA"] = {
                "type": "file",
                "package": {"name": name, "version": version},
            }
        labels = {
            "com.nvidia.fixture": "keep",
            "org.opencontainers.image.version": "0.1+g1234567",
            "org.opencontainers.image.revision": "1234567" + "0" * 33,
            "ai.positive666.lmcache.revision": "abcdef0" + "0" * 33,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(
                snapshot=root / "source",
                policy=root / "policy",
                labels_json=root / "labels",
                image_id=source["image_id"],
                source_ref="local/source:fixture",
                output_dir=root / "plan",
            )
            for path, value in (
                (args.snapshot, source),
                (args.policy, policy),
                (args.labels_json, labels),
            ):
                repack.save(path, value)
            with redirect_stdout(io.StringIO()):
                repack.plan(args)
            planned = repack.load(args.output_dir / "REPACK_PLAN.json")
            self.assertEqual(planned["expected_runtime_config"]["Cmd"], ["serve"])
            self.assertNotIn(
                "old.release", planned["expected_runtime_config"]["Labels"]
            )
            labels["org.opencontainers.image.version"] = "stale"
            repack.save(args.labels_json, labels)
            with self.assertRaises(ValueError):
                repack.plan(args)


if __name__ == "__main__":
    unittest.main()
