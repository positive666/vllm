# H20 / SM90 integrated vLLM + LMCache release

Built from the frozen 2026-11-03 sources; published on 2026-11-04 (Asia/Shanghai).

```bash
docker pull ghcr.io/positive666/model@sha256:ea31ef16a4f389f30f6ec7651814e4a3aeda7fa54b2dc012e43323313b6965fb
```

Readable tag: `ghcr.io/positive666/model:glm53-febcc8ca7-lmc386473dad-h145b-mtp63432-cu130-sm90-20261103-rc2`.
The immutable digest identifies the artifact that passed acceptance. A later rebuild is a new artifact requiring its own validation.

## Source and dependency scope

- vLLM `febcc8ca793b3bbea52e5dabc0ca4f491a3fd3fc`: complete official wheel, including native binaries. Three Python backport files are applied in this Git branch.
- LMCache `386473dad97fdad4112a7333bc6604185acfd084`: Python and native extension rebuilt together; the 15-file preserved runtime diff is `patches/lmcache-preserved.patch`.
- Humming `145bed14896b34b35bd6934dc35063975b47ee28` / `0.1.19.post1`; includes existing SM90 padded-K support and the upstream FIPS-compatible cache hashing change.
- NCCL `2.31.2`; DeepEP `d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb` rebuilt against it. NVSHMEM `3.4.5` is intentionally retained, differing from the upstream install-script default.
- Torch `2.14.0+cu130`, Triton `3.8.0`, FlashInfer `0.7.2.post1`, CUTLASS DSL `4.8.0` remain pinned.
- The installer also applies the documented DeepGEMM SM90 WGMMA register-lifetime header fix and updates package RECORD entries.
- No default NUMA, NVLS or symmetric-memory disabling flags are injected. This is a patched integration image, not an unmodified official image.

LMCache upstream #6256, #6119 and #4072 are integrated with the existing request/session, read-lock and transport fixes. Four overlapping hunks in two files were resolved; exact source hashes are in [patches.json](patches.json).

**Scope correction:** the image contains #5673-style L2 LOAD/PREFETCH destination-buffer protection. It does **not** contain draft #6171's L2 STORE source pins and shutdown drain. Default L1-only acceptance does not exercise that L2 STORE path. Do not present the two fixes as interchangeable.

The #63432-equivalent Qwen3.8-Flash-Next draft-position fix is retained without changing target-model YaRN behavior. Twelve config regressions passed on this image. The previous release's 13 full-model requests covered 262K, 524K, near-1M and interleaved short requests; they are historical evidence, not a full-model rerun of this image.

## Validation and limitations

- Components: 890 passed, 13 architecture/shape skips, zero failures.
- Eight-rank NCCL all-reduce passed for four message sizes, with runtime `ncclGetVersion()` reporting 23102. Torch's compile-time NCCL version is not the runtime-library probe.
- DeepSeek + LMCache: 19 completed synthetic requests plus one intentional cancellation; cold store, CPU restore, branching, C4/C8, cancellation cleanup and model-restart reuse passed.
- CPU restore/restart evidence includes zero GPU-prefix hits, 3,840 external-hit tokens and 48 rank-level loaded chunks.
- Original experiment service was restored. This record does not claim a production rollout, long soak, throughput gain, cross-node DeepEP/GDR/PD acceptance, or new Qwen full-model evaluation.
- The registry manifest/config hashes and every object's existence/length were checked. Fleet delivery additionally requires a complete archive, per-object SHA256 and fresh-content-store import; `docker inspect` alone is insufficient.

See [versions.json](versions.json), [published.json](published.json), [component results](validation/components.json), [service summary](validation/service-summary.json), and [detailed Chinese change record](CHANGES.zh-CN.txt).

## Build records

`build-record/` preserves the installer and audit logic as `.py.txt` source records, preventing vLLM hooks from rewriting foreign build evidence. The LMCache diff applies to the pinned upstream source; it is not a new standalone upstream PR. The Dockerfile base reference is normalized to its verified digest. The directory is **not a standalone turnkey build context**: the source archives, full context, prebuilt input wheels and rebuilt LMCache/Humming/DeepEP wheels named and hashed in `versions.json` are required. They are not committed as large binary files. The published digest is the delivery artifact.

AI assistance was used for integration, conflict resolution, validation and these records. Unmerged third-party fixes retain attribution in the change record; this release branch is not a proposal to merge all patches into upstream vLLM.

## Subsequent LAN delivery verification (2026-11-04)

The identical rc2 manifest was imported on nodes 26, 27, 28, 29 and 176.
Each target verified all 87 unique content objects, 7,090 installed RECORD files,
18 Python patch files, 14 native libraries and 273 dependency versions.
A separate initially empty content store imported the full OCI archive successfully;
this distribution is not a manifest-only or thin archive.

Existing model and LMCache services retained their image IDs and start times.
29 stopped experiment containers had their writable changes archived before removal;
one obsolete build image and about 169 GiB of temporary transfer/export and isolated import files were removed.
The complete offline archive remains on the source node. Rollback images and active
containers were retained. This is image distribution, not a production rollout.

See [per-node verification and cleanup totals](validation/fleet-delivery.json).
The L2 STORE protection from #6171 is still excluded; L1 service tests do not validate it.
