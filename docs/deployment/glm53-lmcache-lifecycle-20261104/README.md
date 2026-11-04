# LMCache lifecycle integration for H20 / SM90

```bash
docker pull ghcr.io/positive666/model@sha256:e800a684027450cc58b1251c30a7c3357bc0876f51f11ae08ed3677a3059f930
```

Tag: `ghcr.io/positive666/model:glm53-febcc8ca7-lmcca439cad-h145b-lifecycle-cu130-sm90-20261104-rc1`. Use the immutable digest above for delivery.

This update rebuilds LMCache Python and native extensions against upstream `ca439cad3ea983ae8dc6197379070f6b71bfee61`, with the refreshed #6067/#6068 patches and #6171 STORE lifetime protection. vLLM `febcc8ca7`, Humming `0.1.19.post1`, Torch/CUDA, NCCL, DeepEP and existing vLLM/Qwen fixes remain pinned to the previous rc2 image. All 14 parent native files and 273 dependency versions were checked; only LMCache changes version. #56697 is absent.

## Prefer upstream fixes

- Official merged [#6256](https://github.com/LMCache/LMCache/pull/6256) supplies non-blocking ZMQ sends. It is already part of the base, rather than a duplicate local implementation.
- Official [#6252](https://github.com/LMCache/LMCache/pull/6252) supplies missing-registration cleanup behavior. The superseded immediate-release helper and its obsolete tests were removed during integration.
- #6067 (`553752cb10c7`), #6068 (`4a4a1bbcb99d`) and #6171 (`69fae0bdd707`) remain unmerged at this freeze and are explicitly retained.
- Eight subsequent upstream commits through `791a25b98313` were reviewed. They concern examples, lazy-offload tests, benchmark IPC, HF3FS tests, cache simulation, optional RawBlock I/O, CI and ROCm build checks; they do not replace these three lifecycle fixes. This release does not claim to track every later dev commit.

## What changed

| Area | Failure protected against | Integrated behavior |
|---|---|---|
| RETRIEVE / #6067 | TTL expiry or request cleanup releases a host buffer before device copy completion | Retain generation/owner-scoped non-expiring transfer pins until completion |
| Full-prefix retry / #6068 | A retry changes lookup boundaries and leaves stale reader reservations | Retire the original reservation using the original group windows and generation |
| L2 STORE / #6171 | A source buffer becomes evictable while an asynchronous store still reads it | Hold a separate pin for each adapter until success/failure completion; drain stores on shutdown |
| Existing LOAD, MQ and request lifecycle patches | Premature destination release, blocked shared polling, duplicate/stale completion | Preserve the existing protections and reconcile overlapping code with the refreshed PRs |

The integration removed duplicate methods and a duplicate dataclass field introduced by overlapping patches. Failed intermediate candidates were rejected before publication. The final installed-wheel tests below cover the reconciled code. The exact 18 LMCache runtime-file differences from the frozen upstream base are in [patches.json](patches.json), with full sources under `patches/lmcache/`.

## Validation

- 872 component cases passed. One accelerator test skipped in the CPU-only group was separately run and passed on the GPU; the original harness exit and coverage explanation are retained in the report.
- The old rc2 image fails both STORE/TTL regression variants with assertion failures, not setup errors. Both pass in this image. These use a real L1 manager and STORE controller with a controlled asynchronous MockL2 adapter; they are not a Mooncake network soak.
- DeepSeek TP8 + MP/L1: cold requests, CPU-cache restore, branches, concurrent requests, cancellation cleanup and model-restart reuse passed. See [service evidence](validation/service.json) for the exact counters.
- Eight-rank NCCL collectives passed with actual runtime 2.31.2. No default NUMA/NVLS/symmetric-memory disabling workarounds were added.
- Registry manifest/config identity, all blob presence/lengths, full OCI assembly, fresh-content-store import, and each target's complete CAS hashes and installed files were checked. The five target nodes received the image without restarting their existing services.

This is a patched integration image. These bounded checks do not establish that every workload is bug-free, do not measure a new throughput improvement, and do not replace long-running production observation or a real cross-node Mooncake/PD acceptance.

Build metadata uses an explicit custom release version; it does not use the source-only setuptools-scm pretend-version override. Generated protobuf bindings are recorded separately from tracked source hashes. Build records are provided for provenance, not as a self-contained build kit; the pinned parent image and hashed source/wheel inputs are also required.

AI assistance was used for integration, conflict resolution, validation and documentation. No upstream PR is marked ready by this release record.
