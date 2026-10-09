# Pinned GDN source review

Scope: read-only review of production head d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6 and downloaded FlashInfer runtime source. No production edits, GPU execution, precision changes, tolerance changes, or PR writes were performed by this review.

## Source binding

- FlashInfer gdn_decode.py SHA256: e5a5a796613a48f5750bed2e7cd4f9917a9580374ce3fa6bc0ea2aa47db3d280, exactly the prior runtime binding.
- Imported pretranspose implementation copy SHA256: 0547298823f4cdc1434c079749fe62a1bcafbd5aae3f129321431f414751b84d.
- d8 packed Triton fused_recurrent.py SHA256: fe6f1311014809040497aa0a623e7fa97f4b3457cdc7ee1f1f8aed21f326f755.
- d8 Qwen integration SHA256: d4056cb0105ca7c13fb3fa41be8f8e2ba3a2ea4f1ac81115abe3f3f4e6b3b45d.
- d8 weight_utils.py SHA256: 15eeb78befb71c75240f5b20c762381905a8f54f83f252de914330b1dddb7dfe.

## Actual kernel comparison

The d8 adapter explicitly passes backend='flashinfer'. Its FP32-state, single-token pool route uses run_pretranspose_decode, which selects the small_batch implementation. Cake and BF16-state paths are not the actual route.

The packed Triton kernel loads a, b, A_log and dt_bias separately into FP32. It computes x = a + dt_bias after those casts; b is used only for sigmoid. FlashInfer also converts both a and bias to Float32 before their sum. An exact promotion of BF16 checkpoint bias therefore does not itself change the gate input values or force a BF16 intermediate addition. This statement is about actual packed Triton, not its separate generic fused_sigmoid_gating implementation.

Both implement FP32 state decay, state-key projection, beta-scaled correction, outer-product update, and state-query output, then BF16 output storage. Observable source differences worth classifying are:

- Packed Triton normalizes q/k by division through sqrt(sum(x*x)+1e-6). FlashInfer multiplies by fast rsqrt(sum(x*x)+1e-6).
- FlashInfer first sums each lane's four adjacent K entries, then butterfly-reduces offsets 16/8/4/2/1; Triton uses tl.sum. Generated reduction association and FMA must not be inferred as bitwise identical from the shared formula.
- FlashInfer uses cute.exp/log/rsqrt with fastmath=True. Triton gate uses tl.exp/log and sigmoid; its decay exp comes from .op.exp, selected by FLA_USE_FAST_OPS. The current review has not bound that runtime environment value or inspected generated PTX.

These are numerical-strategy differences, not proof of a bug. Existing exact own-production replay and unchanged frozen-FP64 tolerance results already pass. Token identity is not an additional correctness threshold; no slow/precision branch is recommended solely to force identical sampled tokens.

## Interface and state handling

FlashInfer's compiled cache key contains q.dtype and shape/stride/scalar options but omits dt_bias.dtype, despite compiling the bias argument from its actual DLPack dtype. A mixed BF16/F32 bias caller with otherwise identical specialization can therefore reuse an incompatible compiled signature. This is a concrete upstream interface-cache candidate; it is not an explanation for the fresh isolated production FI arm, whose bias always has dtype FP32. Any dtype ablation must use separate process and compilation workspace. A minimal cache reproducer should independently demonstrate each dtype succeeds fresh, then retain the failure when the same process switches dtype without changing the other key fields.

The actual small_batch kernel widens pool indices before slot-stride multiplication. Negative read indices produce zero output without entering state writes. Default write indices equal read indices; the eight CTAs partition V rows and do not share writes to the same active row. d8 remaps its reserved null slot 0 and negative indices to FI -1 using stable metadata buffers, including graph metadata updates. Existing indexed/graph guards provide coverage, but the natural eight-prompt eager run has no actual padding rows. No new alias/null/graph defect was identified within the observed production layout. This is not a proof of every original backing-pool alias or graph schedule.

## Checkpoint precision policy

The FI constructor creates dt_bias directly in FP32; baseline auto/Triton uses the default BF16 model parameter dtype. The real sharded loader narrows the rank shard and default_weight_loader copies directly into the chosen parameter dtype. Consequently, an FP32 checkpoint that contains values not representable in BF16 is expected to retain those values under FI but round under baseline. The existing constructor/load fixture explicitly covers a BF16 checkpoint only.

This expected load precision policy difference must not automatically be called an integration bug. An explicit FI specialization can reasonably choose higher precision. Preserving baseline rounding is necessary only if the intended contract requires a decode-backend switch to preserve checkpoint-to-model-dtype rounding and shared prefill/speculative parameter semantics. The actual BF16 checkpoint has exact promoted bias values, so the FP32-checkpoint case does not explain its divergence. The contradictory ten/twenty-stalls q255 question also cannot by itself adjudicate kernel quality.

If preserving baseline parameter semantics becomes an explicit requirement, the smallest principled design is to keep the default parameter dtype and create/refresh a nonpersistent FP32 decode buffer after loading, used only by the FI leaf. GDN inherits AttentionLayerBase; the existing is_deferred_attention_layer predicate selects such a layer when a callable process_weights_after_loading method is present. Standard loading and layerwise reload call that hook. Adding an override would also newly classify GDN as deferred attention, so dummy/IPC/reload behavior, buffer device placement, and stable graph tensor identity need deliberate validation. Do not add an untested hook, per-call allocation, or stale lazy cache now.

## CPU diagnostic status

bias_load_cpu_probe.py reuses the existing Qwen constructor fixture and constructor-installed real TP-sharded loader for BF16 and nonrepresentable FP32 source tensors. Only the existing admission/TP/FI-import fixture surfaces are mocked; the constructor and loader are not replaced. It verifies imported paths, pinned constructor/loader bytes, actual parameter dtype and values, CPU placement, and that CUDA was not initialized.

- Original Windows attempt: actual exit 1 before constructor execution, due to Torch 2.13 template decoding under GBK. Log/exit retained.
- UTF8 Windows attempt: bounded 60-second uv subprocess timeout, SIGTERM/code=null, no output JSON and no constructor result. Timeout is not a contract failure.
- Final exact command-line CIM check: actual exit 0, owned_count=0. No unrelated Python process was stopped.
- Current portable helper: stdlib py_compile actual exit 0; SHA256 3a37c02a6ae6807d86ce2471d249b452e0dc974e1f66c2e16ffae2fe5f0b6ad5. This is syntax readiness, not a passed constructor diagnostic.

CPU-only container CLI, after mounting this one script:

    uv run --offline --no-project /cache/gdn-runtime/bin/python -X utf8 /source-review/bias_load_cpu_probe.py --repo /source --source-copy --output /results/bias-load-cpu-probe.json

The --source-copy mode explicitly checks the two exact source-file SHAs without claiming a Git head was independently verified. The expected d8 binding and actual git verification field are separate. The script hides GPUs before importing Torch/vLLM. Do not describe the expected counterexample as an actual test result until this command completes and its report is reviewed.

## Next bounded classification

The planned A native Triton / B FI initialization with only the packed-Triton decode leaf / C native FI experiment, stopped after 20 outputs, is a useful classifier if effective convolution, attention KV and active GDN initial states are compared first. B should retain FI constructor/config and metadata routing, with the changed leaf recorded explicitly. If A/B agree and C diverges while joins match, that supports a decode numerical-strategy explanation; it does not establish a defective kernel or a model-accuracy regression. If A/B differ, inspect the actual differing initializer/input instead of blaming bias dtype, whose current BF16 values are equal. Keep frozen local math thresholds unchanged.

## Host-script review boundary

The reviewed create/run/monitor scripts select the exact two GPU UUIDs, refuse pre-existing names/CID-file replacement, gate idleness and available disk, mount source/runtime/addon/reference read-only, and provide new per-arm workspaces. They bound each arm to 1200 seconds plus 30-second kill grace and monitoring to 3900 seconds. Cleanup gates container ID and owner label and preserves the original matrix exit code. Cleanup success still requires its actual receipts; a matrix success does not imply all cleanup commands succeeded under set +e. Consumers must retain monitor failure/repeated-unmatched evidence. Rechecking ID/owner immediately before a monitor-triggered stop would narrow its small check-to-use window, but no unrelated process deletion or stop was authorized by these scripts.
