"""Generate review text after immutable TP2 raw data and all audits pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

HEAD = "d8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6"
ARMS = ("triton-a1", "flashinfer-b1", "flashinfer-b2", "triton-a2")
EVIDENCE_URL = (
    "https://github.com/positive666/vllm/tree/"
    "codex/gdn-flashinfer-evidence-20261007/tp2-followup-20261008"
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_archive(archive, manifest, raw):
    require(sha(archive) == manifest["sha256"], "Archive SHA256")
    require(archive.stat().st_size == manifest["bytes"], "Archive byte count")
    expected = {item["path"]: item for item in manifest["members"]}
    require(len(expected) == manifest["files"], "Unique archive member count")
    raw_files = {
        path.relative_to(raw).as_posix() for path in raw.rglob("*") if path.is_file()
    }
    require(raw_files == set(expected), "Complete immutable raw archive coverage")
    with zipfile.ZipFile(archive) as bundle:
        require(bundle.testzip() is None, "Archive CRC integrity")
        require(set(bundle.namelist()) == set(expected), "Archive member names")
        for name, item in expected.items():
            target = (raw / name).resolve()
            target.relative_to(raw.resolve())
            data = bundle.read(name)
            require(len(data) == item["bytes"], "Member size: " + name)
            require(
                hashlib.sha256(data).hexdigest() == item["sha256"],
                "Member hash: " + name,
            )
            require(target.read_bytes() == data, "Raw archive binding: " + name)


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(columns, rows):
    return "\n".join(
        ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        + ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    )


def load_logprob_diagnostics(root, raw):
    folder = raw / "logprob-diagnostics"
    audit_path = root / "audit-logprobs.json"
    if not folder.exists():
        require(not audit_path.exists(), "Missing raw logprob observation directory")
        return None
    require(audit_path.exists(), "Logprob observations require an independent audit")
    report = read(audit_path)
    require(report["status"] == "passed", "Independent logprob audit must pass")
    require(report["source_head"] == HEAD, "Logprob source head")
    require((folder / "run.exit").read_text().strip() == "0", "Logprob runner exit")
    require(
        report["comparison_sha256"] == sha(folder / "comparison.json"),
        "Logprob comparison raw binding",
    )
    for backend in ("triton", "flashinfer"):
        integrity = report["backends"][backend]["integrity"]
        for field, path in (
            ("raw_client_sha256", folder / f"{backend}.json"),
            ("server_record_sha256", folder / f"serve-{backend}.json"),
            ("server_log_sha256", folder / f"serve-{backend}.log"),
            ("primary_quality_sha256", raw / f"quality-{backend}.json"),
        ):
            require(
                integrity[field] == sha(path),
                "Logprob audit binding: " + backend + "/" + field,
            )
    return report


def serving_comparison_table(serving):
    if serving is None:
        return "TP2 performance numbers are pending the final independent audit."
    rows = []
    for c in (1, 8):
        values = serving["comparisons"][str(c)]
        cells = []
        for name in ("output_tokens_per_second", "ttft_p50_ms", "tpot_p50_ms"):
            value = values[name]
            cells.append(
                f"{value['triton_launch_medians']['median']:.3f}→"
                f"{value['flashinfer_launch_medians']['median']:.3f}"
                + (
                    f" ({value['flashinfer_improvement_percent']:+.2f}%)"
                    if name == "output_tokens_per_second"
                    else ""
                )
            )
        rows.append([c, *cells])
    return table(
        ["C", "Output tok/s T→FI (change)", "TTFT p50 ms T→FI", "TPOT p50 ms T→FI"],
        rows,
    )


def c8_tpot_summary(serving):
    if serving is None:
        return ""
    result = serving["comparisons"]["8"]["tpot_p50_ms"]
    delta = -result["flashinfer_improvement_percent"]
    direction = "slower" if delta > 0 else "faster"
    pairs = result["paired_improvement_percent"]
    mixed = min(pairs) < 0 < max(pairs)
    return f"FI C8 TPOT is **{abs(delta):.2f}% {direction}**; " + (
        "paired signs differ." if mixed else "paired effects remain descriptive."
    )


def logprob_text(report, archive):
    matching_answers = sum(
        row["triton_predicted"] is not None
        and row["triton_predicted"] == row["flashinfer_predicted"]
        for row in report["paired"]
    )
    matching_tokens = sum(
        row["first_divergence_zero_based"] is None
        and row["sequence_lengths"][0] == row["sequence_lengths"][1]
        for row in report["paired"]
    )
    divergences = [
        row["divergence"] for row in report["paired"] if row["divergence"] is not None
    ]
    reported_ties = sum(
        any(row[arm]["top2_logprob_margin"] == 0 for arm in ("triton", "flashinfer"))
        for row in divergences
    )
    sensitivity = (
        f"Returned top-two logprobs report zero margin on at least one arm in "
        f"{reported_ties}/{len(divergences)} first-divergence pairs. "
        "This is consistent with low-margin/tie sensitivity within these "
        "observations; it does not establish an epsilon-only difference or "
        "identify the underlying cause.\n\n"
        if reported_ties
        else ""
    )
    scores = table(
        [
            "Backend",
            "Requests",
            "Raw correct",
            "Strict correct",
            "Truncated",
            "No marker",
        ],
        [
            [
                arm,
                row["summary"]["requests"],
                row["summary"]["raw_correct"],
                row["summary"]["strict_correct"],
                row["summary"]["truncated"],
                row["summary"]["missing_marker"],
            ]
            for arm, row in report["backends"].items()
        ],
    )
    pairs = []
    for row in report["paired"]:
        margins = row["minimum_top2_logprob_margin"]
        divergence = row["divergence"]
        at_divergence = (
            f"{divergence['triton']['top2_logprob_margin']:.6g}/"
            f"{divergence['flashinfer']['top2_logprob_margin']:.6g}"
            if divergence is not None
            else "none"
        )
        pairs.append(
            [
                row["index"],
                row["matching_prefix_tokens"],
                row["first_divergence_zero_based"],
                row["common_incoming_prefix_positions_compared"],
                f"{margins['triton']:.6g}/{margins['flashinfer']:.6g}",
                at_divergence,
                row["triton_predicted"],
                row["flashinfer_predicted"],
            ]
        )
    paired = table(
        [
            "Question",
            "Matching prefix",
            "First divergent position (0-based)",
            "Positions compared",
            "Min top2 Δlogp T/FI",
            "At divergence T/FI",
            "T answer",
            "FI answer",
        ],
        pairs,
    )
    return f"""# Additional C8 top-five logprob observation

Reviewed source `{HEAD}` remains unchanged. One C8 batch of eight questions per
backend collects top-five logprobs, max3500/context4096/128 GPU blocks. This is
separate from both the original 100-question screen and expanded C1/C8 replay.
Their scores, wrong answers and FI C8 truncation remain unchanged and unresolved.

{scores}

Parsed answers match {matching_answers}/8; exact generated token sequences match
{matching_tokens}/8. These are additional observations, not replacement scores.

{paired}

{sensitivity}\
The actual incoming prompt tokens match. Comparison uses the shared generated
token prefix through and including the first divergent token, whose incoming
prefix still matches. Later positions have different conditioning histories and
are excluded. For a length-only difference, only overlapping positions are used.
Matching tokens do not establish identical internal hidden/recurrent states.

Top2 Δlogp is the difference between the highest two returned log probabilities,
in natural-log units; the full audit also retains probability margins and only
shared candidate-token differences. Top-five candidates cover part of the
vocabulary: these are not full-logit errors, KL divergence or full-vocabulary
probability distances. No margin threshold is used to declare a benign tie or
attribute any answer difference to this adapter.

Logprob collection can change scheduling and trajectories. This is one C8
observation per backend without forced decode or a full-state shadow reference.
It does not prove a root cause, accuracy equivalence, or resolution of the earlier
FI C8 limitation. Audit pass means observation integrity only.

[Original raw HTTP outputs and comparison in the archive]({archive.name}) are
under `logprob-diagnostics/`: `triton.json`, `flashinfer.json`, `comparison.json`
and both launch records/full logs. [Independent audit](audit-logprobs.json)
retains the final common-prefix window, chosen tokens, top-five candidates,
missing candidates and all unrounded reported margins.
"""


def diagnostic_limit(diagnostic):
    rounds = {
        backend: next(item for item in data["rounds"] if item["concurrency"] == 8)
        for backend, data in diagnostic["backends"].items()
    }
    triton = rounds["triton"]["summary"]
    fi = rounds["flashinfer"]["summary"]
    targets = {item["index"]: item for item in rounds["flashinfer"]["target_outcomes"]}
    outcomes = "; ".join(
        f"FI question {index}: parsed {row['predicted']}, "
        f"gold {row['gold']}, {row['finish_reason']}, {row['token_count']} tokens"
        for index, row in targets.items()
    )
    text = (
        f"Expanded-budget C8 diagnostic: FI **{fi['strict_correct']}/8** strict "
        f"correct versus Triton **{triton['strict_correct']}/8**, with "
        f"**{fi['truncated']} FI truncation(s)**. {outcomes}. "
    )
    if fi["truncated"] or fi["strict_correct"] < triton["strict_correct"]:
        text += (
            "The retained FI C8 long-generation/output differences are an "
            "unresolved limitation in this bounded diagnostic, despite the "
            "larger budget. "
        )
    return text + (
        "The default `auto` path remains Triton; FI remains opt-in. "
        "Integrity audit success does not establish accuracy equivalence "
        "or suitability as a general replacement."
    )


def diagnostic_text(diagnostic):
    rows = []
    targets = []
    for backend, data in diagnostic["backends"].items():
        for item in data["rounds"]:
            summary = item["summary"]
            rows.append(
                [
                    backend,
                    item["concurrency"],
                    summary["requests"],
                    summary["raw_correct"],
                    summary["strict_correct"],
                    str(
                        summary["strict_correct"]
                        - sum(
                            value["strict_correct"] for value in item["target_outcomes"]
                        )
                    )
                    + "/6",
                    summary["truncated"],
                    summary["unparsed"],
                ]
            )
            for value in item["target_outcomes"]:
                targets.append(
                    [
                        backend,
                        item["concurrency"],
                        value["index"],
                        value["gold"],
                        value["predicted"],
                        value["finish_reason"],
                        value["strict_correct"],
                        value["token_count"],
                    ]
                )
    totals = table(
        [
            "Backend",
            "C",
            "Requests",
            "Raw correct",
            "Strict correct",
            "Controls strict",
            "Truncated",
            "Unparsed",
        ],
        rows,
    )
    outcomes = table(
        [
            "Backend",
            "C",
            "Question",
            "Gold",
            "Parsed answer",
            "Finish",
            "Strict",
            "Tokens",
        ],
        targets,
    )
    return f"""## Separate expanded-budget diagnostic

{diagnostic_limit(diagnostic)}

The original 100-question records above remain unchanged, including truncations
and changed answers. A matched targeted diagnostic replays the original C8 batch
`[198, 206, 209, 228, 255, 285, 292, 318]`: questions 209/255 plus six controls,
once at C1 then once at C8 per backend, 16 requests per backend.
Its output budget is 3,500, context length 4,096 and GPU blocks 128, with
matched actual capacity >=32,768 tokens. Source/model/runtime/GPUs are unchanged.

{totals}

{outcomes}

`audit-diagnostics.json` independently retains the paired outcomes and flips.
This diagnostic changes output budget, model context and cache capacity together;
it cannot isolate a single cause, change the original 100-question score or prove
overall accuracy. Wrong and truncated outcomes are retained rather than discarded.
"""


def quality_text(quality, reference, diagnostic, logprobs):
    report = quality["model_quality"]
    backends, comparison = report["backends"], report["comparison"]
    rows = []
    for backend in ("triton", "flashinfer"):
        s = backends[backend]["summary"]
        rows.append(
            [
                backend,
                f"{s['correct']}/{s['samples']}",
                f"{s['completed_correct']}/{s['samples']}",
                f"{s['strict_correct']}/{s['samples']}",
                s["truncated"],
                s["unparsed"],
                s["missing_answer_marker"],
            ]
        )
    scores = table(
        [
            "Backend",
            "Raw correct",
            "Completed correct",
            "Strict correct",
            "Truncated",
            "Unparsed",
            "No ####",
        ],
        rows,
    )
    flips = []
    for metric, value in comparison["flips"].items():
        flips.append(
            f"- {metric}: Triton-to-FI losses "
            f"{value['triton_to_flashinfer_losses']}; gains "
            f"{value['triton_to_flashinfer_gains']}."
        )
    issues = []
    for row in comparison["all_changed_answer_or_exceptional_outcomes"]:
        values = []
        for backend in ("triton", "flashinfer"):
            entry = row[backend]
            values.append(
                f"{entry['predicted']} / {entry['finish_reason']} / "
                f"{entry['strict_correct']}"
            )
        issues.append([row["index"], row["gold"], *values])
    outcomes = (
        table(
            [
                "Question",
                "Gold",
                "Triton parsed/finish/strict",
                "FI parsed/finish/strict",
            ],
            issues,
        )
        if issues
        else "No changed answer or exceptional outcome was recorded."
    )
    prior = ""
    if reference is not None:
        summaries = reference["backends"]
        a = summaries["triton"]["summary"]["strict_correct"]
        b = summaries["flashinfer"]["summary"]["strict_correct"]
        prior = (
            f"The prior same-head [TP1 screen](../quality-followup-20261008) "
            f"reported {a}/100 and {b}/100 strict correct. TP1 and TP2 have different "
            "sharding and communication; a cross-TP score difference cannot be "
            "attributed to FI from these single runs.\n\n"
        )
    closeout = ""
    if quality.get("log_closeout_notes"):
        closeout = (
            "Log integrity retains a separately audited late suffix; original "
            "launch records and logs were not rewritten.\n\n"
            + "\n".join("- " + note for note in quality["log_closeout_notes"])
            + "\n\n"
        )
    additional = (
        "\n[Additional top-five logprob observation](additional-logprob-results.md) "
        "is separately scoped; it does not replace or resolve these retained "
        "quality outcomes.\n"
        if logprobs is not None
        else ""
    )
    return f"""# TP2 model quality and distributed correctness

Source `{HEAD}`, two L20 GPUs, BF16 inputs and FP32 state.
{diagnostic_limit(diagnostic)}

One 100-question GSM8K run per backend, the same deterministic test subset,
zero-shot, temperature 0, seed 42, thinking disabled, concurrency 8 and a
1,750-token output budget. Both source/fixture/protocol/raw-integrity audits pass.

{scores}

Strict correct requires a correct parsed answer, `finish_reason=stop`, no
truncation and a `####` answer marker. Equal counts do not erase changed answers.
Parsed answers match {comparison["matching_parsed_answers"]}/100; exact token
sequences match {comparison["exact_token_sequences"]}/100.

{chr(10).join(flips)}

{outcomes}

`audit-quality-correctness.json` retains all answer flips, exceptional output
text and the first token divergence for changed sequences.

{closeout}{prior}This bounded single run does not prove accuracy equivalence or accuracy
improvement. Do not infer causality from one changed output or a score difference.

Two NCCL ranks pass all {quality["correctness"]["passed_rank_cases"]} rank/cases
at local H8/HV24/K128/V128. Each rank checks B1/B8, int32/int64 indices and B8
padding against the one-step reference and 128 changing-input graph replays,
using pointwise atol/rtol 1e-2 and relative L2 <1%, with exact inactive state
and page-padding checks. NCCL initialization/reduction is checked separately.
This validates local wrapper execution on both ranks; checkpoint sharding and
full-model TP communication are exercised by the separate serving launches.
Correctness records are not TP2 microbenchmark timings.

{diagnostic_text(diagnostic)}
{additional}
"""


def serving_text(serving, protocol, archive, manifest):
    launch_rows = []
    for concurrency in (1, 8):
        for arm in ARMS:
            metrics = serving["launches"][arm]["metrics"][str(concurrency)]
            launch_rows.append(
                [
                    arm,
                    concurrency,
                    f"{metrics['output_tokens_per_second']['median']:.6f}",
                    f"{metrics['ttft_p50_ms']['median']:.6f}",
                    f"{metrics['tpot_p50_ms']['median']:.6f}",
                ]
            )
    comparisons = []
    for concurrency in (1, 8):
        for metric in ("output_tokens_per_second", "ttft_p50_ms", "tpot_p50_ms"):
            value = serving["comparisons"][str(concurrency)][metric]
            pairs = value["paired_improvement_percent"]
            comparisons.append(
                [
                    concurrency,
                    metric,
                    f"{value['flashinfer_improvement_percent']:+.4f}%",
                    f"{pairs[0]:+.4f}%",
                    f"{pairs[1]:+.4f}%",
                ]
            )
    placement = serving["host_placement_and_telemetry"]
    measured = placement["measured_clock_temperature"]
    clock_rows = []
    for device in protocol["gpu_uuids"]:
        records = [measured[arm][str(c)][device] for arm in ARMS for c in (1, 8)]
        clock_rows.append(
            [
                device,
                f"{min(r['sm_clock_mhz']['median'] for r in records):.0f}–"
                f"{max(r['sm_clock_mhz']['median'] for r in records):.0f}",
                f"{min(r['temperature_c']['min'] for r in records):.0f}–"
                f"{max(r['temperature_c']['max'] for r in records):.0f}",
            ]
        )
    audit_notes = "\n".join("- " + str(note) for note in serving["notes"])
    return f"""# Two-L20 TP2 serving performance

Source `{HEAD}`, local GDN shape H8/HV24/K128/V128 per rank.
Both backends use {protocol["gpu_blocks"]} GPU blocks and verified
{protocol["expected_kv_capacity_tokens"]:,} KV tokens, BF16 execution, FP32 SSM
state, Marlin and FlashAttention2, with prefix caching disabled.

ABBA launches Triton/FI/FI/Triton share one fixed 512-input/128-output-token
fixture. C1/C8 each have three rounds of 16 requests plus eight warmups per
concurrency. All {serving["counts"]["measured"]} measured and
{serving["counts"]["warmup"]} warmup requests complete the fixed token protocol
without client errors; startup/warmup are excluded from performance measurements.
Closed-loop workers and loopback HTTP are included in this serving boundary.

{table(["Launch", "C", "Output tokens/s", "TTFT p50 ms", "TPOT p50 ms"], launch_rows)}

Each launch value is the median of its three round metrics. Overall changes
compare medians of the two launch medians per backend; pair 1 is B1/A1 and
pair 2 B2/A2. Positive means higher FI throughput or lower FI TTFT/TPOT.
Negative results are retained; no observed-range significance rule is applied.

{table(["C", "Metric", "FI improvement", "Pair 1", "Pair 2"], comparisons)}

{c8_tpot_summary(serving)}

These are descriptive results from two independent launches per backend,
not statistically established gains, equivalence or a general serving speedup.
Requests within a launch are not independent process/hardware replications.

Five-second host snapshots attribute selected-GPU compute PIDs to the owned
container; sampled foreign/unattributed PID count is
{placement["sampled_foreign_pid_count"]}. Approximate measured windows use remote
original JSON mtimes minus measured wall duration, with write/serialization
uncertainty. Sampled per-launch/concurrency clock medians and temperature ranges:

{table(["GPU UUID", "Median SM clock range MHz", "Temperature range C"], clock_rows)}

Matching sampled clocks cannot exclude transient throttling between samples.
Topology/P2P outputs, full clock ranges and PID evidence remain in the archive.
No clock, fan or power settings were changed for this run.

No TP2 decode microbenchmark was run. Earlier TP1 GDNDecode + RMSNorm GPU
latencies use a different measurement boundary; they are not full-model latency
changes and cannot be transferred to TP2. See the earlier
[TP1 evidence](../performance-followup-20261008) for its recorded CUDA graph
events fallback, warm-state gains, unfavorable rotated-state B8 median and
synthetic HV4 negative results.

Native libraries are reused with recorded hashes. SM80 hardware, H20, BF16 state,
a fresh full CUDA13 build and broader workloads remain unvalidated.

An optional DeepEP import probe is unavailable: the caught
`AssertionError: Cannot find package: nccl` is retained in startup logs. The dense
TP2 serving path does not exercise DeepEP; the separate two-rank NCCL correctness
check passes. DP+EP remains unvalidated, and these results do not establish a
fully validated runtime environment. Independent startup/log classification notes:

{audit_notes}

`{archive.name}`: {manifest["files"]} immutable raw files, {manifest["bytes"]:,}
bytes; SHA256 `{manifest["sha256"]}`. All independent audits pass; their pass
status establishes evidence integrity, not favorable performance or quality.
"""


def readme_text(protocol, manifest, archive, diagnostic, logprobs):
    additional_link = (
        "- [Additional logprob observation](additional-logprob-results.md): "
        "shared-prefix top-five distributions through first divergence only.\n"
        if logprobs is not None
        else ""
    )
    additional_run = (
        "\nRun the independent `run_tp2_logprobs_host.sh` on the prepared host "
        "after ABBA, with the matching owned container/host identity. Preserve "
        "its separate PID/clock snapshots and run.exit; this launches two "
        "C8×8 top-five observations, never a performance round.\n\n"
        "```sh\n"
        "bash run_tp2_logprobs_host.sh OWNED_CONTAINER "
        "/results/tp2-followup-20261008\n"
        "```\n"
        if logprobs is not None
        else ""
    )
    additional_audit = (
        "uv run --no-project .venv/bin/python audit_tp2_logprobs.py \\\n"
        "  --results raw --api-source /path/to/unchanged/d8/source \\\n"
        "  --output audit-logprobs.json\n"
        if logprobs is not None
        else ""
    )
    return f"""# TP2 follow-up for PR #60403

Reviewed source `{HEAD}` is unchanged. Two L20 GPUs use TP2, global
H16/HV48/K128/V128 and local H8/HV24/K128/V128, BF16 inputs and FP32 state.

{diagnostic_limit(diagnostic)}

- [Serving results](serving-results.md): four independent ABBA launches,
  per-launch medians, paired effects and negative results.
- [Quality/correctness results](quality-results.md): two 100-question model
  screens, every answer flip, 12 distributed rank/cases and a separate matched
  expanded-budget diagnostic. Original scores/truncations are preserved.
- [Reviewer alignment](review-alignment.md): completed d8 feedback and limits.
{additional_link}

`protocol.json` freezes {protocol["gpu_blocks"]} blocks,
{protocol["expected_kv_capacity_tokens"]:,} KV tokens and the selected UUIDs after
both compatibility/quality launches succeed. Audits independently verify
source, runtime, model, fixture and logs and recompute HTTP/model results.
`{archive.name}` contains {manifest["files"]} original files with SHA256
`{manifest["sha256"]}`. `archive-manifest.json` hashes every member.

The served Qwen3.8-labelled FP8 checkpoint declares
Qwen3_5ForConditionalGeneration. This is text-only inference; native libraries
are reused. Local correctness, bounded model quality and serving measurements
have separate conclusions. No accuracy equivalence or universal speedup is claimed.
There is no TP2 microbenchmark; prior TP1 GPU timings remain separately scoped.
The optional DeepEP import probe is unavailable; DP+EP remains unvalidated.
No production source changes were made. AI assistance was used.

Reproduce only in a fresh authorized two-GPU experiment namespace with the
same isolated image/runtime/model. Old outputs intentionally cannot be overwritten.
Preserve initial idle-GPU/process/container identity, topology and five-second
host PID/clock snapshots separately using `host_monitor_tp2.sh` on the host.
The record retains those original snapshots; sampling has bounded coverage.

```sh
export PYTHONPATH=/source HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
root=/results/tp2-followup-20261008
uv run --offline --no-project /cache/gdn-runtime/bin/python \\
  /artifacts/runtime_probe.py --source-head {HEAD} \\
  --output "$root/runtime-probe.json"
uv run --offline --no-project /cache/gdn-runtime/bin/python \\
  -m torch.distributed.run --standalone --nproc-per-node=2 \\
  /artifacts/tp2_gdn_correctness.py \\
  --harness /reference-artifacts/benchmark_current_gdn.py \\
  --source-manifest /artifacts/performance-source.json \\
  --output-dir "$root/correctness"
bash /artifacts/run_tp2_preflight.sh
# Each preflight also invokes /reference-artifacts/quality_http.py:
# --dataset /oldcache/gsm8k-test.jsonl --samples 100 --concurrency 8
# --max-tokens 1750 --timeout 300 --server-evidence <own TP2 launch>
# Separate matched C1/C8 diagnostic; keeps the original 100-item screen unchanged.
uv run --offline --no-project /cache/gdn-runtime/bin/python \\
  /artifacts/tp2_diagnostic_supervisor.py --backend triton
uv run --offline --no-project /cache/gdn-runtime/bin/python \\
  /artifacts/tp2_diagnostic_supervisor.py --backend flashinfer
# Audit primary/diagnostic source, quality and capacity before freezing.
uv run --offline --no-project /cache/gdn-runtime/bin/python \\
  /artifacts/freeze_tp2_after_diagnostics.py
bash /artifacts/run_tp2_performance.sh
# run_tp2_performance.sh collects original remote round-file-times.json.
```

Preserve `distributed-correctness.exit`, `preflight.exit` and `pipeline.exit`
from the real executions. The immutable archive requires zero for each.
{additional_run}
Collect host telemetry before archiving. Package on the prepared runtime:

```sh
uv run --offline --no-project /cache/gdn-runtime/bin/python \\
  /artifacts/package_tp2.py archive --root "$root" \\
  --archive /results/tp2-evidence.zip
```

After transporting the archive, use a uv-managed interpreter to verify/extract
and independently audit the unchanged raw files. Retain the source manifest,
frozen correctness benchmark, helpers, original quality harness, model config
and full cached GSM8K test dataset with their hashes.

```sh
uv run --no-project .venv/bin/python package_tp2.py extract \\
  --root raw --archive tp2-evidence.zip --sha256 {manifest["sha256"]}
uv run --no-project .venv/bin/python audit_tp2_quality_correctness.py \\
  --results raw --source-manifest performance-source.json \\
  --correctness-harness benchmark_current_gdn.py \\
  --correctness-helper tp2_gdn_correctness.py --quality-harness quality_http.py \\
  --supervisor tp2_supervisor.py --dataset gsm8k-test.jsonl \\
  --model-config model-config.json --output audit-quality-correctness.json
uv run --no-project .venv/bin/python audit_tp2.py \\
  --results raw --source-manifest performance-source.json \\
  --output audit-tp2.json
uv run --no-project .venv/bin/python audit_tp2_diagnostics.py \\
  --results raw --output audit-diagnostics.json
{additional_audit}
uv run --no-project .venv/bin/python finalize_tp2_evidence.py \\
  --archive tp2-evidence.zip --output-dir ready
```

The archive includes all successful/failing statuses actually recorded; reports
are derived from immutable raw data. Approval/publication and GPU cleanup are
handled separately and are not implied by generated text.
"""


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=root)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evidence-url", default=EVIDENCE_URL)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    raw = (args.results or root / "raw").resolve()
    archive = (args.archive or root / "tp2-evidence.zip").resolve()
    output = (args.output_dir or root / "ready").resolve()
    manifest = read(root / "archive-manifest.json")
    serving = read(root / "audit-tp2.json")
    quality = read(root / "audit-quality-correctness.json")
    diagnostic = read(root / "audit-diagnostics.json")
    logprobs = load_logprob_diagnostics(root, raw)
    protocol = read(raw / "protocol.json")
    source = read(root / "performance-source.json")
    require(
        serving["validation_passed"] is True and not serving["errors"],
        "Serving/placement audit must pass without errors",
    )
    require(quality["status"] == "passed", "Quality/correctness audit must pass")
    require(
        diagnostic["status"] == "passed",
        "Separate diagnostic integrity audit must pass",
    )
    require(
        all(
            value["source_head"] == HEAD
            for value in (source, protocol, serving, quality, diagnostic)
        ),
        "All evidence must refer to the reviewed d8 source head",
    )
    require(protocol["tensor_parallel_size"] == 2, "Frozen TP2 protocol")
    require(quality["correctness"]["passed_rank_cases"] == 12, "12 rank/cases")
    require(quality["correctness"]["world_size"] == 2, "Two correctness ranks")
    require(
        serving["counts"]
        == {
            "measured": 384,
            "warmup": 64,
            "compatibility": 32,
        },
        "Frozen serving request totals",
    )
    require(serving["protocol_sha256"] == sha(raw / "protocol.json"), "Protocol hash")
    require(
        quality["source_manifest_sha256"] == sha(root / "performance-source.json"),
        "Quality audit source-manifest binding",
    )
    for name in ("pipeline.exit", "preflight.exit", "distributed-correctness.exit"):
        require((raw / name).read_text().strip() == "0", "Run exit status: " + name)
    verify_archive(archive, manifest, raw)
    for arm in ARMS:
        require(
            serving["launches"][arm]["client_sha256"]
            == sha(raw / "http" / f"{arm}.json"),
            "Serving audit raw binding: " + arm,
        )
        require(
            serving["launches"][arm]["server_sha256"] == sha(raw / f"serve-{arm}.json"),
            "Serving audit launch binding: " + arm,
        )
    for backend in ("triton", "flashinfer"):
        require(
            quality["model_quality"]["backends"][backend]["integrity"][
                "raw_quality_sha256"
            ]
            == sha(raw / f"quality-{backend}.json"),
            "Quality audit raw binding: " + backend,
        )
        integrity = diagnostic["backends"][backend]["integrity"]
        for field, path in (
            ("raw_client_sha256", raw / "diagnostics" / f"{backend}.json"),
            ("server_record_sha256", raw / "diagnostics" / f"serve-{backend}.json"),
            ("server_log_sha256", raw / "diagnostics" / f"serve-{backend}.log"),
            ("primary_quality_sha256", raw / f"quality-{backend}.json"),
        ):
            require(
                integrity[field] == sha(path),
                "Diagnostic audit raw binding: " + backend + "/" + field,
            )
    prior_path = root.parent / "quality-followup-20261008/audit.json"
    reference = None
    if prior_path.exists():
        prior = read(prior_path)
        if (
            prior.get("source_head") == HEAD
            and prior.get("status") == "integrity_passed"
            and prior["dataset_sha256"] == quality["model_quality"]["dataset_sha256"]
            and prior["question_fixture_sha256"]
            == quality["model_quality"]["question_fixture_sha256"]
            and prior["protocol"] == quality["model_quality"]["protocol"]
        ):
            reference = prior
    q = quality["model_quality"]
    a = q["backends"]["triton"]["summary"]
    b = q["backends"]["flashinfer"]["summary"]
    improvements = [
        serving["comparisons"][str(c)]["output_tokens_per_second"][
            "flashinfer_improvement_percent"
        ]
        for c in (1, 8)
    ]
    expanded_rounds = {
        arm: {row["concurrency"]: row["summary"] for row in data["rounds"]}
        for arm, data in diagnostic["backends"].items()
    }
    t, f = expanded_rounds["triton"], expanded_rounds["flashinfer"]
    extra_draft = (
        " [Additional top-five observation](additional-logprob-results.md) "
        "uses shared prefixes through first divergence; margins are not causal proof."
        if logprobs is not None
        else ""
    )
    draft = (
        f"Unchanged `d8ae9e9`, two-L20 TP2, 512/128-token ABBA "
        f"([evidence]({args.evidence_url})):\n\n"
        f"{serving_comparison_table(serving)}\n\n"
        f"{c8_tpot_summary(serving)} "
        "All **384 measured +64 warmups** meet the token protocol. Two launches "
        "per backend: descriptive results, no general speedup. Both NCCL ranks "
        "pass **12 rank/cases** at H8/HV24, including 128 changing-input steps.\n\n"
        f"Original GSM8K strict: **T {a['strict_correct']}/100, "
        f"FI {b['strict_correct']}/100**; "
        f"FI truncation/absent-marker **{b['truncated']}/"
        f"{b['missing_answer_marker']}**. Expanded strict: "
        f"C1 **T {t[1]['strict_correct']}/8 / FI {f[1]['strict_correct']}/8**, "
        f"C8 **T {t[8]['strict_correct']}/8 / FI {f[8]['strict_correct']}/8**; "
        f"FI C8 truncations **{f[8]['truncated']}**. "
        "**FI C8 output/truncation differences remain unresolved.** "
        "[Detailed quality tables](quality-results.md) retain all negative outcomes. "
        "Integrity audits do not establish accuracy equivalence; FI remains opt-in, "
        "`auto` Triton. No TP2 microbenchmark; prior TP1 GPU timings keep their "
        "separate boundary." + extra_draft + "\n"
    )
    alignment = """# Reviewer alignment on the unchanged d8 implementation

The earlier [reply](https://github.com/vllm-project/vllm/pull/60403#issuecomment-6055210576)
still describes these changes as pending at c3424cc. The reviewed d8 source
completes its five items; update that reply with completed actions and evidence.

1. Removed HV%8 rejection. Packed HV4 gates are cloned when needed for pointer
   alignment, including the B1 offset-view case. Existing synthetic small-head
   timings retain their unfavorable results; TP2 H8/HV24 is separately validated.
   CUDA eligibility is SM80+; BF16 inputs, FP32 state and K=V=128 remain the
   adapter scope, not a claim of universal FI requirements. SM80 was not run.
2. Removed signature/API probing; the lazy direct import uses the pinned FI API
   with explicit `backend="flashinfer"`. Configuration documentation is shorter.
3. FI-only FP32 bias is an adapter specialization choice, not a claim that FI
   rejects BF16 bias, that FP32 is optimal or that it fixes external shared caches.
4. Split configuration errors and added once-only logging for explicit FI
   overriding the disabled packed flag. `.detach()` remains because the default
   DLPack exporter rejects grad-enabled model Parameters; this is retained with
   evidence rather than silently claiming the reviewer-requested deletion.
5. Mixed batches copy only the prefill tail after state updates and preserve decode
   prefix, removing the redundant concatenation/copy.

The new evidence does not change these source decisions. It adds real TP2
execution, rank-local correctness, a bounded model-quality screen and full
serving measurements. Maintenance review/acceptance and official CI remain
separate; generated reports do not assert that either is complete.
The FI C8 expanded-budget diagnostic retains long-generation truncation and
changed target answers. That limitation remains unresolved; keep FI opt-in and
the default Triton path. Audit pass status is not an accuracy or ready-to-merge claim.

For an earlier reply that promised follow-up work, replace future-tense claims
with the completed evidence link and actual source head/results. Preserve the
limits and answer flips. Avoid another pending-work comment or claims that
short successful requests prove model accuracy.
"""
    docs = {
        "README.md": readme_text(protocol, manifest, archive, diagnostic, logprobs),
        "serving-results.md": serving_text(serving, protocol, archive, manifest),
        "quality-results.md": quality_text(quality, reference, diagnostic, logprobs),
        "review-update-draft.md": draft,
        "review-alignment.md": alignment,
    }
    if logprobs is not None:
        docs["additional-logprob-results.md"] = logprob_text(logprobs, archive)
    require(not output.exists(), "Refusing to overwrite generated output")
    output.mkdir(parents=True)
    for name, contents in docs.items():
        (output / name).write_text(contents, encoding="utf-8", newline="\n")
    summary = {
        "source_head": HEAD,
        "archive": {"file": archive.name, "sha256": manifest["sha256"]},
        "audit_tp2_sha256": sha(root / "audit-tp2.json"),
        "audit_quality_correctness_sha256": sha(
            root / "audit-quality-correctness.json"
        ),
        "audit_diagnostics_sha256": sha(root / "audit-diagnostics.json"),
        "audit_logprobs_sha256": (
            sha(root / "audit-logprobs.json") if logprobs is not None else None
        ),
        "generator_sha256": sha(Path(__file__)),
        "derived_files": {name: sha(output / name) for name in docs},
        "quality_strict_correct": {
            "triton": a["strict_correct"],
            "flashinfer": b["strict_correct"],
        },
        "serving_throughput_improvement_percent": improvements,
        "statistical_scope": "Descriptive only; no significance/equivalence claim",
    }
    (output / "generated-manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "files": list(docs), "source_head": HEAD}))


if __name__ == "__main__":
    main()
