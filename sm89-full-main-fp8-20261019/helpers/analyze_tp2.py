"""Aggregate fresh-process runs without treating requests as independent runs."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('/work/results/tp2'))
args = parser.parse_args()
root = args.root
assert json.loads((root / 'complete.json').read_text())['complete']
arms = ['base-a1', 'candidate-b1', 'candidate-b2', 'base-a2']
rows = []
quality = {}
source_hashes = {}
fixture_hashes = set()
model_hashes = set()
total = 0
for arm in arms:
    server = json.loads((root / f'{arm}-server.json').read_text())
    assert server['status'] == 'complete', server
    source_hashes[arm] = server['source_sha256']
    model_hashes.add(server['model_manifest_sha256'])
    data = json.loads((root / 'http' / f'{arm}.json').read_text())
    assert data['all_successful'] and len(data['rounds']) == 9
    fixture_hashes.add(data['fixture_sha256'])
    for concurrency in [1, 2, 8]:
        rounds = [r['summary'] for r in data['rounds'] if r['concurrency'] == concurrency]
        assert len(rounds) == 3
        assert all(r['requests'] == r['successful'] == 16 and r['failed'] == 0 for r in rounds)
        assert all(r['output_tokens'] == 2048 for r in rounds)
        total += sum(r['requests'] for r in rounds)
        rows.append({'arm': arm, 'concurrency': concurrency,
                     'throughput': statistics.median(r['output_tokens_per_second'] for r in rounds),
                     'tpot_p50_ms': statistics.median(r['tpot_ms']['p50'] for r in rounds),
                     'tpot_p95_ms': statistics.median(r['tpot_ms']['p95'] for r in rounds),
                     'ttft_p50_ms': statistics.median(r['ttft_ms']['p50'] for r in rounds),
                     'ttft_p95_ms': statistics.median(r['ttft_ms']['p95'] for r in rounds),
                     'request_p50_s': statistics.median(r['request_seconds']['p50'] for r in rounds),
                     'round_throughput': [r['output_tokens_per_second'] for r in rounds]})
    q = json.loads((root / f'{arm}-quality.json').read_text())
    quality[arm] = {k: q[k] for k in ['correct', 'truncated', 'scope']}
assert len(fixture_hashes) == len(model_hashes) == 1
assert source_hashes['base-a1'] == source_hashes['base-a2']
assert source_hashes['candidate-b1'] == source_hashes['candidate-b2']
aggregate = []
for c in [1, 2, 8]:
    row = {'concurrency': c}
    for label, prefix in [('base', 'base'), ('candidate', 'candidate')]:
        runs = [r for r in rows if r['concurrency'] == c and r['arm'].startswith(prefix)]
        assert len(runs) == 2
        row[label] = {key: statistics.median(r[key] for r in runs) for key in
                      ['throughput', 'tpot_p50_ms', 'tpot_p95_ms', 'ttft_p50_ms',
                       'ttft_p95_ms', 'request_p50_s']}
        row[label]['throughput_run_range'] = [min(r['throughput'] for r in runs), max(r['throughput'] for r in runs)]
    row['throughput_change_pct'] = 100 * (row['candidate']['throughput'] / row['base']['throughput'] - 1)
    row['tpot_reduction_pct'] = 100 * (1 - row['candidate']['tpot_p50_ms'] / row['base']['tpot_p50_ms'])
    aggregate.append(row)
texts = {arm: [r['response']['choices'][0]['message'].get('content') or ''
               for r in json.loads((root / f'{arm}-quality.json').read_text())['records']]
         for arm in arms}
identical = sum(len({texts[arm][i] for arm in arms}) == 1 for i in range(16))
token_comparisons = []
for base_arm, cand_arm in [('base-a1', 'candidate-b1'), ('base-a2', 'candidate-b2')]:
    base_data = json.loads((root / 'http' / f'{base_arm}.json').read_text())
    cand_data = json.loads((root / 'http' / f'{cand_arm}.json').read_text())
    def index_tokens(data):
        return {(r['concurrency'], r['repeat'], item['index']): item['output_token_ids']
                for r in data['rounds'] for item in r['records']}
    base_tokens, cand_tokens = index_tokens(base_data), index_tokens(cand_data)
    assert base_tokens.keys() == cand_tokens.keys()
    token_comparisons.append({'pair': [base_arm, cand_arm], 'requests': len(base_tokens),
        'exact_output_tokens': sum(base_tokens[k] == cand_tokens[k] for k in base_tokens)})
result = {'scope': 'FP8 27B TP2; complete main 3ddc6702a plus matching dd9e07a25 native artifacts',
          'measured_requests': total, 'failed': 0, 'fresh_process_order': arms,
          'model_manifest_sha256': next(iter(model_hashes)), 'source_hashes': source_hashes,
          'rows': rows, 'aggregate': aggregate, 'quality': quality,
          'quality_text_identical_all_four_runs': identical,
          'paired_measured_output_tokens': token_comparisons,
          'limits': ['16 arithmetic smoke questions are not a model accuracy benchmark',
                     'Two starts per arm do not establish statistical significance of small changes',
                     'The local model directory label is not an official model revision; the full file hash manifest identifies the checkpoint',
                     'TPOT is a client streaming estimate, not isolated kernel time or per-token ITL distribution']}
(root / 'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
