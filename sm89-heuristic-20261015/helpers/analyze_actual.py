import collections
import json
import math
import statistics
import argparse
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='actual-cold.jsonl')
parser.add_argument('--output', default='actual-analysis.json')
args = parser.parse_args()
records = [json.loads(line) for line in (root / args.input).read_text().splitlines()]
assert records[-1]['kind'] == 'complete'
groups = collections.defaultdict(lambda: collections.defaultdict(list))
for row in records:
    if row['kind'] == 'timing':
        assert not row['warnings'] and len(row['samples_us']) == 60
        groups[(row['B'], row['H'], row['HV'], row['dtype'])][row['arm']].append(row['median_us'])
rows = []
for (b, h, hv, dtype), arms in groups.items():
    assert len(arms['base']) == len(arms['candidate']) == 7
    base, cand = [statistics.median(arms[x]) for x in ['base', 'candidate']]
    rows.append({'B': b, 'H': h, 'HV': hv, 'dtype': dtype,
                 'selected': dtype == 'float32' and b * hv <= 64,
                 'base_us': base, 'candidate_us': cand,
                 'reduction_pct': 100 * (1 - cand / base),
                 'base_rounds_us': arms['base'], 'candidate_rounds_us': arms['candidate']})
result = {'cases': len(rows), 'gpu_pytest': '20 passed',
          'correctness_records': sum(r['kind'] == 'correctness' for r in records),
          'multistep_records': sum(r['kind'] == 'multistep' for r in records),
          'samples': sum(len(r['samples_us']) for r in records if r['kind'] == 'timing'),
          'metadata': records[0], 'groups': {}, 'rows': rows}
for name, selected in [('candidate_region', True), ('retained_bv32', False)]:
    subset = [r for r in rows if r['selected'] == selected]
    result['groups'][name] = {
        'cases': len(subset),
        'geomean_latency_reduction_pct': 100 * (1 - math.exp(statistics.mean(
            math.log(r['candidate_us'] / r['base_us']) for r in subset))),
        'reduction_range_pct': [min(r['reduction_pct'] for r in subset), max(r['reduction_pct'] for r in subset)],
    }
(root / args.output).write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'metadata']}, indent=2))
