"""Check actual GDN dispatch and summarize separate short profiler traces."""

import gzip
import json
from collections import Counter
from pathlib import Path

root = Path('/work/results/tp2')
result = {}
for source in ('baseline', 'candidate'):
    arm = root / f'diagnostic-{source}'
    launch_files = sorted((arm / 'launches').glob('rank-*.jsonl'))
    assert len(launch_files) == 2, ('Missing GDN launch evidence', source)
    launches = []
    for file in launch_files:
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        assert any(0 < row['B'] <= 2 for row in rows), file
        for row in rows:
            assert (row['H'], row['HV'], row['K'], row['V']) == (8, 24, 128, 128), row
            assert row['qkv_dtype'] == 'torch.bfloat16', row
            assert row['state_dtype'] == 'torch.float32', row
            assert row['USE_QK_L2NORM_IN_KERNEL'], row
            expected = 4 if source == 'candidate' and row['B'] * row['HV'] <= 64 else 32
            assert row['BV'] == expected, row
        launches.extend(rows)
    profiles = []
    for path in sorted((arm / 'traces').rglob('*.json*')):
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as handle:
            data = json.load(handle)
        kernels = [event for event in data.get('traceEvents', [])
                   if event.get('cat') == 'kernel' and 'dur' in event]
        if not kernels:
            continue
        target = [event for event in kernels if
                  'fused_recurrent_gated_delta_rule_packed_decode_kernel' in event.get('name', '')]
        total = sum(event['dur'] for event in kernels)
        duration = sum(event['dur'] for event in target)
        profiles.append({'trace': str(path.relative_to(root)), 'kernels': len(kernels),
            'packed_gdn_kernels': len(target), 'all_kernel_us': total,
            'packed_gdn_us': duration, 'summed_kernel_duration_share_pct': 100 * duration / total,
            'packed_gdn_grids': dict(Counter(json.dumps(event.get('args', {}).get('grid'))
                                            for event in target))})
    assert profiles and all(row['packed_gdn_kernels'] > 0 for row in profiles), source
    backends = {path.name: json.loads(path.read_text())
                for path in (arm / 'launches').glob('*-fp8-backends.json')}
    assert len(backends) == 2 and all(backends.values()), source
    result[source] = {'launches': launches, 'profiles': profiles, 'fp8_backends': backends}
assert result['baseline']['fp8_backends'] == result['candidate']['fp8_backends']
result['limits'] = [
    'Diagnostic workers use launch observation and profiling; excluded from performance measurements',
    'Profiles contain prefill and 16-token decode; kernel-duration sums are not wall-clock or pure-decode shares',
    'Overlapping compute and communication durations are not an Amdahl speedup bound',
]
(root / 'diagnostic-summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps({key: {'launch_variants': len(value['launches']),
                      'profiles': len(value['profiles'])}
                  for key, value in result.items() if key != 'limits'}, indent=2))
