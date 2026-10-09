"""Run the frozen CPU exporter with explicit, bounded supporting records."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


root = Path('/results')
helper_root = Path('/late-helpers')
reference_root = Path('/late-reference')
exporter = root / 'export_late_evidence.py'
sanitizer = root / 'public_log_redaction.py'
expected = {
    exporter: '39a14ef9be9cf6b16958baca1a946ecba0a6f78d5c102910f1fd3e6a7c717310',
    sanitizer: '4a9a2c0661996a566b7f75a2a508dee7a7aa839097907bbe0711769c2f3a9f0f',
}
for path, digest in expected.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
supports = sorted(path for path in (root / 'support-source').iterdir()
                  if path.is_file())
assert len(supports) == 26, 'All explicitly uploaded preparation/review/attempt records'
supports += sorted(path for path in helper_root.rglob('*.py')
                   if '__pycache__' not in path.parts)
supports += [reference_root / 'bundle-manifest.json',
             reference_root / 'reference-lock.json']
supports += sorted(path for path in (reference_root / 'reference').iterdir()
                   if path.is_file())
lock = json.loads((reference_root / 'reference-lock.json').read_text())
supports += [Path('/source') / name for name in lock['source_files']]
names = [
    'runtime-preflight.json', 'pre-export-idle.txt', 'matrix.log', 'matrix.exit',
    'pair-audit-exec-state.json', 'audit_after_pair_v3.py', 'audit_late_host_v3.py',
    'monitor_host.sh', 'preflight_runtime.py', 'replay-input-gates.json',
    'replay-supervisor-summary.json', 'replay-matrix.log', 'replay-matrix.exit',
]
for backend in ('triton', 'flashinfer'):
    prefix = 'replay-' + backend + '-1'
    names += [prefix + suffix for suffix in (
        '.log', '-host.exit', '-supervisor.json', '-monitor.log', '-monitor.exit',
        '-hostwatch.exit', '-hostwatch-host.csv', '-hostwatch-processes.log',
    )]
supports += [root / name for name in names]
audit_root = root / 'pair-audit-v1'
supports += [audit_root / 'host.json', audit_root / 'preflight.json']
for stage in ('native', 'host', 'snapshots'):
    supports += [audit_root / (stage + suffix) for suffix in (
        '-command.json', '-status.json', '.stdout', '.stderr',
    )]
assert all(path.is_file() for path in supports), 'All requested support files exist'
assert len({path.name for path in supports}) == len(supports), 'Unique support basenames'
argv = ['uv', 'run', '--offline', '--no-project', sys.executable, str(exporter),
        '--triton-run', '/results/late-triton-2',
        '--flashinfer-run', '/results/late-flashinfer-2',
        '--triton-replay', '/results/replay-triton-1.json',
        '--flashinfer-replay', '/results/replay-flashinfer-1.json',
        '--capture-audit', '/results/pair-audit-v1/snapshots.json',
        '--native-audit', '/results/pair-audit-v1/native.json',
        '--pair-summary', '/results/pair-audit-v1/summary.json',
        '--source-manifest', '/artifacts/performance-source.json',
        '--output-dir', '/results/public-late-v1',
        '--archive', '/results/public-late-v1.zip']
for path in supports:
    argv += ['--support-file', str(path)]
record = {'argv': argv, 'gpu_execution': False,
          'support_count': len(supports), 'frozen_exporter_sha256': expected[exporter],
          'frozen_sanitizer_sha256': expected[sanitizer]}
with (root / 'export-command.json').open('x') as handle:
    json.dump(record, handle, indent=2)
    handle.write('\n')
result = subprocess.run(argv, check=False, timeout=600)
with (root / 'export.exit').open('x') as handle:
    handle.write(str(result.returncode) + '\n')
raise SystemExit(result.returncode)
