"""Pinned full-main FP8 comparison; timing and diagnostic runs are separate."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

import http_performance_client as client
from frozen_quality_http import FIXED, answer_number

ROOT = Path('/work/results/tp2')
ROOT.mkdir(exist_ok=True)
COMMON = ['uv', 'run', '--offline', '--no-project', sys.executable]
MODEL = 'qwen27b-fp8-pinned'
BASE = 'http://127.0.0.1:8000'
REL = 'vllm/third_party/flash_linear_attention/ops/fused_recurrent.py'
EXPECTED = {'baseline': 'b065f160e94371192ce743bcd1ce3a9c75ddcecf31be243af4d55b39a898685e',
            'candidate': 'bd51c3421cfb7ecb7d8d8b5eafd6536d9023f4269f6e8b6d8705ae457fd69a36'}


def save(path, data):
    path.write_text(json.dumps(data, indent=2))


def machine_snapshot():
    fields = ('timestamp,index,uuid,pstate,memory.used,utilization.gpu,'
              'clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu')
    return {'gpu': subprocess.check_output([
        'nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader'
    ], text=True), 'loadavg': Path('/proc/loadavg').read_text().strip()}


def check_model():
    manifest = json.loads(Path('/work/results/model-manifest.json').read_text())
    for entry in manifest['files']:
        path = Path('/model') / entry['name']
        stat = path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (entry['size'], entry['mtime_ns'])
        if path.suffix != '.safetensors':
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    return hashlib.sha256(Path('/work/results/model-manifest.json').read_bytes()).hexdigest()


def quality(arm):
    records = []
    for index, (question, gold) in enumerate(FIXED):
        response = client.post_json(BASE, '/v1/chat/completions', {
            'model': MODEL, 'messages': [{'role': 'user', 'content':
                'Answer concisely. End with #### followed by the final number.\n' + question}],
            'chat_template_kwargs': {'enable_thinking': False},
            'temperature': 0, 'seed': 42, 'max_tokens': 256,
        }, 120)
        choice = response['choices'][0]
        text = choice['message'].get('content') or ''
        records.append({'index': index, 'question': question, 'gold': gold,
                        'parsed': answer_number(text), 'correct': answer_number(text) == gold,
                        'response': response})
    save(ROOT / f'{arm}-quality.json', {
        'scope': '16 arithmetic smoke questions; not an accuracy benchmark',
        'correct': sum(r['correct'] for r in records),
        'truncated': sum(r['response']['choices'][0]['finish_reason'] == 'length' for r in records),
        'records': records,
    })
    if not all(r['correct'] for r in records):
        raise RuntimeError('Arithmetic smoke check failed; inspect recorded responses')
    if any(r['response']['choices'][0]['finish_reason'] == 'length' for r in records):
        raise RuntimeError('Arithmetic smoke check was truncated')


def profile(arm, fixture):
    """Short diagnostic traces; request latencies are not performance results."""
    tokens = json.loads(fixture.read_text())['input_tokens']

    def request(_):
        return client.post_json(BASE, '/v1/completions', {
            'model': MODEL, 'prompt': tokens, 'temperature': 0, 'seed': 42,
            'max_tokens': 16, 'min_tokens': 16, 'ignore_eos': True,
        }, 180)

    def control(route, payload):
        req = urllib.request.Request(BASE + route, method='POST',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with client.HTTP_OPENER.open(req, timeout=180) as response:
            assert response.status == 200
            response.read()

    for concurrency in (1, 2, 8):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(request, range(concurrency * 2)))
            control('/start_profile', {'profile_prefix': f'{arm}-c{concurrency}'})
            try:
                responses = list(pool.map(request, range(concurrency * 2)))
            finally:
                control('/stop_profile', {})
        save(ROOT / f'{arm}-c{concurrency}-profile-requests.json', responses)


def run(arm, source, *, smoke=False, diagnostic=False):
    applications = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'
    ], text=True).strip()
    if applications:
        raise RuntimeError(f'GPU already in use before {arm}: {applications}')
    root = Path('/work/source') / source
    sha = hashlib.sha256((root / REL).read_bytes()).hexdigest()
    assert sha == EXPECTED[source]
    model_sha = check_model()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               LD_LIBRARY_PATH='/work/compat13/usr/local/cuda-13.0/compat',
               VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE='1',
               VLLM_WORKER_MULTIPROC_METHOD='spawn',
               VLLM_CACHE_ROOT=f'/work/cache/tp2-{arm}/vllm',
               TRITON_CACHE_DIR=f'/work/cache/tp2-{arm}/triton',
               TORCHINDUCTOR_CACHE_DIR=f'/work/cache/tp2-{arm}/inductor',
               CUDA_CACHE_PATH=f'/work/cache/tp2-{arm}/cuda')
    command = COMMON + ['-m', 'vllm.entrypoints.openai.api_server',
        '--model', '/model', '--served-model-name', MODEL, '--host', '127.0.0.1',
        '--port', '8000', '--language-model-only', '--dtype', 'bfloat16',
        '--quantization', 'fp8', '--linear-backend', 'auto',
        '--tensor-parallel-size', '2', '--max-model-len', '2048',
        '--max-num-seqs', '8', '--max-num-batched-tokens', '1024',
        '--gpu-memory-utilization', '0.85', '--no-enable-prefix-caching',
        '--mamba-ssm-cache-dtype', 'float32', '--seed', '42',
        '--attention-config', json.dumps({'backend': 'FLASH_ATTN', 'flash_attn_version': 2}),
        '--additional-config', json.dumps({'gdn_prefill_backend': 'triton'})]
    if diagnostic:
        env['PYTHONPATH'] = str(root) + ':/work/helpers'
        env['GDN_DIAGNOSTIC_DIR'] = str(ROOT / arm / 'launches')
        command += ['--worker-extension-cls', 'gdn_diagnostic.WorkerExtension',
            '--profiler-config', json.dumps({
                'profiler': 'torch', 'torch_profiler_dir': str(ROOT / arm / 'traces'),
                'torch_profiler_with_stack': False, 'ignore_frontend': True})]
    evidence_path = ROOT / f'{arm}-server.json'
    assert not evidence_path.exists(), 'Refuse to overwrite an earlier run'
    evidence = {'arm': arm, 'command': command, 'source_sha256': sha,
                'scope': 'full main 3ddc6702 + identical dd9e07a25 native artifacts; FP8 TP2',
                'model_manifest_sha256': model_sha,
                'environment': {k: env[k] for k in ('PYTHONPATH', 'LD_LIBRARY_PATH',
                    'VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE', 'TRITON_CACHE_DIR')},
                'status': 'starting', 'started_epoch': time.time(), 'smoke_only': smoke,
                'diagnostic_only': diagnostic, 'machine_before_start': machine_snapshot()}
    save(evidence_path, evidence)
    with (ROOT / f'{arm}-server.log').open('x') as log:
        server = subprocess.Popen(command, env=env, cwd=root, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        evidence['pid'] = server.pid
        save(evidence_path, evidence)
        try:
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited: {server.returncode}')
                try:
                    with client.HTTP_OPENER.open(BASE + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                time.sleep(3)
            else:
                raise TimeoutError('Readiness exceeded 1200 seconds')
            evidence['status'] = 'ready'
            evidence['machine_ready'] = machine_snapshot()
            save(evidence_path, evidence)
            fixture = ROOT / 'fixture.json'
            common = ['--base-url', BASE, '--model', MODEL, '--fixture', str(fixture)]
            if not fixture.exists():
                subprocess.run(COMMON + ['/work/helpers/http_performance_client.py', 'fixture'] + common,
                               check=True, timeout=120)
            if diagnostic:
                profile(arm, fixture)
            elif not smoke:
                subprocess.run(COMMON + ['/work/helpers/http_performance_client.py', 'bench'] + common + [
                    '--arm', arm, '--output-dir', str(ROOT / 'http'), '--requests', '16',
                    '--server-evidence', str(evidence_path)], check=True, timeout=2400)
            quality(arm)
            assert check_model() == model_sha
            evidence['machine_after_requests'] = machine_snapshot()
            evidence['status'] = 'complete'
        except Exception as exc:
            evidence.update(status='failed', error=repr(exc))
            raise
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
            evidence.update(finished_epoch=time.time(), server_exit=server.returncode)
            save(evidence_path, evidence)
            time.sleep(5)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError(f'Terminated by signal {signum}; stopping owned server')

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    if args.diagnostic:
        run('diagnostic-baseline', 'baseline', diagnostic=True)
        run('diagnostic-candidate', 'candidate', diagnostic=True)
    elif args.smoke:
        run('smoke-baseline', 'baseline', smoke=True)
        run('smoke-candidate', 'candidate', smoke=True)
    else:
        for arm, source in [('base-a1', 'baseline'), ('candidate-b1', 'candidate'),
                            ('candidate-b2', 'candidate'), ('base-a2', 'baseline')]:
            run(arm, source)
        save(ROOT / 'complete.json', {'complete': True, 'order': ['A', 'B', 'B', 'A']})
