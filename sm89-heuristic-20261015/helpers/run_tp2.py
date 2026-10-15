"""Supplemental BF16 TP2 A/B/B/A; isolated container, recorded image runtime.

This exercises the exact latest packed-decode module with vLLM 0.27.1 native
dependencies. It does not validate a full latest-main build or the FP8 model.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

import http_performance_client as client
from frozen_quality_http import FIXED, answer_number

ROOT = Path('/results/tp2')
ROOT.mkdir(exist_ok=True)
TARGET = Path('/opt/venv/lib/python3.12/site-packages/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py')
ORIGINAL = Path('/cache/image-original-fused_recurrent.py')
REL = 'vllm/third_party/flash_linear_attention/ops/fused_recurrent.py'
COMMON = ['uv', 'run', '--offline', '--no-project', sys.executable]
MODEL = 'qwen27b-bf16'
BASE = 'http://127.0.0.1:8000'


def save(path, data):
    path.write_text(json.dumps(data, indent=2))


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


def run(arm, root):
    source = Path(root) / REL
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = ('b065f160e94371192ce743bcd1ce3a9c75ddcecf31be243af4d55b39a898685e'
                if root == '/baseline' else
                'bd51c3421cfb7ecb7d8d8b5eafd6536d9023f4269f6e8b6d8705ae457fd69a36')
    assert sha == expected
    shutil.copyfile(source, TARGET)  # The writable layer of our dedicated container only.
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == sha
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE='1',
               VLLM_WORKER_MULTIPROC_METHOD='spawn',
               VLLM_CACHE_ROOT=f'/cache/tp2-{arm}/vllm',
               TRITON_CACHE_DIR=f'/cache/tp2-{arm}/triton',
               TORCHINDUCTOR_CACHE_DIR=f'/cache/tp2-{arm}/inductor',
               CUDA_CACHE_PATH=f'/cache/tp2-{arm}/cuda')
    command = COMMON + ['-m', 'vllm.entrypoints.openai.api_server',
        '--model', '/model', '--served-model-name', MODEL, '--host', '127.0.0.1',
        '--port', '8000', '--language-model-only', '--dtype', 'bfloat16',
        '--tensor-parallel-size', '2', '--max-model-len', '2048',
        '--max-num-seqs', '8', '--max-num-batched-tokens', '1024',
        '--gpu-memory-utilization', '0.85', '--no-enable-prefix-caching',
        '--mamba-ssm-cache-dtype', 'float32', '--seed', '42',
        '--attention-config', json.dumps({'backend': 'FLASH_ATTN', 'flash_attn_version': 2}),
        '--additional-config', json.dumps({'gdn_prefill_backend': 'triton'})]
    evidence_path = ROOT / f'{arm}-server.json'
    evidence = {'arm': arm, 'command': command, 'source_sha256': sha,
                'scope': 'BF16 TP2 supplemental, vLLM 0.27.1 image + exact upstream/candidate decode module',
                'model_config_sha256': hashlib.sha256(Path('/model/config.json').read_bytes()).hexdigest(),
                'state_dtype': 'float32', 'status': 'starting', 'started_epoch': time.time()}
    save(evidence_path, evidence)
    with (ROOT / f'{arm}-server.log').open('x') as log:
        server = subprocess.Popen(command, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        evidence['pid'] = server.pid
        save(evidence_path, evidence)
        try:
            deadline = time.monotonic() + 900
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
                raise TimeoutError('Readiness exceeded 900 seconds')
            evidence['status'] = 'ready'
            save(evidence_path, evidence)
            fixture = ROOT / 'fixture.json'
            common = ['--base-url', BASE, '--model', MODEL, '--fixture', str(fixture)]
            if not fixture.exists():
                subprocess.run(COMMON + ['/helpers/http_performance_client.py', 'fixture'] + common,
                               check=True, timeout=120)
            subprocess.run(COMMON + ['/helpers/http_performance_client.py', 'bench'] + common + [
                '--arm', arm, '--output-dir', str(ROOT / 'http'), '--requests', '16',
                '--server-evidence', str(evidence_path)], check=True, timeout=1800)
            quality(arm)
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


if not ORIGINAL.exists():
    shutil.copyfile(TARGET, ORIGINAL)
try:
    for arm, root in [('base-a1', '/baseline'), ('candidate-b1', '/source'),
                      ('candidate-b2', '/source'), ('base-a2', '/baseline')]:
        run(arm, root)
    save(ROOT / 'complete.json', {'complete': True, 'order': ['A', 'B', 'B', 'A']})
finally:
    shutil.copyfile(ORIGINAL, TARGET)
