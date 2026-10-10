import ast, importlib.util, json, sys
from pathlib import Path
root=Path(sys.argv[1]); helpers=root/'helpers'
spec=importlib.util.spec_from_file_location('term_common',helpers/'long_free_common.py'); common=importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
for backend in ('triton','flashinfer'):
    old=json.loads((root/'reference'/backend/'launch.json').read_text())
    assert common.model_options(backend,'eager')==dict(old['model_options'],max_model_len=16384)
    assert common.sampling_options()==dict(old['sampling_options'],max_tokens=8192)
    rows=json.loads((root/'reference'/backend/'completed.json').read_text())['examples']
    assert [x['index'] for x in rows]==common.INDICES
    assert all(len(x['prompt_token_ids'])+8192<=16384 for x in rows)
    for name, expected in common.SCORER_HASHES.items():assert common.sha(helpers/name)==expected
for file in helpers.glob('*.py'): ast.parse(file.read_text(encoding='utf8'),filename=str(file))
print(json.dumps({'status':'CPU_BUDGET_PROTOCOL_PASS','only_engine_change':'max_model_len 4096 -> 16384','only_sampling_change':'max_tokens 3500 -> 8192','both_backends_checked':True,'original_eos_scoring_prompts_retained':True,'gpu_execution':False},indent=2))
