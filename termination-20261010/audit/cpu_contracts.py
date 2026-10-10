import ast, importlib.util, json, sys, copy
from pathlib import Path
p=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('audit',p/'audit/termination_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old=next(x for x in json.loads((p/'reference/flashinfer/completed.json').read_text())['examples'] if x['index']==255)
new=copy.deepcopy(old);new.update(token_ids=old['token_ids']+[248046],finish_reason='stop')
assert m.compare(old,new)['prior_budget_truncation_resolved_for_same_token_prefix']
changed=copy.deepcopy(new);changed['token_ids'][19]=71072
assert not m.compare(old,changed)['prior_budget_truncation_resolved_for_same_token_prefix']
assert m.compare(old,changed)['first_differing_token_index']==19
still=copy.deepcopy(new);still['finish_reason']='length'
assert not m.compare(old,still)['prior_budget_truncation_resolved_for_same_token_prefix']
for path in (p/'audit').glob('*.py'):ast.parse(path.read_text())
print(json.dumps({'status':'AUDITOR_PREFIX_AND_STOP_CONTRACT_PASS','synthetic_cases':3,'gpu_execution':False,'model_result':False}))
