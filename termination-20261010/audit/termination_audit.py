"""Audit unforced budget extension against complete frozen prior outputs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

EOS = {248044, 248046}
INDICES = [198, 206, 209, 228, 255, 285, 292, 318]

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def digest(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())

def check(value, label):
    if not value:
        raise ValueError(label)

def compare(old, new):
    check(old['index'] == new['index'], 'Stable item identity')
    for field in ('question', 'gold', 'gold_text', 'prompt', 'prompt_token_ids'):
        check(old[field] == new[field], 'Original item preserved: ' + field)
    prior, current = old['token_ids'], new['token_ids']
    first = next((i for i, (a,b) in enumerate(zip(prior,current)) if a != b), None)
    reproduced = len(current) >= len(prior) and current[:len(prior)] == prior
    return {
        'index': new['index'], 'old_tokens': len(prior), 'new_tokens': len(current),
        'first_differing_token_index': first, 'entire_old_output_reproduced': reproduced,
        'old_finish_reason': old['finish_reason'], 'new_finish_reason': new['finish_reason'],
        'new_stop_reason': new['stop_reason'], 'new_last_token_id': current[-1],
        'eos_terminated': new['finish_reason'] == 'stop' and current[-1] in EOS,
        'old_predicted': old['predicted'], 'new_predicted': new['predicted'],
        'old_strict_correct': old['strict_correct'], 'new_strict_correct': new['strict_correct'],
        'new_has_answer_marker': new['has_answer_marker'],
        'prior_budget_truncation_resolved_for_same_token_prefix':
            old['finish_reason'] == 'length' and reproduced and
            new['finish_reason'] == 'stop' and current[-1] in EOS,
        'old_text_tail': old['text'][-500:], 'new_text_tail': new['text'][-800:],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    check(not args.output.exists(), 'Refuse to overwrite audit result')
    sys.path.insert(0, str(root/'helpers'))
    from long_free_common import flags, summarize
    files = {}
    def read(path):
        data = path.read_bytes()
        files[str(path.relative_to(root)).replace('\\','/')] = {
            'sha256': sha_bytes(data), 'bytes': len(data)}
        return json.loads(data)
    result = {'status':'RECORD_CONTRACT_PASS', 'gpu_execution':False,
              'scope':'Eight selected natural-generation cases; no population quality or performance claim',
              'arms':{}}
    for backend in ('flashinfer', 'triton'):
        current_dir = root/'results'/('budget-'+backend)
        old_launch = read(root/'reference'/backend/'launch.json')
        old_output = read(root/'reference'/backend/'completed.json')
        launch = read(current_dir/'launch.json')
        completed = read(current_dir/'completed.json')
        sampling = read(current_dir/'sampling-runtime.json')
        shutdown = read(current_dir/'shutdown.json')
        check((root/'results'/('budget-'+backend+'.exit')).read_text().strip()=='0','Actual arm exit0')
        check(shutdown['generation_completed'] and shutdown['status']=='shutdown-returned', 'Actual engine shutdown')
        for key in ('source_head', 'source_files', 'model_files', 'runtime_versions', 'tokenization_options'):
            check(launch[key] == old_launch[key], 'Unchanged binding '+key)
        check(launch['model_options'] == dict(old_launch['model_options'], max_model_len=16384), 'Only context ceiling changed')
        check(launch['sampling_options'] == dict(old_launch['sampling_options'], max_tokens=8192), 'Only output budget changed')
        for name,expected in launch['helper_sha256'].items():
            check(sha_bytes((root/'helpers'/name).read_bytes())==expected, 'Executed helper hash '+name)
        check(completed['launch_sha256'] == sha_bytes((current_dir/'launch.json').read_bytes()), 'Launch hash')
        check(sampling['max_tokens']==8192 and sampling['min_tokens']==0 and not sampling['ignore_eos']
              and sampling['trace_decode_token_ids'] is None and sampling['temperature']==0, 'Native original stop/sampling contract')
        for when in ('before','after'):
            runtime = read(current_dir/('runtime-'+when+'.json'))
            check(runtime['engine_core_class']=='InprocClient', 'Same synchronous runner')
            ranks = runtime['worker_ranks']
            check(all(r['config']['model_config']['max_model_len']==16384 and
                      r['config']['model_config']['enforce_eager'] and
                      r['config']['kernel_config']['gdn_decode_backend']==backend for r in ranks),
                  'Actual model ceiling, mode and backend')
            check(len(ranks)==2 and {r['tp_rank'] for r in ranks}=={0,1}, 'Both TP ranks')
            check(all(r['tp_world_size']==2 and not r['graphs_captured'] and not r['captured_token_counts'] for r in ranks), 'Actual TP2 eager')
        old_rows, new_rows = old_output['examples'], completed['examples']
        check([r['index'] for r in new_rows]==INDICES==[r['index'] for r in old_rows], 'Original eight-item ordering')
        rows=[]
        for old,new in zip(old_rows,new_rows):
            for row in (old,new):
                check(digest(row['token_ids'])==row['token_ids_sha256'], 'Returned token hash')
                check(digest(row['prompt_token_ids'])==row['prompt_token_ids_sha256'], 'Prompt hash')
                check(row['usage']['completion_tokens']==len(row['token_ids']), 'Returned token count')
                for key,val in flags(row['text'],row['gold'],row['finish_reason']).items():
                    check(row[key]==val, 'Frozen scoring '+key)
            check(not (EOS & set(new['token_ids'][:-1])), 'No returned EOS followed by further output')
            check(new['finish_reason'] in ('stop','length'), 'Known termination')
            rows.append(compare(old,new))
        check(completed['summary']==summarize(new_rows),'Summary agrees with original scorer')
        result['arms'][backend]={'comparisons':rows,'new_summary':completed['summary'],
            'all_prior_outputs_reproduced':all(r['entire_old_output_reproduced'] for r in rows),
            'q255':next(r for r in rows if r['index']==255)}
    result['input_files']=files
    result['auditor_sha256']=sha_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'arms':{k:{'all_prior_outputs_reproduced':v['all_prior_outputs_reproduced'],'summary':v['new_summary'],'q255':v['q255']} for k,v in result['arms'].items()}},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
