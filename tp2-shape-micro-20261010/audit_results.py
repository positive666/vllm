import hashlib, json, math, statistics, sys
from pathlib import Path
root=Path(sys.argv[1])
def check(v,label):
    if not v: raise ValueError(label)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
original=(root/'original-benchmark.py').read_text(); current=(root/'helpers/benchmark_current_gdn.py').read_text()
check(original.split('def main():')[0]==current.split('def main():')[0], 'Benchmark and correctness functions unchanged')
marker='    # Complete correctness for every configuration before publishing timing.'
check(original.split(marker)[1]==current.split(marker)[1], 'Correctness and timing execution unchanged')
manifest=json.loads((root/'producer-manifest.json').read_text())
check(sha(root/'helpers/benchmark_current_gdn.py')==manifest['harness_sha256'],'Frozen harness')
def all_true(value):
    if type(value) is bool:return value
    if isinstance(value,dict):return bool(value) and all(all_true(v) for v in value.values())
    raise ValueError('Malformed sentinel check')
metrics=[]
def visit(value):
    if isinstance(value,dict):
        if {'finite','relative_l2','pointwise_over_threshold'}.issubset(value):
            check(value['finite'] and value['relative_l2']<0.01 and value['pointwise_over_threshold']==0,'Unchanged numerical thresholds')
            metrics.append(value['relative_l2'])
        for child in value.values():visit(child)
    elif isinstance(value,list):
        for child in value:visit(child)
rows=[];inputs={};correctness_count=0;methods=set();sample_counts=[]
for gpu in (4,6):
    path=root/'results'/f'micro-gpu{gpu}.json';data=json.loads(path.read_text());inputs[str(path.relative_to(root))]=sha(path)
    check((root/'results'/f'micro-gpu{gpu}.exit').read_text().strip()=='0','Actual benchmark exit')
    check(data['source_head']==manifest['source_head'] and data['harness_sha256']==manifest['harness_sha256'],'Executed source/harness')
    check(data['shape']['head_shapes']==[{'H':8,'HV':24}] and data['shape']['TP']==1 and data['shape']['model_TP']==2,'Single-GPU TP2 per-rank scope')
    check(data['environment']['CUDA_VISIBLE_DEVICES']==manifest['placement']['gpu_uuids'][0 if gpu==4 else 1],'Actual selected GPU')
    check(data['all_correctness_passed'] and data['args']['steps']==128 and data['args']['rounds']==5,'Unchanged correctness duration and actual rounds')
    check({(c['batch'],c['padding']) for c in data['cases']}=={(1,False),(8,False),(8,True)},'Three expected configurations')
    for case in data['cases']:
        check(case['passed'] and case['H']==8 and case['HV']==24 and case['correctness']['steps']==128,'Actual correctness case')
        for key in ('inactive_slots_bitwise_unchanged','cache_page_padding_bitwise_unchanged'):
            check(all_true(case['correctness'][key]),key)
        visit(case['correctness']);correctness_count+=1
        if case['padding']:continue
        for cache in ('warm','cold'):
            rounds=case['timings'][cache];medians={}
            for backend in ('triton','flashinfer'):
                values=rounds[backend];check(len(values)==5,'Five actual timing rounds')
                for entry in values:
                    check(math.isfinite(entry['median_us']) and entry['median_us']>0 and entry['samples']>0,'Valid measured times')
                    check(entry['p10_us']<=entry['median_us']<=entry['p90_us'],'Consistent quantiles')
                    methods.add(entry['timer']['method']);sample_counts.append(entry['samples'])
                medians[backend]=statistics.median(r['median_us'] for r in values)
            rows.append({'gpu':gpu,'batch':case['batch'],'state_cache':cache,
                'triton_us':medians['triton'],'flashinfer_us':medians['flashinfer'],
                'fi_latency_change_pct':(medians['flashinfer']/medians['triton']-1)*100,
                'triton_round_medians_us':[r['median_us'] for r in rounds['triton']],
                'fi_round_medians_us':[r['median_us'] for r in rounds['flashinfer']],
                'rotated_active_state_bytes':rounds['triton'][0]['rotated_active_state_bytes'],
                'samples_per_round_triton':[r['samples'] for r in rounds['triton']],
                'samples_per_round_fi':[r['samples'] for r in rounds['flashinfer']]})
check(correctness_count==6 and len(rows)==8 and metrics,'Complete two-GPU record coverage')
report={'status':'SIX_CONFIGURATIONS_AND_TIMING_RECORDS_PASS','source_head':manifest['source_head'],
    'scope':'Two independent single-GPU measurements of model TP2 per-rank H8/HV24 dimensions; no collectives/distributed or serving latency',
    'correctness_configurations':correctness_count,'numerical_metric_count':len(metrics),'max_relative_l2':max(metrics),
    'actual_timer_methods':sorted(methods),'minimum_event_samples_per_round':min(sample_counts),
    'input_sha256':inputs,'auditor_sha256':sha(Path(__file__)),'rows':rows,
    'limits':['One fresh process per GPU, five timed rounds per condition; public production defaults, no private tuning',
              'State-only rotation; QKV/gates/output reused; no physical HBM or cache-hit counter measurement',
              'No NCCL/TP critical-path, Python dispatch/detach CPU overhead, projection, convolution or serving timing']}
output=root/'micro-summary-v1.json';check(not output.exists(),'Refuse overwrite');output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
for row in rows:print(json.dumps({k:row[k] for k in ('gpu','batch','state_cache','triton_us','flashinfer_us','fi_latency_change_pct')}))
