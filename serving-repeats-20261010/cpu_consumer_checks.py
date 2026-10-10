import ast, hashlib, importlib.util, json, sys, tarfile, tempfile
from pathlib import Path
root=Path(sys.argv[1]); checks=[]
for name in ('audit_idle_wait.py','audit_serving.py','resource_audit.py','audit_telemetry.py'):
    ast.parse((root/name).read_text(encoding='utf8')); checks.append('syntax '+name)
spec=importlib.util.spec_from_file_location('idle_audit',root/'audit_idle_wait.py'); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
protocol=json.loads((root/'protocol.json').read_text()); manifest=json.loads((root/'producer-manifest.json').read_text())
with tempfile.TemporaryDirectory(dir=root,prefix='cpu-idle-') as tmp:
    p=Path(tmp);(p/'results').mkdir();(p/'protocol.json').write_text(json.dumps(protocol));(p/'producer-manifest.json').write_text(json.dumps(manifest));(p/'results/container-id.txt').write_text('a'*64)
    phases=['initial']
    for arm in protocol['order']:phases.extend(['before-'+arm,'after-'+arm])
    record={'timestamp':'2026-10-10T00:00:00Z','container_id':'a'*64,'memory_csv':'\n'.join(g+', 1' for g in protocol['gpu_uuids']),'compute_csv':'','owned_pids':'10','busy':False,'unmatched':False}
    good=[dict(record,phase=phase,attempt=n) for phase in phases for n in (1,2)]
    def run(rows):
        (p/'results/idle-wait.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));return module.audit(p)
    assert run(good)['status']=='PASS';checks.append('all17 phases with two idle samples')
    bad=[dict(x) for x in good];bad[0]['busy']=True
    try:run(bad)
    except AssertionError:checks.append('reject inconsistent busy flag')
    else:raise ValueError('accepted false flag')
    unknown=dict(record,phase='initial',attempt=1,compute_csv=protocol['gpu_uuids'][0]+', 99, 100 MiB',busy=True,unmatched=True)
    rows=[unknown,dict(good[0],attempt=2),dict(good[1],attempt=3)]+good[2:]
    reviewed=run(rows);assert reviewed['status']=='REVIEW_REQUIRED' and len(reviewed['unmatched_observations'])==1;checks.append('retain unmatched despite later idle')
    for label,rows in [('missing phase',good[:-2]),('one idle',good[:-1]),('wrong CID',[dict(good[0],container_id='b'*64)]+good[1:])]:
        try:run(rows)
        except AssertionError:checks.append('reject '+label)
        else:raise ValueError('accepted '+label)
archive=root.parent/'serving-repeats-v2-20261010/interrupted-results-v2.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='3c989e24198e86dc910eced86e3fcde5bcd33eb4c08ac391d0aca282e17ea97e'
dest=archive.parent/'interrupted-v2';dest.mkdir()
with tarfile.open(archive) as tar:tar.extractall(dest,filter='data')
assert (dest/'results/matrix.exit').read_text().strip()=='1';assert (dest/'results/cleanup-stop.exit').read_text().strip()=='0'
print(json.dumps({'status':'CPU_CONSUMER_CHECKS_PASS','checks':checks,'v2_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'gpu_runs':False}))
