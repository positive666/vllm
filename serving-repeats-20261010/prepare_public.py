"""Prepare redacted, hash-linked serving evidence without changing model records."""
import hashlib, json, re, shutil, sys, tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve()
work=root/'publication-work-v1'
assert not work.exists();work.mkdir()
def sha(b):return hashlib.sha256(b).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
private_ip=re.compile(rb'(?<![\d.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\d.])')
def redacted_copy(src,dst,allow_log_redaction=False):
    data=src.read_bytes();matches=list(private_ip.finditer(data))
    assert not matches or allow_log_redaction, ('Private IP outside log',str(src))
    public=private_ip.sub(lambda m:re.sub(rb'\d',b'x',m[0]),data)
    assert len(public)==len(data) and not private_ip.search(public)
    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(public)
    return {'raw_sha256':sha(data),'public_sha256':sha(public),'bytes':len(data),'redactions':[{'offset':m.start(),'length':m.end()-m.start(),'kind':'RFC1918 IPv4 digit masking'} for m in matches]}
raw_summary=json.loads((root/'serving-summary-v1.json').read_text())
assert raw_summary['status']=='SERVING_RECORDS_AND_METRICS_PASS'
map_files={}
for folder in ('helpers','reference','results'):
    for src in sorted((root/folder).rglob('*')):
        if not src.is_file() or '__pycache__' in src.parts:continue
        assert not src.is_symlink()
        relative=src.relative_to(root).as_posix()
        map_files[relative]=redacted_copy(src,work/relative,src.suffix=='.log')
for src in sorted(root.iterdir()):
    if src.suffix=='.sh' or src.name in ('protocol.json','http-fixture.json','producer-manifest.json','PLAN.md','cpu-preflight-receipt-v1.json','cpu-consumer-checks-receipt-v1.json','serving-summary-v1.json','resource-summary-v1.json','idle-summary-v1.json','telemetry-summary-v1.json','audit_client_load.py','client-load-summary-v1.json','audit_serving.py','resource_audit.py','audit_idle_wait.py','audit_telemetry.py','cpu_consumer_checks.py','prepare_public.py') or (src.name.startswith('audit-') and src.name.endswith('-receipt-v1.json')):
        map_files[src.name]=redacted_copy(src,work/src.name)
log_bindings={}
for launch in raw_summary['launches']:
    name='results/serve-'+launch['arm']+'.log';record=json.loads((root/'results'/('serve-'+launch['arm']+'.json')).read_text())
    raw=(root/name).read_bytes();public=(work/name).read_bytes();prefix=launch['log_binding']['recorded_prefix_bytes']
    assert sha(raw[:prefix])==record['server_log_sha256']
    log_bindings[name]={'raw_sha256':sha(raw),'public_sha256':sha(public),'recorded_raw_prefix_sha256':record['server_log_sha256'],'public_prefix_sha256':sha(public[:prefix]),'prefix_bytes':prefix}
write_json(work/'raw-public-map.json',{'schema':'gdn-serving-redactions-v1','files':map_files,'log_bindings':log_bindings,'scope':'Raw records and metric values unchanged. Only RFC1918 IPv4 digits in server logs masked; exact offsets and raw/public hashes retained. Raw originals remain local.'})
code=(root/'audit_serving.py').read_text()
old="log_path=results/f'serve-{arm}.log';binding=log_binding(log_path,record['server_log_sha256']);raw_log=log_path.read_text(errors='replace');checks=LogChecks()"
new="""log_path=results/f'serve-{arm}.log'
        redactions=read(root/'raw-public-map.json');mapping=redactions['log_bindings'][log_path.relative_to(root).as_posix()]
        check(mapping['recorded_raw_prefix_sha256']==record['server_log_sha256'],'Raw/public recorded prefix mapping '+arm)
        check(sha(log_path)==mapping['public_sha256'],'Exact public log binding '+arm)
        binding=log_binding(log_path,mapping['public_prefix_sha256'])
        check(binding['recorded_prefix_bytes']==mapping['prefix_bytes'],'Same byte-length log prefix '+arm)
        binding['raw_public_mapping']=mapping
        raw_log=log_path.read_text(errors='replace');checks=LogChecks()"""
assert code.count(old)==1;code=code.replace(old,new)
(work/'audit_serving_public.py').write_text(code,encoding='utf8')
histories=[('v1',root.parent/'serving-repeats-20261010','interrupted-v1','interrupted-results-v1.tar.gz','e0a10580bd581fd9665cf446e8c55475ea43e9cad875a6b77d1d2b7e4b46c4c7',2),('v2',root.parent/'serving-repeats-v2-20261010','interrupted-v2','interrupted-results-v2.tar.gz','3c989e24198e86dc910eced86e3fcde5bcd33eb4c08ac391d0aca282e17ea97e',3)]
for label,origin,extracted,archive,wanted,completed in histories:
    assert sha((origin/archive).read_bytes())==wanted
    dest=work/'history'/label;dest.mkdir(parents=True)
    history_map={}
    for src in sorted((origin/extracted/'results').rglob('*')):
        if not src.is_file():continue
        relative=src.relative_to(origin/extracted).as_posix()
        history_map[relative]=redacted_copy(src,dest/relative,src.suffix=='.log')
    for name in ('protocol.json','producer-manifest.json','run_cells.sh','monitor_host.sh'):
        history_map[name]=redacted_copy(origin/name,dest/name)
    write_json(dest/'raw-public-map.json',{'files':history_map})
    write_json(dest/'interruption.json',{'label':label,'raw_archive_sha256':wanted,'actual_matrix_exit':1,'actual_cleanup_stop_exit':0,'completed_serving_launches':completed,'pooled_into_v3':False,'notes':'See README for recorded failure boundary and attribution limits.'})
write_json(root/'publication-work-receipt-v1.json',{'work':str(work),'files':len(map_files),'redacted_files':{p:len(v['redactions']) for p,v in map_files.items() if v['redactions']},'status':'PREPARED_PUBLIC_COPY_RECHECK_REQUIRED'})
print(json.dumps({'status':'PREPARED_PUBLIC_COPY_RECHECK_REQUIRED','work':str(work),'files':len(map_files)}))
