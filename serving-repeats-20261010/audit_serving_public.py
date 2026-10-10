"""Independently recompute fixed-workload serving metrics from recorded requests."""
import argparse, csv, hashlib, importlib.util, json, math, re, statistics
from datetime import datetime
from pathlib import Path

def check(value, message):
    if not value: raise ValueError(message)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf8'))
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def quantile(values, q):
    ordered=sorted(values); at=(len(ordered)-1)*q; lo=math.floor(at); hi=math.ceil(at)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)
def close(a,b,label): check(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9),label)
def utc(value): return datetime.fromisoformat(value.replace('Z','+00:00'))
def option(command, key):return command[command.index(key)+1]
def function(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
class LogChecks:
    def __init__(self):self.notes=[]
    def check(self,v,label):check(v,label)
def log_binding(path, wanted):
    raw=path.read_bytes(); actual=hashlib.sha256(raw).hexdigest()
    if actual==wanted:return {'sha256':actual,'bytes':len(raw),'recorded_prefix_bytes':len(raw),'appended_bytes':0}
    h=hashlib.sha256();offset=0;found=None
    for line in raw.splitlines(keepends=True):
        h.update(line);offset+=len(line)
        if h.hexdigest()==wanted:found=offset;break
    check(found is not None,'Recorded server-log hash must match an exact byte prefix')
    return {'sha256':actual,'bytes':len(raw),'recorded_prefix_bytes':found,'appended_bytes':len(raw)-found,'appended_sha256':hashlib.sha256(raw[found:]).hexdigest()}
def run(root):
    results=root/'results'; p=read(results/'protocol.json');fixture=read(results/'http-fixture.json')
    check(sha(root/'protocol.json')==sha(results/'protocol.json'),'Frozen protocol')
    check(sha(results/'http-fixture.json')==p['fixture_sha256'],'Frozen prompt fixture')
    check(p['serving']=={'input_tokens':512,'output_tokens':128,'requests_per_round':16,'warmup_per_concurrency':8,'rounds':3,'concurrency':[1,8],'measured_total':768,'warmup_total':128,'independent_launches_per_backend':4},'Prespecified workload')
    for name,wanted in p['producer_helpers'].items():check(sha(root/'helpers'/name)==wanted,'Frozen producer '+name)
    runtime=read(results/'runtime-probe.json')
    check(runtime['required_imports_succeeded'] and not runtime['record_integrity_failures'] and runtime['torch']['gpu_count']==2,'Runtime probe')
    for path,wanted in p['native_files'].items():check(any(x['path']==path and x['sha256']==wanted for x in runtime['modules'].values()),'Native binding')
    source=read(root/'helpers/performance-source.json');classifier=function(root/'reference/audit_tp2.py','frozen_log_rules')
    reports=[]; all_rounds=[]; measured=warm=0; histogram={}; inputs={}; commands=[]
    for arm in p['order']:
        record_path=results/f'serve-{arm}.json';record=read(record_path);backend=arm.split('-')[0]
        check((results/f'serve-{arm}.exit').read_text().strip()=='0','Actual process exit '+arm)
        check(record['status']=='completed' and record['source_head']==p['source_head'],'Completed reviewed source '+arm)
        check(record['gpu_uuids']==p['gpu_uuids'] and record['gpu_blocks']==64 and record['kv_capacity_tokens']==21845,'Placement/capacity '+arm)
        check(record['state_dtype']=='float32' and record['tensor_parallel_size']==2,'Actual TP2 FP32 state '+arm)
        check(record['source_files']=={k:v['sha256'] for k,v in source['files'].items()},'Source file hashes '+arm)
        check(record['runtime']==runtime and record['protocol_sha256']==sha(results/'protocol.json'),'Runtime/protocol binding '+arm)
        check(record['supervisor_sha256']==p['producer_helpers']['tp2_supervisor.py'] and record['client_sha256']==p['producer_helpers']['http_performance_client.py'],'Executed helpers '+arm)
        command=list(record['command']); config=json.loads(option(command,'--kernel-config'))
        check(config=={'gdn_decode_backend':backend,'linear_backend':'marlin'},'Backend selection '+arm)
        command[command.index('--kernel-config')+1]='BACKEND_ONLY';commands.append(command)
        check(option(command,'--tensor-parallel-size')=='2' and option(command,'--mamba-ssm-cache-dtype')=='float32' and '--no-enable-prefix-caching' in command and '--enforce-eager' not in command,'Serving config '+arm)
        check(option(record['client_command'],'--requests')=='16' and option(record['client_command'],'--warmup-requests')=='8','Request plan '+arm)
        client_path=results/'http'/f'{arm}.json';client=read(client_path)
        check(sha(client_path)==record['client_result_sha256'],'Client file hash '+arm)
        check(client['all_successful'] and client['client_sha256']==p['producer_helpers']['http_performance_client.py'],'Completed client '+arm)
        check(client['fixture_sha256']==p['fixture_sha256'] and client['input_tokens_sha256']==fixture['input_tokens_sha256'],'Client fixture '+arm)
        hp=client['protocol'];check(client['protocol_sha256']==digest(hp),'Client protocol digest '+arm)
        check(hp['input_tokens']==512 and hp['output_tokens']==128 and hp['min_tokens']==128 and hp['ignore_eos'] and hp['temperature']==0 and hp['seed']==42 and hp['concurrency']==[1,8] and hp['repeats']==3,'Actual fixed-token protocol '+arm)
        check(len(client['warmup'])==2 and len(client['rounds'])==6,'Complete rounds '+arm)
        check([(r['repeat'],r['concurrency']) for r in client['rounds']]==[(1,1),(1,8),(2,8),(2,1),(3,1),(3,8)],'Frozen concurrency order '+arm)
        for round in client['warmup']+client['rounds']:
            count=8 if round['kind']=='warmup' else 16
            path=results/'http'/f"{arm}-{round['kind']}-c{round['concurrency']}-r{round['repeat']}.json"
            check(read(path)==round,'Standalone/full-result record equality '+str(path.name))
            check(round['requests_target']==count and round['protocol_sha256']==client['protocol_sha256'] and round['input_tokens_sha256']==fixture['input_tokens_sha256'],'Round binding')
            rows=round['records'];check(len(rows)==count and [r['index'] for r in rows]==list(range(count)),'Every request retained')
            metrics={key:[] for key in ('ttft_ms','token_ttft_ms','tpot_ms','request_seconds')}
            for row in rows:
                check(row['success'] and row['http_status']==200 and row['done'] and row['finish_reason']=='length','Completed streamed request')
                check(row['usage']['prompt_tokens']==512 and row['usage']['completion_tokens']==128 and len(row['output_token_ids'])==128,'Fixed work per request')
                check(row['prompt_token_ids']==fixture['input_tokens'],'Exact input tokens')
                check(digest(row['output_token_ids'])==row['output_token_ids_sha256'] and hashlib.sha256(row['text'].encode()).hexdigest()==row['text_sha256'],'Output integrity')
                chunks=row['token_chunks'];check(sum(c['token_count'] for c in chunks)==128,'Streamed token count')
                for c in chunks:
                    histogram[str(c['token_count'])]=histogram.get(str(c['token_count']),0)+1
                    check(0<=c['received_after_send_seconds']<=row['request_seconds'],'Stream timing bounds')
                first=chunks[0]['received_after_send_seconds'];last=chunks[-1]['received_after_send_seconds']
                check(all(a['received_after_send_seconds']<=b['received_after_send_seconds'] for a,b in zip(chunks,chunks[1:])),'Ordered stream timestamps')
                expected={'ttft_ms':row['chunks'][0]['received_after_send_seconds']*1000,'token_ttft_ms':first*1000,'tpot_ms':(last-first)/127*1000,'request_seconds':row['ended_after_round_seconds']-row['started_after_round_seconds']}
                for key,value in expected.items():close(row[key],value,'Request metric '+key);metrics[key].append(value)
                check(row['started_after_round_seconds']>=0 and row['ended_after_round_seconds']<=round['summary']['wall_seconds'],'Round clock bounds')
            summary=round['summary'];elapsed=summary['wall_seconds'];check(elapsed>0 and summary['requests']==summary['successful']==count and summary['failed']==0 and summary['output_tokens']==count*128,'Round totals')
            close(summary['output_tokens_per_second'],count*128/elapsed,'Token throughput');close(summary['requests_per_second'],count/elapsed,'Request throughput')
            check((utc(round['finished_utc'])-utc(round['started_utc'])).total_seconds()+.001>=elapsed,'Measured UTC window covers wall time')
            check(utc(record['ready_utc'])<=utc(round['started_utc'])<utc(round['finished_utc'])<=utc(record['finished_utc']),'Startup/shutdown excluded')
            for key,values in metrics.items():
                for stat,expected in [('p50',quantile(values,.5)),('p95',quantile(values,.95)),('mean',statistics.mean(values))]:close(summary[key][stat],expected,'Round '+key+'.'+stat)
            if round['kind']=='warmup':warm+=count
            else:measured+=count;all_rounds.append(round)
            inputs[path.relative_to(root).as_posix()]=sha(path)
        log_path=results/f'serve-{arm}.log'
        redactions=read(root/'raw-public-map.json');mapping=redactions['log_bindings'][log_path.relative_to(root).as_posix()]
        check(mapping['recorded_raw_prefix_sha256']==record['server_log_sha256'],'Raw/public recorded prefix mapping '+arm)
        check(sha(log_path)==mapping['public_sha256'],'Exact public log binding '+arm)
        binding=log_binding(log_path,mapping['public_prefix_sha256'])
        check(binding['recorded_prefix_bytes']==mapping['prefix_bytes'],'Same byte-length log prefix '+arm)
        binding['raw_public_mapping']=mapping
        raw_log=log_path.read_text(errors='replace');checks=LogChecks()
        log_status=classifier.verify_log_runtime_messages(raw_log,checks,arm)
        check('Capturing CUDA graphs (FULL)' in raw_log and 'CUDA graph pool memory:' in raw_log,'Actual graph capture log '+arm)
        check('tensor_parallel_size=2' in raw_log and "gdn_decode_backend='"+backend+"'" in raw_log,'Logged TP/backend '+arm)
        check(record['server_exit'] in (0,-15,143),'Controlled server shutdown '+arm)
        values={}
        for c in (1,8):
            rounds=[r for r in client['rounds'] if r['concurrency']==c];values[str(c)]={}
            for metric in p['analysis']['metrics']:
                vals=[r['summary'][metric] if '.' not in metric else r['summary'][metric.split('.')[0]][metric.split('.')[1]] for r in rounds]
                values[str(c)][metric]={'median':statistics.median(vals),'round_values':vals}
        reports.append({'arm':arm,'backend':backend,'started_utc':record['started_utc'],'ready_utc':record['ready_utc'],'finished_utc':record['finished_utc'],'server_exit':record['server_exit'],'metrics':values,'log_binding':binding,'log_classification':log_status,'notes':checks.notes})
        inputs[record_path.relative_to(root).as_posix()]=sha(record_path);inputs[client_path.relative_to(root).as_posix()]=sha(client_path)
    check(all(c==commands[0] for c in commands),'Only backend differs in native server command')
    check(measured==768 and warm==128,'Full planned matrix completed')
    summary=[]
    by_arm={r['arm']:r for r in reports}
    for c in (1,8):
        for metric in p['analysis']['metrics']:
            t=[r['metrics'][str(c)][metric]['median'] for r in reports if r['backend']=='triton'];f=[r['metrics'][str(c)][metric]['median'] for r in reports if r['backend']=='flashinfer']
            paired=[(by_arm[b]['metrics'][str(c)][metric]['median']/by_arm[a]['metrics'][str(c)][metric]['median']-1)*100 for a,b in p['paired_launches']]
            summary.append({'concurrency':c,'metric':metric,'triton':{'median':statistics.median(t),'min':min(t),'max':max(t),'launch_values':t},'flashinfer':{'median':statistics.median(f),'min':min(f),'max':max(f),'launch_values':f},'fi_relative_change_pct':(statistics.median(f)/statistics.median(t)-1)*100,'paired_relative_changes_pct':paired})
    return {'status':'SERVING_RECORDS_AND_METRICS_PASS','source_head':p['source_head'],'measured_requests':measured,'warmup_requests':warm,'independent_launches_per_backend':4,'unit':p['analysis']['unit'],'analysis':p['analysis'],'scope':p['scope'],'token_chunk_size_histogram':histogram,'launches':reports,'comparisons':summary,'input_sha256':inputs,'auditor_sha256':sha(Path(__file__)),'resource_audit':'Separate verdict required; mathematical/metric checks do not override unmatched PID observations','limits':['One synthetic 512/128 token workload; closed loop, loopback, no external network or natural-EOS evaluation','Four process launches per backend on the same shared host are descriptive replications, not a population confidence bound','TTFT uses first nonempty text SSE; token TTFT also retained. TPOT uses first/last token-ID chunk span; no pure ITL claim','Per-launch metrics are medians of three round metrics; request p95 is descriptive at 16 requests per round','Earlier graph quality differences and late-prefix REVIEW_REQUIRED remain separate']}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();check(not a.output.exists(),'Refuse overwrite')
    try:report=run(a.root)
    except Exception as e:
        a.output.write_text(json.dumps({'status':'FAIL','error':str(e),'error_type':type(e).__name__},indent=2)+'\n',encoding='utf8');raise
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k in ('status','measured_requests','warmup_requests','independent_launches_per_backend','token_chunk_size_histogram','comparisons')},indent=2))
if __name__=='__main__':main()
