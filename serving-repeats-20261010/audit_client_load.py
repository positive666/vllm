"""Verify actual overlapping HTTP requests from recorded client timestamps."""
import argparse, hashlib, json, math
from pathlib import Path

def inspect_round(round):
    events=[]; duration=0.0
    for row in round['records']:
        start=row['started_after_round_seconds'];end=row['ended_after_round_seconds']
        assert math.isfinite(start) and math.isfinite(end) and 0<=start<end
        events.extend([(start,1),(end,-1)]);duration+=end-start
    active=peak=0;previous=integral=0.0
    for when,change in sorted(events):
        integral+=active*(when-previous);active+=change;previous=when
        assert 0<=active<=round['concurrency'];peak=max(peak,active)
    assert active==0 and peak==round['concurrency']
    assert math.isclose(integral,duration,rel_tol=1e-10,abs_tol=1e-8)
    wall=round['summary']['wall_seconds'];assert events and max(t for t,_ in events)<=wall
    return {'kind':round['kind'],'concurrency':round['concurrency'],'repeat':round['repeat'],'requests':len(round['records']),'peak_inflight':peak,'time_weighted_mean_inflight':integral/wall,'wall_seconds':wall,'request_span_seconds':duration}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
    result={'status':'FAIL'}
    try:
        plan=json.loads((args.root/'protocol.json').read_text());rounds=[];inputs={}
        for arm in plan['order']:
            path=args.root/'results/http'/f'{arm}.json';data=path.read_bytes();client=json.loads(data);inputs[arm]=hashlib.sha256(data).hexdigest()
            for round in client['warmup']+client['rounds']:rounds.append(dict(inspect_round(round),arm=arm))
        assert len(rounds)==64
        result={'status':'CLIENT_CONCURRENCY_PASS','rounds':rounds,'input_sha256':inputs,'scope':'In-flight HTTP request overlap at the client, not active GPU sequences or individual CUDA work.'}
    except Exception as error:result['error']=f'{type(error).__name__}: {error}'
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({'status':result['status'],'rounds':len(result.get('rounds',[])),'error':result.get('error')}))
    raise SystemExit(0 if result['status']=='CLIENT_CONCURRENCY_PASS' else 1)
if __name__=='__main__':main()
