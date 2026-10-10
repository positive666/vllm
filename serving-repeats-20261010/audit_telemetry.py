"""Describe sampled SM clocks and temperatures in recorded measured UTC windows."""
import argparse,csv,json,re,statistics
from datetime import datetime
from pathlib import Path
def utc(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
def number(s):
    m=re.match(r'\s*(\d+(?:\.\d+)?)',s)
    if not m:raise ValueError('Malformed NVML value')
    return float(m.group(1))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    results=a.root/'results';protocol=json.loads((results/'protocol.json').read_text());rows=[]
    for row in csv.reader((results/'host-gpus.csv').read_text().splitlines()):
        if len(row)!=8:raise ValueError('Incomplete telemetry row')
        stamp,index,uuid,mem,util,temp,clock,pstate=[x.strip() for x in row]
        if uuid not in protocol['gpu_uuids']:raise ValueError('Unexpected GPU')
        rows.append({'time':utc(stamp),'index':int(index),'uuid':uuid,'memory_mib':number(mem),'utilization_pct':number(util),'temperature_c':number(temp),'sm_clock_mhz':number(clock),'pstate':pstate})
    descriptions=[]
    for arm in protocol['order']:
        client=json.loads((results/'http'/f'{arm}.json').read_text())
        for c in (1,8):
            windows=[(utc(r['started_utc']),utc(r['finished_utc'])) for r in client['rounds'] if r['concurrency']==c]
            for uuid in protocol['gpu_uuids']:
                samples=[r for r in rows if r['uuid']==uuid and any(start<=r['time']<=end for start,end in windows)]
                if not samples:raise ValueError('No measured-window telemetry')
                d={'arm':arm,'concurrency':c,'gpu_uuid':uuid,'samples':len(samples),'pstates':sorted({r['pstate'] for r in samples})}
                for field in ('memory_mib','utilization_pct','temperature_c','sm_clock_mhz'):
                    v=[r[field] for r in samples];d[field]={'median':statistics.median(v),'min':min(v),'max':max(v)}
                descriptions.append(d)
    report={'status':'TELEMETRY_WINDOWS_DESCRIBED','samples_total':len(rows),'measured_windows':descriptions,'limits':['Five-second sequential sampling with second-resolution host timestamps; boundary assignment is approximate and transients between samples are not excluded','SM clocks and temperatures only; memory clocks and power are not continuously sampled','No clock, power or fan settings changed by these scripts; measurements are not clock-locked','Shared-host CPU/network/other-GPU activity is not eliminated by selected-GPU ownership checks']}
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps({'status':report['status'],'samples_total':len(rows),'groups':len(descriptions)}))
if __name__=='__main__':main()
