"""Independent CPU checks and 80-digit reference for two first-divergence values."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import io
import json
from pathlib import Path
import tarfile
import torch

torch.set_num_threads(2)
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--v1',type=Path,required=True)
a=parser.parse_args()
r=json.loads((a.root/'results/report.json').read_text())
r1=json.loads((a.v1/'results/report.json').read_text())
assert r['exit_code']==r1['exit_code']==0
archive=a.root/'selected-captures.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='64d98642b2674031d0cf1aa775cee1afb89be8f14a237d7eb95cbe4bfd4821f4'
with tarfile.open(archive,'r:gz') as tf:
    names=tf.getnames()
    assert len(names)==12 and all(n.startswith(('short-B/trace/captures/','short-C/trace/captures/')) and '..' not in Path(n).parts for n in names)
    caps={}
    for n in names:
        if not n.endswith('.pt'):
            continue
        raw=tf.extractfile(n).read()
        meta=json.load(tf.extractfile(n[:-3]+'.json'))
        assert hashlib.sha256(raw).hexdigest()==meta['binary_sha256']
        caps[n]=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        assert len(raw)==meta['binary_size']

def D(x):
    return Decimal.from_float(float(x))

def decimal_ref(t,h,v):
    with localcontext() as ctx:
        ctx.prec=80
        qkv=t['mixed_qkv'][4]
        qi=qkv[(h//3)*128:(h//3+1)*128]
        ki=qkv[1024+(h//3)*128:1024+(h//3+1)*128]
        q=[D(x) for x in qi]; k=[D(x) for x in ki]
        # Decimal(1e-6) intentionally uses the exact literal, not binary FP32 epsilon.
        qn=(sum(x*x for x in q)+Decimal('0.000001')).sqrt()
        kn=(sum(x*x for x in k)+Decimal('0.000001')).sqrt()
        q=[x/qn/Decimal(128).sqrt() for x in q]; k=[x/kn for x in k]
        x=D(t['a'][4,h])+D(t['dt_bias'][h])
        soft=(Decimal(1)+x.exp()).ln() if x<=20 else x
        decay=(-D(t['A_log'][h]).exp()*soft).exp()
        beta=Decimal(1)/(Decimal(1)+(-D(t['b'][4,h])).exp())
        old=[D(x)*decay for x in t['target_state_before'][h,v]]
        delta=(D(qkv[2048+h*128+v])-sum(x*y for x,y in zip(old,k)))*beta
        return sum((x+delta*y)*z for x,y,z in zip(old,k,q))

checks=[]
for case in r1['cases']+r['cases']:
    rank=case['rank'];layer=case.get('layer',0)
    b=caps[f'short-B/trace/captures/rank-{rank}-layer-{layer}-step-1-eid-2.pt']
    c=caps[f'short-C/trace/captures/rank-{rank}-layer-{layer}-step-1-eid-2.pt']
    assert all(torch.equal(b[k][4],c[k][4]) for k in ('mixed_qkv','a','b'))
    assert all(torch.equal(b[k],c[k]) for k in ('A_log','dt_bias','target_state_before'))
    bo,co=b['out'][4].reshape(24,128),c['out'][4].reshape(24,128)
    diff=(bo.double()-co.double()).abs()
    assert int((diff!=0).sum())==case['observed_output']['different']
    assert diff.max().item()==case['observed_output']['absmax']
    assert len(case['replays'])==12
    for rec in case['replays']:
        assert rec['output_vs_captured']['equal'] and rec['state_vs_captured']['equal']
        assert rec['output_vs_rounded_fp64']['frozen_tolerance'] and rec['state_vs_rounded_fp64']['frozen_tolerance']
    checks.append(dict(rank=rank,layer=layer,input_state_exact=True,own_fidelity_records=12,output_differences=int((diff!=0).sum()),output_absmax=diff.max().item()))

selected=caps['short-B/trace/captures/rank-1-layer-1-step-1-eid-2.pt']
decimal=[]
for element in r['cases'][0]['rounding']['elements']:
    h,v=element['head'],element['value']
    value=decimal_ref(selected,h,v)
    difference=abs(value-D(element['fp64']))
    assert difference<Decimal('1e-16')
    decimal.append(dict(head=h,value=v,reference_decimal80=str(value),fp64_abs_error=str(difference),triton_abs_error=str(abs(value-D(element['triton']))),flashinfer_abs_error=str(abs(value-D(element['flashinfer'])))))

resources=[]
for folder in [a.v1,a.root]:
    p=folder/'results'
    assert all((p/name).read_text().strip()=='0' for name in ['run.exit','docker-exec.exit','numeric.exit','cleanup.exit'])
    assert len((p/'final-processes.csv').read_text().strip().splitlines())==1
    assert ', 1 MiB, 0 %' in (p/'final-gpus.csv').read_text()
    text=(p/'host-processes.log').read_text()
    frames=[]
    for line in text.splitlines():
        if line.startswith('2026-'):
            frames.append(dict(time=line,compute=[],owned=[]));owned=False
        elif line=='OWNED':owned=True
        elif line:
            if owned:frames[-1]['owned'].append(line.strip())
            else:frames[-1]['compute'].append(line)
    matched=0;unmatched=[]
    for frame in frames:
        for row in frame['compute']:
            pid=row.split(',')[1].strip()
            if pid in frame['owned']:matched+=1
            else:unmatched.append(dict(time=frame['time'],row=row))
    resources.append(dict(folder=folder.name,snapshots=len(frames),matched_compute_rows=matched,unmatched=unmatched,cleanup_exit=0,final_gpu_idle=True,limitation='Sequential five-second sampling; zero observed compute rows does not prove absence during the short kernel execution.'))
report=dict(status='CPU_BINDING_AND_HIGH_PRECISION_REFERENCE_PASS',cases=checks,decimal80=decimal,resource_observations=resources,gpu_execution=False,scope='Independent arithmetic checks on original captures, not GPU re-execution or model-quality equivalence.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (a.root/'cpu-audit.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
