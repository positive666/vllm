"""First-step q255 arithmetic localization; isolated, untimed rank-local replay."""
import hashlib
import io
import json
import os
from pathlib import Path
import traceback

os.environ.update(FLASHINFER_WORKSPACE_BASE='/cache/fi', TRITON_CACHE_DIR='/cache/triton', CUDA_CACHE_PATH='/cache/cuda', VLLM_CACHE_ROOT='/cache/vllm')
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ROOT = Path('/results')
CAP = Path('/captures')
HEAD = 'd8ae9e900e3c96a7c1b5cb53cc9f572ab0e20fd6'

def require(ok, label):
    if not ok:
        raise ValueError(label)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, value):
    with (ROOT / name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')

def compare(x, y):
    x, y = x.double().cpu(), y.double().cpu()
    require(x.shape == y.shape, 'comparison shape')
    require(bool(torch.isfinite(x).all() and torch.isfinite(y).all()), 'finite tensors')
    diff = x-y
    denom = y.norm().item()
    rel = diff.norm().item()/denom if denom else (0 if not diff.any() else None)
    return dict(equal=bool(torch.equal(x,y)), different=int((x!=y).sum()), numel=x.numel(), absmax=diff.abs().max().item(), relative_l2=rel, frozen_tolerance=bool(torch.allclose(x,y,atol=.01,rtol=.01) and rel is not None and rel<.01))

def load(arm, rank, layer=0):
    path = CAP / ('short-'+arm) / 'trace/captures' / f'rank-{rank}-layer-{layer}-step-1-eid-2.pt'
    meta = json.loads(path.with_suffix('.json').read_text())
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==meta['binary_sha256'] and len(raw)==meta['binary_size'], 'capture sidecar binding')
    require(meta['arm']==arm and meta['rank']==rank, 'capture identity')
    join=meta['runtime_join']
    require(join['layer']==layer and join['actual_target_step']==1 and join['execution_id']==2, 'first-layer first-step')
    rows=[r for r in join['rows'] if r['index']==255]
    require(len(rows)==1, 'unique target')
    payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    require(payload['mixed_qkv'].shape==(8,5120) and payload['target_state_before'].shape==(24,128,128), 'fixed captured shape')
    return payload,meta,rows[0],dict(path=str(path),sha256=meta['binary_sha256'],sidecar_sha256=sha(path.with_suffix('.json')))

def reference(t, row):
    q,k,v=t['mixed_qkv'][row].double().split([1024,1024,3072])
    q=q.reshape(8,128).repeat_interleave(3,0)
    k=k.reshape(8,128).repeat_interleave(3,0)
    q=q/(q.square().sum(-1,keepdim=True)+1e-6).sqrt()*128**-.5
    k=k/(k.square().sum(-1,keepdim=True)+1e-6).sqrt()
    decay=(-t['A_log'].double().exp()*F.softplus(t['a'][row].double()+t['dt_bias'].double(),beta=1,threshold=20)).exp()
    beta=t['b'][row].double().sigmoid()
    state=t['target_state_before'].double()*decay[:,None,None]
    delta=(v.reshape(24,128)-(state*k[:,None,:]).sum(-1))*beta[:,None]
    state=state+delta[:,:,None]*k[:,None,:]
    output=(state*q[:,None,:]).sum(-1)
    return output,state

def rounding_details(t,f,ref):
    records=[]
    for ix in torch.nonzero(t!=f):
        h,v=ix.tolist()
        a,b,r=float(t[h,v]),float(f[h,v]),float(ref[h,v])
        lo,hi=sorted((a,b))
        adjacent=bool(torch.nextafter(torch.tensor(lo,dtype=torch.bfloat16),torch.tensor(float('inf'),dtype=torch.bfloat16)).item()==hi)
        records.append(dict(head=h,value=v,triton=a,flashinfer=b,fp64=r,nearest_bf16=float(ref[h,v].to(torch.bfloat16)),adjacent_bf16=adjacent,midpoint=(lo+hi)/2,distance_to_midpoint_in_interval=abs(r-(lo+hi)/2)/(hi-lo)))
    return dict(different=len(records),adjacent_count=sum(r['adjacent_bf16'] for r in records),elements=records)

def replay(cls,t,meta,target,backend,batch):
    row=target['row']
    values={}
    for name in ('mixed_qkv','a','b','A_log','dt_bias'):
        val=t[name]
        if val.ndim==2 and batch==1:
            val=val[row:row+1]
        strides=tuple(meta['original_tensors'][name]['original_stride'])
        values[name]=torch.empty_strided(val.shape,strides,dtype=val.dtype,device='cuda')
        values[name].copy_(val)
    pages=meta['runtime_join']['actual_pages'] if batch==8 else [1]
    target_page=target['gdn_state_page'] if batch==8 else 1
    target_row=row if batch==8 else 0
    state=torch.zeros(max(pages)+2,24,128,128,device='cuda',dtype=torch.float32)
    state[target_page].copy_(t['target_state_before'])
    state[0].fill_(.9375)
    state[-1].fill_(.9375)
    indices=torch.tensor(pages,dtype=torch.int32,device='cuda')
    out=torch.empty((batch,1,24,128),device='cuda',dtype=torch.bfloat16)
    op=cls.__new__(cls)
    for k,v in dict(num_k_heads=8,num_v_heads=24,head_k_dim=128,head_v_dim=128).items():
        object.__setattr__(op,k,v)
    if backend=='flashinfer':
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
        object.__setattr__(op,'_flashinfer_decode',gated_delta_rule_decode_pretranspose)
    fn=cls.forward_cuda if backend=='flashinfer' else cls.forward_native
    with torch.inference_mode():
        fn(op,**values,initial_state=state,ssm_state_indices=indices,out=out)
    torch.cuda.synchronize()
    require(bool((state[0]==.9375).all() and (state[-1]==.9375).all()),'reserved and unused pages unchanged')
    return out[target_row,0].cpu(),state[target_page].cpu()

report=dict(source_head=HEAD,scope='Scan q255 first-decode step all48 GDN layers and both ranks. Replay earliest layer with observed output disagreement under equal target inputs/state on one GPU. Other B8 state pages synthesized as zero; original full-pool stride/addresses are not reconstructed. Own target fidelity required. No TP collectives, natural generation, graph, performance or broad accuracy claim.',cases=[])
exit_code=1
try:
    manifest=json.loads(Path('/artifacts/performance-source.json').read_text())
    require(manifest['source_head']==HEAD,'source manifest revision')
    for name,value in manifest['files'].items():
        require(sha(Path('/source')/name)==value['sha256'],'source hash: '+name)
    report['source_manifest_sha256']=sha('/artifacts/performance-source.json')
    report['script_sha256']=sha(__file__)
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import GDNDecode
    import importlib.metadata as im
    report['versions']={p:im.version(p) for p in ['torch','triton','flashinfer-python','nvidia-cutlass-dsl']}
    require(report['versions']=={'torch':'2.13.0+cu129','triton':'3.7.1','flashinfer-python':'0.7.0.post1','nvidia-cutlass-dsl':'4.8.0'},'pinned runtime')
    props=torch.cuda.get_device_properties(0)
    report['device']=dict(name=props.name,uuid=str(props.uuid),capability=[props.major,props.minor])
    scan=[]
    candidates=[]
    for layer in range(64):
        if layer % 4 == 3:
            continue
        for rank in [0,1]:
            bt,bm,br,bb=load('B',rank,layer)
            ct,cm,cr,cb=load('C',rank,layer)
            target_eq={k:torch.equal(bt[k][4],ct[k][4]) for k in ['mixed_qkv','a','b']}
            target_eq.update({k:torch.equal(bt[k],ct[k]) for k in ['A_log','dt_bias','target_state_before']})
            oo=compare(bt['out'][4],ct['out'][4])
            record=dict(layer=layer,rank=rank,target_inputs_equal=target_eq,output=oo,state=compare(bt['target_state_after'],ct['target_state_after']),captures=[bb,cb])
            scan.append(record)
            if all(target_eq.values()) and not oo['equal']:
                candidates.append((rank,layer))
    report['first_step_layer_scan']=scan
    require(candidates,'At least one target output difference with identical incoming values')
    first=min(layer for rank,layer in candidates)
    selection=[(rank,layer) for rank,layer in candidates if layer==first]
    report['selected_identical_input_output_divergence']=selection
    write('scan.json',dict(scan=scan,selected=selection,candidates=candidates))
    for rank,layer in selection:
        bt,bm,br,bb=load('B',rank,layer)
        ct,cm,cr,cb=load('C',rank,layer)
        eq={key:torch.equal(bt[key][4],ct[key][4]) for key in ['mixed_qkv','a','b']}
        eq.update({key:torch.equal(bt[key],ct[key]) for key in ['A_log','dt_bias','target_state_before']})
        require(all(eq.values()),'identical initial inputs/state, rank '+str(rank))
        require(br['row']==cr['row']==4 and br['input_token']==cr['input_token']==8160,'matching actual target')
        ref_o,ref_s=reference(bt,br['row'])
        obs_t=bt['out'][br['row']].reshape(24,128)
        obs_f=ct['out'][cr['row']].reshape(24,128)
        case=dict(rank=rank,layer=layer,captures=[bb,cb],initial_exact=eq,observed_output=compare(obs_t,obs_f),observed_state=compare(bt['target_state_after'],ct['target_state_after']),rounding=rounding_details(obs_t,obs_f,ref_o),replays=[])
        for backend,t,m,r,obs in [('triton',bt,bm,br,obs_t),('flashinfer',ct,cm,cr,obs_f)]:
            for batch in [8,1]:
                for repeat in range(3):
                    o,s=replay(GDNDecode,t,m,r,backend,batch)
                    rec=dict(backend=backend,batch=batch,repeat=repeat,output_vs_captured=compare(o,obs),state_vs_captured=compare(s,t['target_state_after']),output_vs_fp64=compare(o,ref_o),state_vs_fp64=compare(s,ref_s),output_vs_rounded_fp64=compare(o,ref_o.to(torch.bfloat16)),state_vs_rounded_fp64=compare(s,ref_s.float()))
                    case['replays'].append(rec)
                    if batch == 8:
                        require(rec['output_vs_captured']['equal'] and rec['state_vs_captured']['equal'],'exact B8 own target fidelity')
                    require(rec['output_vs_rounded_fp64']['frozen_tolerance'] and rec['state_vs_rounded_fp64']['frozen_tolerance'],'unchanged reference tolerances')
        report['cases'].append(case)
        print(json.dumps(dict(rank=rank,layer=layer,output_differences=case['rounding']['different'],adjacent=case['rounding']['adjacent_count'],state_absmax=case['observed_state']['absmax'],replays=len(case['replays']))),flush=True)
    report['status']='BOUNDED_FIRST_OUTPUT_DIVERGENCE_REPLAY_PASS'
    exit_code=0
except Exception as e:
    report.update(status='FAILED',error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
finally:
    report['exit_code']=exit_code
    write('report.json',report)
    (ROOT/'numeric.exit').write_text(str(exit_code)+'\n')
raise SystemExit(exit_code)
