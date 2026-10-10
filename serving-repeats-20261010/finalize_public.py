"""Archive verified public results and create a manifest for the evidence commit."""
import hashlib,json,shutil,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();work=root/'publication-work-v1';stage=root/'publication-stage-v1'
assert work.is_dir() and not stage.exists();stage.mkdir()
def sha(data):return hashlib.sha256(data).hexdigest()
maps=[(work,root,None),(work/'history/v1',root.parent/'serving-repeats-20261010','interrupted-v1'),(work/'history/v2',root.parent/'serving-repeats-v2-20261010','interrupted-v2')]
for public_root,raw_root,extracted in maps:
    mapping=json.loads((public_root/'raw-public-map.json').read_text())
    for relative,record in mapping['files'].items():
        source=(raw_root/extracted/relative) if extracted and relative.startswith('results/') else (raw_root/relative)
        raw_bytes=source.read_bytes();public_bytes=(public_root/relative).read_bytes()
        assert sha(raw_bytes)==record['raw_sha256'] and sha(public_bytes)==record['public_sha256']
        assert len(raw_bytes)==len(public_bytes)==record['bytes']
        expected=bytearray(raw_bytes)
        for change in record['redactions']:
            start=change['offset'];end=start+change['length']
            expected[start:end]=bytes(ord('x') if 48<=b<=57 else b for b in raw_bytes[start:end])
        assert bytes(expected)==public_bytes, ('Change outside declared digit masks',relative)
raw=json.loads((root/'serving-summary-v1.json').read_text());public=json.loads((work/'public-serving-summary-v1.json').read_text())
for key in ('comparisons','measured_requests','warmup_requests','token_chunk_size_histogram'):assert raw[key]==public[key],('Public metric differs',key)
for name in ('resource','idle','telemetry','client-load'):
    a=json.loads((root/(name+'-summary-v1.json')).read_text());b=json.loads((work/('public-'+name+'-summary-v1.json')).read_text());assert a==b,(name,'public result differs')
for src in sorted(work.rglob('*')):
    rel=src.relative_to(work)
    if not src.is_file() or 'results' in rel.parts or '__pycache__' in rel.parts:continue
    assert not src.is_symlink();target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
shutil.copyfile(root/'publication-readme.md',stage/'README.md')
archives=[]
for directory in [work,work/'history/v1',work/'history/v2']:
    relative=directory.relative_to(work);target=stage/relative/'results-public-v1.tar.gz'
    with tarfile.open(target,'w:gz') as tar:
        for src in sorted((directory/'results').rglob('*')):
            if src.is_file():tar.add(src,arcname=src.relative_to(directory).as_posix())
    with tarfile.open(target) as tar:
        count=0
        for member in tar.getmembers():
            assert member.isfile() and member.name.startswith('results/')
            data=tar.extractfile(member).read();assert sha(data)==sha((directory/member.name).read_bytes());count+=1
    archives.append({'path':target.relative_to(stage).as_posix(),'entries':count,'sha256':sha(target.read_bytes()),'bytes':target.stat().st_size})
manifest={'schema':'gdn60403-serving-public-v1','source_head':raw['source_head'],'production_changed':False,'scope':'Eight complete native TP2 serving launches, with interrupted histories retained separately','archives':archives,'files':[]}
for src in sorted(stage.rglob('*')):
    if not src.is_file():continue
    data=src.read_bytes();manifest['files'].append({'path':src.relative_to(stage).as_posix(),'sha256':sha(data),'bytes':len(data)})
(stage/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
receipt={'status':'PUBLIC_PACKAGE_VERIFIED','stage':str(stage),'files':len(manifest['files']),'archives':archives,'manifest_sha256':sha((stage/'package-manifest.json').read_bytes())}
(root/'publication-stage-receipt-v1.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt))
