import os,site,subprocess
from pathlib import Path
includes=[]
for root in site.getsitepackages():
 for p in (Path(root)/'nvidia').glob('*/include'):
  if p.is_dir():includes.append(str(p))
assert any((Path(p)/'cusparse.h').is_file() for p in includes),includes
env=dict(os.environ,CPATH=':'.join(includes))
print('BUILD_USING_INSTALLED_NVIDIA_HEADERS',includes,flush=True)
subprocess.run(['/opt/glm53-image-gn/bin/uv','build','--wheel','--no-build-isolation','--offline','--python','/opt/glm53-image-gn/.venv/bin/python','--out-dir','/build/wheels'],cwd='/build/source/lmcache',env=env,check=True)
