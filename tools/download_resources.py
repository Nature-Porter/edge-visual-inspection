"""Fetch official resources to project only; no alternate mirrors or privileged installs."""
import concurrent.futures
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from common import ROOT, save, sha256

RESOURCES = {
 'metal_nut.tar.xz': ('https://www.mydrive.ch/shares/150459/c68856a21dca589b0f8ff6d4ee0f18f4/download/420937637-1629959294/metal_nut.tar.xz', None),
 'imagenette2.tgz': ('https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz', '6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd'),
 'efficientad_pretrained_weights.zip': ('https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip','c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a')}

def fetch(item):
    name,(url,expected) = item
    dest=ROOT/'data/raw'; dest.mkdir(parents=True,exist_ok=True)
    target=dest/name
    try:
        if not target.exists():
            raise FileNotFoundError(f'User-managed downloads only. Download {url} to {target}')
        digest=sha256(target)
        if expected and expected != digest: raise ValueError('official SHA256 mismatch')
        if name.endswith('.zip'):
            extract=ROOT/'pre_trained'; extract.mkdir(exist_ok=True)
            with zipfile.ZipFile(target) as z:
                for entry in z.namelist():
                    if not (extract/entry).resolve().is_relative_to(extract.resolve()): raise ValueError('unsafe archive path')
                z.extractall(extract)
        else:
            with tarfile.open(target) as t: t.extractall(dest,filter='data')
        return {'name':name,'source':url,'sha256':digest,'expected_sha256':expected,'status':'passed'}
    except Exception as e:
        return {'name':name,'source':url,'status':'blocked','reason':str(e),'target':str(target)}

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(fetch,RESOURCES.items()))
    save(ROOT/'reports/downloads.json',results)
    for r in results: print(r,flush=True)
    raise SystemExit(0 if all(r['status']=='passed' for r in results) else 3)
