import argparse
import math
import random
from collections import Counter
from pathlib import Path
from common import ROOT, config, save, sha256

def build(root,seed=42,max_calibration=128):
    import cv2
    rng=random.Random(seed)
    seen={}
    def records(folder):
        result=[]
        for p in sorted(folder.glob('*/*.png')):
            digest=sha256(p)
            if digest in seen: raise ValueError(f'duplicate content: {p} and {seen[digest]}')
            seen[digest]=str(p)
            image=cv2.imread(str(p))
            if image is None: raise ValueError(f'bad image: {p}')
            label=int(p.parent.name!='good')
            mask=root/'ground_truth'/p.parent.name/(p.stem+'_mask.png') if label else None
            if mask:
                m=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
                if m is None or m.shape!=image.shape[:2] or not m.any(): raise ValueError(f'bad/missing mask: {mask}')
            result.append({'source_image_id':digest,'path':p.relative_to(ROOT).as_posix(),'sha256':digest,
              'label':label,'defect_type':p.parent.name,'mask_path':mask.relative_to(ROOT).as_posix() if mask else None,
              'mask_sha256':sha256(mask) if mask else None})
        return result
    normal=records(root/'train'); official_test=records(root/'test')
    if len(normal)<7 or any(r['label'] for r in normal) or not official_test: raise ValueError('missing/invalid real dataset')
    rng.shuffle(normal); n=len(normal); a=math.floor(n*.7); b=math.floor(n*.15)
    train,stats,dev=normal[:a],normal[a:a+b],normal[a+b:]
    validation=list(dev); test=[]
    for group in sorted({r['defect_type'] for r in official_test}):
        rows=[r for r in official_test if r['defect_type']==group]; rng.shuffle(rows)
        if len(rows)<2: raise ValueError(f'cannot split group {group}')
        cut=max(1,min(len(rows)-1,math.floor(len(rows)*.3+.5)))
        validation+=rows[:cut]; test+=rows[cut:]
    calibration=rng.sample(train,min(max_calibration,len(train)))
    return {'train_normal':train,'stats_normal':stats,'dev_normal':dev,'calibration':calibration,'validation':validation,'test':test}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/metal_nut.yaml'); a=p.parse_args(); c=config(a.config)
    splits=build(ROOT/c['data']['root'],c['project']['seed'],c['calibration']['max_images'])
    summary={}
    for name,rows in splits.items():
        path=ROOT/'data'/f'{name}.json'
        save(path,{'schema_version':1,'split':name,'seed':42,'calibration_overlap':'train_normal only','records':sorted(rows,key=lambda r:r['path'])})
        summary[name]={'count':len(rows),'labels':dict(Counter(r['label'] for r in rows)),'sha256':sha256(path)}
    save(ROOT/'data/replay.json',{'schema_version':1,'split':'validation','records':sorted(splits['validation'],key=lambda r:r['path'])})
    save(ROOT/'reports/data_protocol.json',{'status':'passed','splits':summary,'rounding':'floor(train .70/.15); round-half-up stratified .30; clamp 1..n-1','duplicates':0})
    print(summary)
