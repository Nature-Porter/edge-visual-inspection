"""Summarize fixed, same-model CPU/CUDA postprocessing experiments."""
import json
from statistics import median
from common import ROOT, read, save, sha256

def main():
    root=ROOT/'runs'
    manifest=read(ROOT/'artifacts/metal_nut/fp32/model_manifest.json')
    rounds={}
    for post in ('cpu','cuda'):
        data=[]
        for index in range(1,4):
            path=root/f'bench-pipeline-{post}-{index}'/'summary.json'
            summary=read(path)
            if summary['model_sha256']!=manifest['onnx_sha256'] or summary['backend']!='ort-cuda' or summary['mode']!='pipeline' or summary['wall_ms']['count']!=1000:
                raise ValueError(f'benchmark mismatch: {path}')
            data.append({'summary':str(path.relative_to(ROOT)),'sha256':sha256(path),
                         'wall_p50_ms':summary['wall_ms']['p50'],'wall_p95_ms':summary['wall_ms']['p95'],
                         'gpu_p50_ms':summary['model_gpu_ms']['p50'],'post_p50_ms':summary['postprocess_ms']['p50']})
        rounds[post]={'runs':data,'median_wall_p50_ms':median(r['wall_p50_ms'] for r in data),
                      'median_post_p50_ms':median(r['post_p50_ms'] for r in data)}
    interleaved=[]
    for index,post in enumerate(('cuda','cpu','cuda','cpu'),1):
        path=root/f'bench-interleaved-{index}-{post}'/'summary.json';summary=read(path)
        if summary['model_sha256']!=manifest['onnx_sha256'] or summary['wall_ms']['count']!=1000:raise ValueError(f'interleaved mismatch: {path}')
        interleaved.append({'postprocess':post,'summary':str(path.relative_to(ROOT)),
                            'wall_p50_ms':summary['wall_ms']['p50'],'gpu_p50_ms':summary['model_gpu_ms']['p50'],
                            'post_p50_ms':summary['postprocess_ms']['p50']})
    event_sets=[]
    for post in ('cpu','cuda'):
        path=root/f'ab-post-{post}'/'events.jsonl'
        event_sets.append({e['frame_id']:e for e in map(json.loads,path.read_text(encoding='utf-8').splitlines())})
    if set(event_sets[0])!=set(event_sets[1]) or len(event_sets[0])!=69:raise ValueError('A/B frame coverage mismatch')
    differences=[];same_decisions=0
    for frame_id in sorted(event_sets[0]):
        a,b=(events[frame_id] for events in event_sets)
        if a['source_image_id']!=b['source_image_id'] or a['score'] is None or b['score'] is None:raise ValueError('A/B source or score mismatch')
        differences.append(abs(a['score']-b['score']))
        same_decisions+=a['status']==b['status'] and a['decision']==b['decision']
    save(ROOT/'reports/postprocess_ab.json',{'model_sha256':manifest['onnx_sha256'],'validation_replay_sha256':sha256(ROOT/'data/replay.json'),
        'three_rounds':rounds,'interleaved':interleaved,'score_comparison':{'n':len(differences),'max_abs':max(differences),'same_status_decision':same_decisions},
        'temperature_observations':['reports/benchmark-gpu-before.txt','reports/benchmark-gpu-after.txt','reports/benchmark-gpu-interleaved-after.txt'],
        'interpretation':'CUDA kernel reduces postprocess time. Whole-pipeline benefit is small and varies with GPU model time and temperature; do not attribute all wall-time differences to the kernel.'})
    print('reports/postprocess_ab.json')

if __name__=='__main__':main()
