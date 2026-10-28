"""Summarize global GPU and process working-set samples, keeping WDDM VRAM null."""
import argparse
import json
from common import ROOT,save,sha256

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='reports/soak-highres-memory.jsonl')
    parser.add_argument('--output',default='reports/soak-highres-memory-summary.json')
    args=parser.parse_args()
    source=ROOT/args.source;samples=[]
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line)
            gpu=item['gpu_global'];process=item['inspection_working_set_bytes']
            if gpu.get('exit_code')!=0 or process.get('exit_code')!=0 or not process.get('stdout'):continue
            fields=[x.strip() for x in gpu['stdout'].split(',')]
            if len(fields)!=6:continue
            samples.append({'utc':item['utc'],'elapsed_seconds':item['elapsed_seconds'],'gpu_name':fields[0],
                'global_gpu_used_mib':float(fields[1]),'global_gpu_total_mib':float(fields[2]),
                'temperature_c':float(fields[3]),'gpu_util_percent':float(fields[4]),'power_w':float(fields[5]),
                'inspection_working_set_mib':int(process['stdout'])/2**20})
    if len(samples)<2:raise ValueError('insufficient runtime memory samples')
    def series(key):
        values=[sample[key] for sample in samples]
        return {'first':values[0],'last':values[-1],'min':min(values),'max':max(values),'last_minus_first':values[-1]-values[0]}
    save(ROOT/args.output,{'source_sha256':sha256(source),'sample_count':len(samples),
        'first_utc':samples[0]['utc'],'last_utc':samples[-1]['utc'],'sampled_seconds':samples[-1]['elapsed_seconds']-samples[0]['elapsed_seconds'],
        'global_gpu_used_mib':series('global_gpu_used_mib'),'inspection_working_set_mib':series('inspection_working_set_mib'),
        'temperature_c':series('temperature_c'),'process_gpu_memory_bytes':None,
        'gpu_memory_method':'nvidia-smi global device memory; WDDM process-specific VRAM unavailable'})
    print(args.output)

if __name__=='__main__':main()
