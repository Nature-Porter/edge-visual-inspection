"""Read-only device and process memory trend during a replay."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from common import ROOT

def command(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,errors='replace',timeout=10)
        return {'exit_code':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'error':str(exc)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='reports/soak-highres-memory.jsonl');p.add_argument('--until-summary',default='runs/soak-highres/summary.json');p.add_argument('--interval-seconds',type=float,default=30);p.add_argument('--max-seconds',type=float,default=1800);a=p.parse_args()
    if a.interval_seconds<=0 or a.max_seconds<=0:raise ValueError('positive interval and duration required')
    out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    start=time.monotonic()
    with out.open('w',encoding='utf-8') as log:
        while True:
            gpu=command(['nvidia-smi','--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw','--format=csv,noheader,nounits'])
            process=command(['powershell','-NoProfile','-Command','Get-Process inspection -ErrorAction SilentlyContinue | Select-Object -ExpandProperty WorkingSet64'])
            log.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
                'gpu_global':gpu,'inspection_working_set_bytes':process,'gpu_process_memory_bytes':None,
                'gpu_process_memory_note':'WDDM process-specific VRAM not reliably available'},ensure_ascii=False)+'\n');log.flush()
            if (ROOT/a.until_summary).is_file() or time.monotonic()-start>=a.max_seconds:break
            time.sleep(a.interval_seconds)
    print(out)

if __name__=='__main__':main()
