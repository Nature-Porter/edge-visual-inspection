"""Provider assignment evidence, explicitly not a proof of INT8 layer execution."""
import argparse
from collections import Counter
from common import read,save,sha256
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    nodes=[e['args'] for e in read(a.profile) if e.get('cat')=='Node' and e.get('args',{}).get('provider')]
    counts=Counter((n['provider'],n.get('op_name','unknown')) for n in nodes)
    summary=[{'provider':provider,'operator':op,'events':n} for (provider,op),n in sorted(counts.items())]
    heavy={'Conv','MatMul','Gemm','QLinearConv','QLinearMatMul','ConvInteger','MatMulInteger'}
    save(a.output,{'profile_sha256':sha256(a.profile),'model_sha256':sha256(a.model),'provider_operator_events':summary,
        'cpu_heavy_fallback':[n for n in summary if n['provider']=='CPUExecutionProvider' and n['operator'] in heavy],
        'cpu_other_nodes':[n for n in summary if n['provider']=='CPUExecutionProvider' and n['operator'] not in heavy],
        'int8_execution_verified':False,'precision_note':'ORT provider trace proves provider assignment only; TensorRT layer inspector evidence required for precision.'})
