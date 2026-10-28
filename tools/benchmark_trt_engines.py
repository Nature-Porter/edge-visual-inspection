"""Three-round same-TRT engine-only FP32/INT8 comparison; no pipeline speedup claim."""
import argparse
import statistics
import subprocess

import numpy as np

from common import ROOT, read, save, sha256
from integration_tests import runtime_env


ENGINES = {
    'fp32': ROOT / 'runs/trt-detailed-fp32',
    'int8_selective_2': ROOT / 'runs/trt-detailed-int8_selective_2_trt_compat',
}


def percentiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {'count': len(array), 'p50': float(np.percentile(array, 50)),
            'p95': float(np.percentile(array, 95)),
            'p99': float(np.percentile(array, 99))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--warmup-ms', type=int, default=1000)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup_ms < 0:
        raise ValueError('invalid benchmark counts')
    trt = ROOT / 'third_party/TensorRT-10.9.0.34'
    env = runtime_env()
    env['PATH'] = str(trt / 'lib') + ';' + str(trt / 'bin') + ';' + env['PATH']
    source = ENGINES['int8_selective_2'] / 'input.bin'
    if not source.is_file():
        raise FileNotFoundError(source)
    input_hash = sha256(source)
    report = {'backend': 'TensorRT trtexec 10.9.0', 'mode': 'engine_iteration_with_transfers',
              'input_sha256': input_hash, 'iterations_requested': args.iterations,
              'warmup_ms_requested': args.warmup_ms, 'rounds': 3,
              'flags': ['--avgRuns=1', '--duration=0'],
              'data_transfers': 'enabled; specified input file is copied to the device',
              'fp32_tf32_policy': 'TensorRT default permits TF32; --noTF32 was not passed',
              'candidates': {}, 'speedup_verified_for_full_cpp_pipeline': False,
              'final_test_used': False}
    for name, folder in ENGINES.items():
        model = ROOT / 'artifacts/metal_nut' / ('fp32' if name == 'fp32' else 'int8_selective_2_trt_compat') / 'model.onnx'
        build = read(folder / 'evidence.json')
        engine = folder / 'model.engine'
        if sha256(model) != build['model_sha256'] or sha256(engine) != build['engine_sha256']:
            raise ValueError(f'{name} model/engine hash mismatch')
        report['candidates'][name] = {'model_sha256': build['model_sha256'],
                                      'engine_sha256': build['engine_sha256'], 'runs': []}
    order = [('fp32', 'int8_selective_2'), ('int8_selective_2', 'fp32'),
             ('fp32', 'int8_selective_2')]
    for round_number, pair in enumerate(order, 1):
        for name in pair:
            folder = ENGINES[name]
            run_dir = ROOT / 'runs' / 'trt-engine-benchmark-with-transfers' / f'{name}-{round_number}'
            run_dir.mkdir(parents=True, exist_ok=True)
            engine = folder / 'model.engine'
            times = run_dir / 'times.json'
            log = run_dir / 'trtexec.log'
            command = [str(trt / 'bin/trtexec.exe'), f'--loadEngine={engine}',
                       f'--loadInputs=image:{source.relative_to(ROOT).as_posix()}',
                       f'--iterations={args.iterations}', f'--warmUp={args.warmup_ms}',
                       '--duration=0', '--avgRuns=1',
                       f'--exportTimes={times}']
            with log.open('w', encoding='utf-8', errors='replace') as handle:
                done = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                      stderr=subprocess.STDOUT, timeout=180)
            if done.returncode or not times.is_file():
                raise RuntimeError(f'{name} round {round_number} failed: {log}')
            rows = read(times)
            compute = [item['computeMs'] for item in rows]
            if len(compute) < args.iterations or not np.isfinite(compute).all():
                raise ValueError(f'{name} round {round_number} incomplete timings')
            run = {'round': round_number, 'command': command,
                   'times': str(times.relative_to(ROOT)), 'times_sha256': sha256(times),
                   'log': str(log.relative_to(ROOT)), 'log_sha256': sha256(log),
                   'gpu_compute_ms': percentiles(compute),
                   'h2d_ms': percentiles([item['h2dMs'] for item in rows]),
                   'd2h_ms': percentiles([item['d2hMs'] for item in rows]),
                   'latency_ms': percentiles([item['latencyMs'] for item in rows])}
            report['candidates'][name]['runs'].append(run)
            print(name, round_number, run['gpu_compute_ms'], flush=True)
    for name in ENGINES:
        runs = report['candidates'][name]['runs']
        report['candidates'][name]['median_round_p50_ms'] = statistics.median(
            item['gpu_compute_ms']['p50'] for item in runs)
        report['candidates'][name]['median_round_latency_p50_ms'] = statistics.median(
            item['latency_ms']['p50'] for item in runs)
    fp32 = report['candidates']['fp32']['median_round_p50_ms']
    int8 = report['candidates']['int8_selective_2']['median_round_p50_ms']
    report['engine_only_speedup_fp32_over_int8'] = fp32 / int8
    report['engine_iteration_latency_speedup_fp32_over_int8'] = (
        report['candidates']['fp32']['median_round_latency_p50_ms'] /
        report['candidates']['int8_selective_2']['median_round_latency_p50_ms'])
    save(ROOT / 'reports/trt_engine_benchmark_with_transfers.json', report)
    print('engine-only speedup', report['engine_only_speedup_fp32_over_int8'])


if __name__ == '__main__':
    main()
