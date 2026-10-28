"""Three paired C++ TensorRT pipeline benchmarks using existing local engines."""
import argparse
import json
import statistics
import subprocess

from common import ROOT, read, save, sha256, verified_records
from integration_tests import runtime_env


CASES = {
    'fp32': ('artifacts/metal_nut/fp32/model_manifest.json',
             'runs/trt-detailed-fp32/model.engine',
             'runs/trt-detailed-fp32/evidence.json', 'trt-fp32'),
    'int8': ('artifacts/metal_nut/int8_selective_2_trt_compat/model_manifest.json',
             'runs/trt-detailed-int8_selective_2_trt_compat/model.engine',
             'reports/trt_verified_selective_2.json', 'trt-int8'),
}


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, int(len(values) * fraction + 0.999999) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 100 or args.warmup < 10:
        raise ValueError('minimum 100 iterations and 10 warmups')
    verified_records('validation')
    env = runtime_env()
    env['PATH'] = str(ROOT / 'third_party/TensorRT-10.9.0.34/lib') + ';' + env['PATH']
    root = ROOT / 'runs/trt-cpp-pipeline-benchmark'
    root.mkdir(parents=True, exist_ok=True)
    rounds = {'fp32': [], 'int8': []}
    for index, order in enumerate((('fp32', 'int8'), ('int8', 'fp32'), ('fp32', 'int8')), 1):
        for name in order:
            manifest, engine, evidence, backend = CASES[name]
            run = root / f'round-{index}-{name}'
            if run.exists():
                raise FileExistsError(f'benchmark output exists: {run}')
            command = [str(ROOT / 'build/inspection.exe'), 'benchmark', '--manifest', manifest,
                       '--source', 'data/validation.json', '--backend', backend,
                       '--engine', engine, '--engine-evidence', evidence,
                       '--postprocess', 'cuda', '--mode', 'pipeline', '--warmup', str(args.warmup),
                       '--iterations', str(args.iterations), '--output', str(run)]
            log_path = root / f'round-{index}-{name}.log'
            with log_path.open('w', encoding='utf-8', errors='replace') as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=180)
            if completed.returncode:
                raise RuntimeError(f'{name} round {index} exit {completed.returncode}: {log_path}')
            snapshot = read(run / 'snapshot.json')
            summary = read(run / 'summary.json')
            events_path = run / 'benchmark.jsonl'
            rows = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()]
            model = read(ROOT / manifest)
            engine_evidence = read(ROOT / evidence)
            if (len(rows) != args.iterations or summary['mode'] != 'pipeline'
                    or summary['backend'] != backend or summary['quality_claim'] is not False
                    or snapshot['source'] != read(ROOT / 'data/validation.json')['records']
                    or snapshot['manifest_sha256'] != sha256(ROOT / manifest)
                    or snapshot['engine_sha256'] != sha256(ROOT / engine)
                    or snapshot['engine_sha256'] != engine_evidence['engine_sha256']
                    or model['onnx_sha256'] != engine_evidence['model_sha256']
                    or any(row['iteration'] != i or row['status'] != 'SUCCESS'
                           or row['model_gpu_ms'] is None or row['model_gpu_ms'] <= 0
                           for i, row in enumerate(rows))):
                raise ValueError(f'{name} round {index} benchmark evidence mismatch')
            rounds[name].append({'round': index, 'command': command,
                                 'summary': str((run / 'summary.json').relative_to(ROOT)),
                                 'summary_sha256': sha256(run / 'summary.json'),
                                 'raw': str(events_path.relative_to(ROOT)),
                                 'raw_sha256': sha256(events_path), 'log_sha256': sha256(log_path),
                                 'wall_p50_ms': percentile([row['wall_ms'] for row in rows], .5),
                                 'wall_p95_ms': percentile([row['wall_ms'] for row in rows], .95),
                                 'gpu_p50_ms': percentile([row['model_gpu_ms'] for row in rows], .5),
                                 'gpu_p95_ms': percentile([row['model_gpu_ms'] for row in rows], .95)})
            print(index, name, rounds[name][-1]['wall_p50_ms'], flush=True)
    median = {name: {'wall_p50_ms': statistics.median(r['wall_p50_ms'] for r in rows),
                     'gpu_p50_ms': statistics.median(r['gpu_p50_ms'] for r in rows)}
              for name, rows in rounds.items()}
    report = {'status': 'measured', 'backend_family': 'direct TensorRT C++',
              'mode': 'pipeline (preprocess, transfer, inference, CUDA postprocess; no scheduler/decode)',
              'iterations_per_round': args.iterations, 'warmups_per_round': args.warmup,
              'rounds': rounds, 'median_of_round_p50': median,
              'wall_ratio_fp32_over_int8': median['fp32']['wall_p50_ms'] / median['int8']['wall_p50_ms'],
              'gpu_ratio_fp32_over_int8': median['fp32']['gpu_p50_ms'] / median['int8']['gpu_p50_ms'],
              'same_validation_image': read(ROOT / 'data/validation.json')['records'][0]['source_image_id'],
              'validation_sha256': sha256(ROOT / 'data/validation.json'),
              'model_sha256': {name: read(ROOT / case[0])['onnx_sha256'] for name, case in CASES.items()},
              'engine_sha256': {name: sha256(ROOT / case[1]) for name, case in CASES.items()},
              'fp32_tf32_policy': 'TensorRT default permits TF32; --noTF32 was not used in engine build',
              'not_decision_latency': True, 'final_test_used': False}
    save(ROOT / 'reports/trt_cpp_pipeline_benchmark.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
