"""Cross-check C++ timing records against validation quality and engine identities."""
import json
import statistics

from common import ROOT, read, save, sha256


def main():
    source_path = ROOT / 'reports/trt_cpp_pipeline_benchmark.json'
    source = read(source_path)
    fp_quality = read(ROOT / 'reports/trt-cpp-fp32-validation-quality.json')
    int8_quality = read(ROOT / 'reports/trt-cpp-validation-quality.json')
    errors = []
    if source['status'] != 'measured' or source['iterations_per_round'] != 1000:
        errors.append('benchmark protocol mismatch')
    for name in ('fp32', 'int8'):
        rounds = source['rounds'][name]
        if len(rounds) != 3:
            errors.append(f'{name} round count')
        for item in rounds:
            summary_path = ROOT / item['summary']
            raw_path = ROOT / item['raw']
            if sha256(summary_path) != item['summary_sha256'] or sha256(raw_path) != item['raw_sha256']:
                errors.append(f'{name} round {item["round"]} hash mismatch')
                continue
            rows = [json.loads(line) for line in raw_path.read_text(encoding='utf-8').splitlines()]
            if len(rows) != source['iterations_per_round'] or any(row['status'] != 'SUCCESS' for row in rows):
                errors.append(f'{name} round {item["round"]} invalid rows')
        if rounds and source['median_of_round_p50'][name]['wall_p50_ms'] != statistics.median(
                item['wall_p50_ms'] for item in rounds):
            errors.append(f'{name} median mismatch')
    if fp_quality['status'] != 'measured' or int8_quality['status'] != 'passed':
        errors.append('quality audit missing')
    if fp_quality['engine_sha256'] != source['engine_sha256']['fp32'] or int8_quality['engine_sha256'] != source['engine_sha256']['int8']:
        errors.append('quality engine differs from timed engine')
    report = {'status': 'passed' if not errors else 'failed', 'errors': errors,
              'benchmark_sha256': sha256(source_path),
              'fp32_quality_sha256': sha256(ROOT / 'reports/trt-cpp-fp32-validation-quality.json'),
              'int8_quality_sha256': sha256(ROOT / 'reports/trt-cpp-validation-quality.json'),
              'cpp_pipeline_wall_ratio_fp32_over_int8': source['wall_ratio_fp32_over_int8'],
              'cpp_model_gpu_ratio_fp32_over_int8': source['gpu_ratio_fp32_over_int8'],
              'fp32_comparator_quality_pass': fp_quality['quality_gate_pass'],
              'int8_validation_quality_pass': int8_quality['quality_gate_pass'],
              'both_routes_quality_pass': fp_quality['quality_gate_pass'] and int8_quality['quality_gate_pass'],
              'decision_latency_speedup_claim': False,
              'scope': 'C++ benchmark pipeline without scheduler or decode; FP32 TensorRT comparator fails frozen FNR gate'}
    save(ROOT / 'reports/trt-cpp-benchmark-audit.json', report)
    print(report['status'], report['cpp_pipeline_wall_ratio_fp32_over_int8'],
          'both_quality_pass', report['both_routes_quality_pass'])
    return 0 if not errors else 5


if __name__ == '__main__':
    raise SystemExit(main())
