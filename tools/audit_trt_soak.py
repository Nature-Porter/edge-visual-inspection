"""Audit completed INT8 C++ soak without counting repeated frames as quality data."""
import json

from common import ROOT, read, save, sha256


def main():
    run = ROOT / 'runs/trt-cpp-soak'
    snapshot = read(run / 'snapshot.json')
    summary = read(run / 'summary.json')
    schema_path = ROOT / 'reports/trt-cpp-soak-schema.json'
    schema = read(schema_path)
    route_path = ROOT / 'reports/trt_verified_selective_2.json'
    route = read(route_path)
    route_sha = sha256(route_path)
    events_path = run / 'events.jsonl'
    checks = {
        'schema': schema['status'] == 'passed' and schema['event_count'] == 36001
                  and schema['events_sha256'] == sha256(events_path)
                  and schema['summary_sha256'] == sha256(run / 'summary.json'),
        'engine': snapshot['args']['--backend'] == 'trt-int8'
                  and snapshot['engine_sha256'] == route['engine_sha256']
                  and snapshot['engine_evidence_sha256'] == route_sha
                  and snapshot['manifest']['onnx_sha256'] == route['model_sha256'],
        'coverage': summary['generated'] == summary['completed'] == summary['successful_inference'] == 36001,
        'duration': summary['elapsed_seconds'] >= 1800,
        'failures': summary['error'] == summary['queue_overflow'] == 0 and summary['exit_code'] == 0,
        'latency': summary['decision_latency_ms']['p95'] <= 40 and summary['timeout'] / summary['generated'] <= 0.01,
        'gpu_timing': summary['model_gpu_ms']['count'] == summary['completed']
                      and summary['model_gpu_ms']['p50'] > 0,
    }
    identities = 0
    with events_path.open(encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            identities += (event['requested_backend'] == 'trt-int8'
                           and event['model_sha256'] == route['model_sha256']
                           and event['verified_precision_evidence_id'] == route_sha
                           and event['observed_provider_summary'] == 'TensorRT engine direct CUDA runtime; no ORT EP')
    checks['event_identity'] = identities == summary['completed']
    output = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
              'run': 'runs/trt-cpp-soak', 'event_count': identities,
              'repeated_replay_not_quality_samples': True, 'quality_sample_count_added': 0,
              'summary_sha256': sha256(run / 'summary.json'), 'events_sha256': sha256(events_path),
              'schema_audit_sha256': sha256(schema_path), 'engine_sha256': route['engine_sha256'],
              'model_sha256': route['model_sha256'], 'evidence_sha256': route_sha,
              'elapsed_seconds': summary['elapsed_seconds'],
              'decision_p95_ms': summary['decision_latency_ms']['p95'],
              'timeout_count': summary['timeout'], 'timeout_rate': summary['timeout'] / summary['generated'],
              'process_gpu_memory_bytes': None,
              'memory_note': 'WDDM process VRAM unavailable; no working-set samples recorded for this INT8 soak'}
    save(ROOT / 'reports/trt-cpp-soak-audit.json', output)
    print(output['status'], output['event_count'], output['decision_p95_ms'])
    return 0 if output['status'] == 'passed' else 5


if __name__ == '__main__':
    raise SystemExit(main())
