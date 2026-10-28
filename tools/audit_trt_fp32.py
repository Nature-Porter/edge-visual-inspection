"""Record validation quality of the FP32 TensorRT timing comparator."""
import json

from common import ROOT, read, save, sha256, verified_records
from metrics import join_predictions, metrics


def main():
    run = ROOT / 'runs/trt-cpp-fp32-validation'
    schema_path = ROOT / 'reports/trt-cpp-fp32-validation-schema.json'
    schema = read(schema_path)
    summary = read(run / 'summary.json')
    snapshot = read(run / 'snapshot.json')
    manifest_path = ROOT / 'artifacts/metal_nut/fp32/model_manifest.json'
    manifest = read(manifest_path)
    engine_path = ROOT / 'runs/trt-detailed-fp32/model.engine'
    evidence_path = ROOT / 'runs/trt-detailed-fp32/evidence.json'
    evidence = read(evidence_path)
    records = verified_records('validation')
    events_path = run / 'events.jsonl'
    events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()]
    predictions = [{'source_image_id': item['source_image_id'], 'score': item['score']}
                   for item in events if item['status'] == 'SUCCESS']
    valid, coverage = join_predictions(records, predictions)
    result = metrics(*zip(*valid), manifest['image_threshold']) if coverage['ERROR'] == 0 else None
    baseline = read(ROOT / 'artifacts/metal_nut/fp32/validation_quality.json')['metrics']
    source = {item['source_image_id']: item['score']
              for item in read(ROOT / 'artifacts/metal_nut/fp32/validation_predictions.json')}
    differences = [abs(item['score'] - source[item['source_image_id']]) for item in events]
    checks = {
        'schema': schema['status'] == 'passed' and schema['events_sha256'] == sha256(events_path),
        'coverage': summary['generated'] == summary['completed'] == 69 and len(events) == 69
                    and all(item['status'] == 'SUCCESS' for item in events)
                    and coverage['valid'] == 69 and coverage['ERROR'] == 0,
        'identity': snapshot['args']['--backend'] == 'trt-fp32'
                    and snapshot['manifest_sha256'] == sha256(manifest_path)
                    and snapshot['engine_sha256'] == sha256(engine_path) == evidence['engine_sha256']
                    and snapshot['engine_evidence_sha256'] == sha256(evidence_path)
                    and manifest['onnx_sha256'] == evidence['model_sha256'],
    }
    # Frozen ORT threshold is used; a different backend need not retain the same quality.
    quality_gate = (result is not None and baseline['AUROC'] - result['AUROC'] <= 0.01
                    and result['FNR'] - baseline['FNR'] <= 0.02)
    report = {'status': 'measured' if all(checks.values()) else 'invalid',
              'checks': checks, 'quality_gate_pass': quality_gate,
              'timing_comparator_only': not quality_gate,
              'metrics': result, 'baseline_ort_cuda_metrics': baseline,
              'max_score_abs_vs_ort_cuda': max(differences),
              'threshold_recalibrated': False, 'final_test_used': False,
              'events_sha256': sha256(events_path), 'summary_sha256': sha256(run / 'summary.json'),
              'schema_audit_sha256': sha256(schema_path), 'engine_sha256': sha256(engine_path),
              'model_sha256': manifest['onnx_sha256'], 'validation_sha256': sha256(ROOT / 'data/validation.json')}
    save(ROOT / 'reports/trt-cpp-fp32-validation-quality.json', report)
    print(report['status'], report['metrics'], 'quality_gate_pass', quality_gate)
    return 0 if all(checks.values()) else 5


if __name__ == '__main__':
    raise SystemExit(main())
