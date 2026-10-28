"""Fail-closed validation audit for the C++ TensorRT path; validation split only."""
import argparse
import json
import math

from common import ROOT, config, read, save, sha256, verified_records
from metrics import join_predictions, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='runs/trt-cpp-validation')
    parser.add_argument('--schema', default='reports/trt-cpp-validation-schema.json')
    parser.add_argument('--cpu-post-run', default='runs/trt-cpp-validation-cpu-post')
    parser.add_argument('--output', default='reports/trt-cpp-validation-quality.json')
    args = parser.parse_args()
    run = ROOT / args.run
    snapshot = read(run / 'snapshot.json')
    summary = read(run / 'summary.json')
    schema = read(ROOT / args.schema)
    route_path = ROOT / 'reports/trt_verified_selective_2.json'
    route = read(route_path)
    reference = read(ROOT / 'reports/trt_direct_validation_quality_selective_2.json')
    manifest = read(ROOT / 'artifacts/metal_nut/int8_selective_2_trt_compat/model_manifest.json')
    records = verified_records('validation')
    expected = {r['source_image_id']: r for r in records}
    events_path = run / 'events.jsonl'
    events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()]
    direct = {p['source_image_id']: p for p in reference['predictions']}
    cpu_run = ROOT / args.cpu_post_run
    cpu_events_path = cpu_run / 'events.jsonl'
    cpu_events = [json.loads(line) for line in cpu_events_path.read_text(encoding='utf-8').splitlines()]
    cpu_by_id = {item['source_image_id']: item for item in cpu_events}
    cpu_schema_path = ROOT / 'reports/trt-cpp-validation-cpu-post-schema.json'
    cpu_schema = read(cpu_schema_path)
    errors = []
    require = lambda ok, message: errors.append(message) if not ok else None
    require(schema['status'] == 'passed' and schema['events_sha256'] == sha256(events_path)
            and schema['summary_sha256'] == sha256(run / 'summary.json'), 'schema audit mismatch')
    require(snapshot['manifest_sha256'] == sha256(ROOT / 'artifacts/metal_nut/int8_selective_2_trt_compat/model_manifest.json'), 'manifest hash mismatch')
    require(snapshot['manifest'] == manifest and snapshot['source'] == records, 'snapshot input mismatch')
    require(snapshot['args']['--backend'] == 'trt-int8' and snapshot['engine_sha256'] == route['engine_sha256']
            and snapshot['engine_evidence_sha256'] == sha256(route_path), 'engine identity mismatch')
    require(route['int8_execution_verified'] and route['validation_quality_pass']
            and route['model_sha256'] == manifest['onnx_sha256']
            and not route['final_test_used'], 'INT8 provenance mismatch')
    require(len(events) == len(records) == 69 and summary['generated'] == summary['completed'] == 69
            and summary['error'] == summary['timeout'] == summary['queue_overflow'] == 0, 'coverage or runtime failure')
    require(len(direct) == len(records) and reference['coverage']['ERROR'] == 0, 'direct engine reference incomplete')
    require(len(cpu_events) == len(cpu_by_id) == len(records), 'CPU postprocess coverage mismatch')
    require(cpu_schema['status'] == 'passed' and cpu_schema['events_sha256'] == sha256(cpu_events_path)
            and cpu_schema['summary_sha256'] == sha256(cpu_run / 'summary.json'), 'CPU postprocess schema mismatch')
    require(read(cpu_run / 'snapshot.json')['engine_sha256'] == route['engine_sha256'], 'CPU postprocess engine mismatch')
    predictions = []
    differences = []
    decision_agreement = 0
    for event in events:
        source_id = event['source_image_id']
        if source_id not in expected or source_id not in direct:
            errors.append(f'unknown source {source_id}')
            continue
        score = event['score']
        if event['status'] != 'SUCCESS' or score is None or not math.isfinite(score):
            errors.append(f'bad status or score {source_id}')
            continue
        if (event['requested_backend'] != 'trt-int8' or event['model_sha256'] != manifest['onnx_sha256']
                or event['verified_precision_evidence_id'] != sha256(route_path)
                or event['image_threshold'] != manifest['image_threshold']
                or event['model_gpu_ms'] is None or event['model_gpu_ms'] <= 0):
            errors.append(f'event backend/precision/timing contract {source_id}')
        expected_decision = 'REVIEW' if direct[source_id]['score'] >= manifest['image_threshold'] else 'OK'
        decision_agreement += event['decision'] == expected_decision
        cpu_event = cpu_by_id.get(source_id)
        if not cpu_event or cpu_event['status'] != event['status'] or cpu_event['decision'] != event['decision'] or cpu_event['score'] != score:
            errors.append(f'CUDA/CPU postprocess mismatch {source_id}')
        differences.append(abs(score - direct[source_id]['score']))
        predictions.append({'source_image_id': source_id, 'score': score})
    valid, coverage = join_predictions(records, predictions)
    result = metrics(*zip(*valid), manifest['image_threshold']) if not coverage['ERROR'] else None
    baseline = read(ROOT / 'artifacts/metal_nut/fp32/validation_quality.json')['metrics']
    gates = config(ROOT / 'configs/metal_nut.yaml')['quality']
    quality_pass = (result is not None and baseline['AUROC'] - result['AUROC'] <= gates['max_auroc_drop']
                    and result['FNR'] - baseline['FNR'] <= gates['max_fnr_increase'])
    require(coverage['ERROR'] == 0 and coverage['valid'] == 69, 'unique image coverage failed')
    require(decision_agreement == 69, 'C++/direct engine decisions disagree')
    require(differences and max(differences) <= 1e-4, 'C++/direct engine score difference too large')
    require(quality_pass, 'frozen validation quality gate failed')
    report = {'status': 'passed' if not errors else 'failed', 'run': args.run,
              'events_sha256': sha256(events_path), 'summary_sha256': sha256(run / 'summary.json'),
              'cpu_post_events_sha256': sha256(cpu_events_path),
              'cpu_post_schema_audit_sha256': sha256(cpu_schema_path),
              'schema_audit_sha256': sha256(ROOT / args.schema), 'engine_sha256': route['engine_sha256'],
              'model_sha256': manifest['onnx_sha256'], 'evidence_sha256': sha256(route_path),
              'validation_sha256': sha256(ROOT / 'data/validation.json'), 'coverage': coverage,
              'metrics': result, 'quality_gate_pass': quality_pass,
              'max_score_abs_vs_direct_trt': max(differences) if differences else None,
              'decision_agreement': decision_agreement, 'final_test_used': False, 'errors': errors}
    save(ROOT / args.output, report)
    print(report['status'], report['metrics'], 'max_abs', report['max_score_abs_vs_direct_trt'])
    return 0 if not errors else 5


if __name__ == '__main__':
    raise SystemExit(main())
