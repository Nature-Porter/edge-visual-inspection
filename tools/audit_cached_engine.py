"""Audit the metadata available in an already cached TensorRT engine export."""
import argparse
from pathlib import Path

from common import ROOT, read, save, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', default='reports/trt_int8_evidence.json')
    parser.add_argument('--output', default='reports/trt_layer_metadata_audit.json')
    args = parser.parse_args()
    pointer = read(ROOT / args.evidence)
    evidence_path = Path(pointer['run_dir']) / 'evidence.json'
    if sha256(evidence_path) != pointer['evidence_sha256']:
        raise ValueError('target evidence hash mismatch')
    evidence = read(evidence_path)
    inspection = evidence['engine_inspection']
    layer_path = Path(inspection['file'])
    if sha256(layer_path) != inspection['sha256']:
        raise ValueError('layer export hash mismatch')
    layer_document = read(layer_path)
    layers = layer_document['Layers'] if isinstance(layer_document, dict) else layer_document
    structured = [layer for layer in layers if isinstance(layer, dict)]
    precision = [layer for layer in structured if layer.get('Precision')]
    result = {
        'model_sha256': evidence['key']['model_sha256'],
        'evidence_file': str(evidence_path),
        'evidence_sha256': sha256(evidence_path),
        'engine_files': evidence.get('cached_engines', []),
        'layer_export': str(layer_path),
        'layer_export_sha256': sha256(layer_path),
        'layer_count': len(layers),
        'structured_layer_count': len(structured),
        'layers_with_precision_count': len(precision),
        'int8_execution_verified': False,
        'reason': 'cached engine layer export has no per-layer precision metadata' if not precision else 'precision requires separate computation-layer audit',
    }
    save(ROOT / args.output, result)
    print(result['reason'], f"({len(layers)} layers)")


if __name__ == '__main__':
    main()
