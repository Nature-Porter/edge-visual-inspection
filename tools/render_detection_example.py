"""Plot an audited real detection event without running or changing the model."""
import argparse
import json

import cv2
import numpy as np

from common import ROOT, read, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/trt-cpp-validation')
    parser.add_argument('--audit', default='reports/trt-cpp-validation-quality.json')
    parser.add_argument('--frame-id', type=int, default=33)
    parser.add_argument('--output', default='reports/real-detection.png')
    args = parser.parse_args()
    run = ROOT / args.run
    audit = read(ROOT / args.audit)
    snapshot = read(run / 'snapshot.json')
    events_path = run / 'events.jsonl'
    if (audit['status'] != 'passed' or audit['events_sha256'] != sha256(events_path)
            or audit['summary_sha256'] != sha256(run / 'summary.json')
            or audit['model_sha256'] != snapshot['manifest']['onnx_sha256']
            or audit['engine_sha256'] != snapshot['engine_sha256']):
        raise ValueError('Run does not match the recorded quality audit')
    events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()]
    event = next(item for item in events if item['frame_id'] == args.frame_id)
    record = next(item for item in snapshot['source']
                  if item['source_image_id'] == event['source_image_id'])
    source = ROOT / record['path']
    if (event['status'] != 'SUCCESS' or event['artifact_error'] is not None
            or sha256(source) != record['sha256']
            or event['model_sha256'] != audit['model_sha256']):
        raise ValueError('Event or source image failed provenance checks')
    map_path = run / event['anomaly_map_path']
    heat_path = run / f"heat-{event['frame_id']}.png"
    raw_map = cv2.imread(str(map_path), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    heat = cv2.imread(str(heat_path), cv2.IMREAD_COLOR)
    if raw_map is None or image is None or heat is None or not np.isfinite(raw_map).all():
        raise ValueError('Missing or invalid image / anomaly map')
    if raw_map.shape != (256, 256) or not np.isclose(float(raw_map.max()), event['score'], atol=1e-6, rtol=0):
        raise ValueError('Anomaly map does not match the recorded image score')
    expected_heat = cv2.applyColorMap(cv2.normalize(raw_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET)
    if not np.array_equal(heat, expected_heat):
        raise ValueError('Heatmap does not match the C++ anomaly map')
    image = cv2.cvtColor(cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    relative_map = (raw_map - raw_map.min()) / max(float(np.ptp(raw_map)), 1e-12)
    alpha = (relative_map * 0.58)[..., None]
    overlay = np.uint8(np.clip(image * (1 - alpha) + heat * alpha, 0, 255))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    background, foreground, muted = '#0e1726', '#eff6ff', '#b4c6dc'
    figure, axes = plt.subplots(1, 3, figsize=(14, 6.2), dpi=150, facecolor=background)
    figure.subplots_adjust(left=0.025, right=0.975, top=0.76, bottom=0.19, wspace=0.055)
    figure.text(0.03, 0.935, 'REAL MODEL DETECTION', color=foreground, fontsize=22, weight='bold')
    sample_name = '/'.join(source.parts[-3:])
    figure.text(0.03, 0.878, f'MVTec AD / metal_nut / {sample_name}  |  EfficientAD-S / TensorRT INT8',
                color=muted, fontsize=12)
    for axis, pixels, title in zip(axes, [image, heat, overlay],
                                   ['Input image', 'Model anomaly map', 'Heatmap overlay']):
        axis.imshow(pixels)
        axis.set_title(title, color=foreground, fontsize=14, pad=12)
        axis.axis('off')
    figure.text(0.03, 0.115,
                f"Decision: {event['decision']}    |    Score: {event['score']:.6f}    |    Image threshold: {event['image_threshold']:.6f}",
                color='#6ee7cf', fontsize=13, weight='bold')
    figure.text(0.03, 0.065, f"C++ event #{event['frame_id']} / project validation split. Heatmap scaled per image for display; no pixel threshold.",
                color=muted, fontsize=10)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor=background)
    plt.close(figure)
    metadata = {
        'source': 'MVTec AD metal_nut; see data/README.md for dataset provenance and terms',
        'split': 'validation', 'independent_final_test': False,
        'source_image': str(source.relative_to(ROOT)).replace('\\', '/'),
        'source_image_sha256': sha256(source), 'run': args.run,
        'audit': args.audit, 'audit_sha256': sha256(ROOT / args.audit),
        'event': event, 'events_sha256': sha256(events_path),
        'anomaly_map_sha256': sha256(map_path), 'heatmap_sha256': sha256(heat_path),
        'output_sha256': sha256(output),
        'visualization': 'Existing C++ heatmap; per-image min/max scaling; alpha overlay; no pixel-level acceptance claim',
        'generator': 'tools/render_detection_example.py',
    }
    output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8', newline='\n')
    print(f'Rendered audited event {event["frame_id"]}: {output}')


if __name__ == '__main__':
    main()
