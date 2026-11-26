"""Run single-image CPU detection with a trusted local Ultralytics checkpoint."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=ROOT / 'examples/bus.jpg')
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights/yolo11n.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/demo/bus.png')
    options = parser.parse_args()
    if not options.image.is_file():
        parser.error(f'image not found: {options.image}')
    if options.weights.suffix.lower() != '.pt' or not options.weights.is_file():
        parser.error(f'expected a trusted local .pt checkpoint at {options.weights}; see README')
    if options.output.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        parser.error('output must end in .jpg, .jpeg or .png')
    if options.output.resolve() in {options.image.resolve(), options.weights.resolve()}:
        parser.error('output must differ from the input image and weights')

    from PIL import Image, ImageOps

    try:
        with Image.open(options.image) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
    except OSError as error:
        parser.error(f'cannot read input image: {error}')

    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache/ultralytics'))
    from ultralytics import YOLO

    model = YOLO(str(options.weights))
    if model.task != 'detect':
        parser.error('expected an object detection checkpoint')
    (prediction,) = model.predict(source=image, device='cpu', imgsz=640,
                                  conf=0.25, save=False, verbose=False)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.plot(pil=True).save(options.output)
    counts = Counter(prediction.names[int(index)] for index in prediction.boxes.cls.tolist())
    summary = ', '.join(f'{name} ×{count}' for name, count in sorted(counts.items()))
    count = len(prediction.boxes)
    print(f'{count} detections ({summary})' if summary else '0 detections')
    print(f'Saved {options.output}')


if __name__ == '__main__':
    main()
