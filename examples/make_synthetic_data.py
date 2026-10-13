"""Generate artificial multiview images to exercise file layout; no patient data."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate(root, patients_per_grade=10, seed=42):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Choose an empty destination: {root}')
    images = root / 'images'
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for grade, stenosis in enumerate([0, 12, 37, 60, 85, 100]):
        for index in range(patients_per_grade):
            patient = f'SYN{grade}{index:04d}'
            for vessel in ['LAD', 'RCA']:
                record = f'{patient}_{vessel}'
                rows.append({'record_id': record, 'patient_id': patient,
                             'stenosis_percentage': stenosis})
                for view in range(4):
                    pixels = rng.integers(0, 45, (64, 48), dtype=np.uint8)
                    image = Image.fromarray(pixels)
                    draw = ImageDraw.Draw(image)
                    draw.line([(24 + view, 0), (19, 32), (25, 63)], fill=180, width=5)
                    image.save(images / f'{record}_cpr_view{view}.png')
    with (root / 'metadata.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root / 'metadata.csv', images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/synthetic')
    args = parser.parse_args()
    metadata, images = generate(args.output)
    print(f'Artificial metadata: {metadata}\nArtificial images: {images}')
