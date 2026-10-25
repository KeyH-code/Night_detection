"""Prepare reversible box normalization and private visual-review candidates.

Raw annotations are never modified. Brightness is a review ranking only, never
an automatic replacement for the supplied Daytime/Night attributes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepare_soda10m import sha256, write_json


def clip_box(bbox, width, height):
    x, y, w, h = bbox
    x1, y1 = min(width, max(0, x)), min(height, max(0, y))
    x2, y2 = min(width, max(0, x + w)), min(height, max(0, y + h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Clipping would create an empty box; manual review required")
    return [x1, y1, x2 - x1, y2 - y1]


def contact_sheet(keys, lookup, raw, output, labels=None, columns=4):
    tile_w, tile_h = 384, 250
    image = Image.new("RGB", (tile_w*columns, tile_h*((len(keys)+columns-1)//columns)), "#101722")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    for index, key in enumerate(keys):
        row = lookup[key]
        x, y = index % columns * tile_w, index // columns * tile_h
        split = key.split(":", 1)[0]
        with Image.open(raw / split / row["file_name"]) as source:
            tile = source.convert("RGB").resize((384, 216), Image.Resampling.LANCZOS)
        image.paste(tile, (x, y))
        label = labels[index] if labels else f"{key} {row['period']} {row['city']}"
        draw.text((x+3, y+218), label, font=font, fill="white")
        draw.text((x+3, y+235), f"{row['location']} / {row['weather']}", font=font, fill="#b8cad9")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/soda10m"))
    args = parser.parse_args()
    root = args.data_root.resolve()
    raw = root / "raw/SSLAD-2D/labeled"
    audit = root / "audit"
    normalized = root / "normalized/annotations"
    normalized.mkdir(parents=True, exist_ok=True)
    changes, normalization = [], {}
    for split in ("train", "val"):
        source = raw / "annotations" / f"instance_{split}.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        result = deepcopy(data)
        by_id = {im["id"]: im for im in data["images"]}
        split_changes = []
        for annotation in result["annotations"]:
            im = by_id[annotation["image_id"]]
            original = annotation["bbox"]
            clipped = clip_box(original, im["width"], im["height"])
            if clipped != original:
                change = {"official_split": split, "image_id": im["id"], "annotation_id": annotation["id"],
                          "category_id": annotation["category_id"], "original_bbox": original,
                          "normalized_bbox": clipped, "original_area": annotation["area"],
                          "normalized_area": clipped[2]*clipped[3]}
                changes.append(change)
                split_changes.append(change)
                annotation["bbox"] = clipped
                annotation["area"] = clipped[2]*clipped[3]
        target = normalized / source.name
        write_json(target, result)
        normalization[split] = {
            "original_sha256": sha256(source), "normalized_sha256": sha256(target),
            "images": len(data["images"]), "annotations": len(data["annotations"]),
            "clipped_boxes": len(split_changes), "dropped_boxes": 0,
            "changed_category_ids": 0, "changed_period_labels": 0,
        }
    write_json(audit / "box_normalization_changes.json", changes)
    write_json(audit / "normalization.json", {
        "version": "clip_xyxy_to_image_bounds_v1",
        "operation": "x1,y1,x2,y2 independently clipped to [0,width] and [0,height]; COCO xywh and area recomputed",
        "raw_unchanged": True, "experiment_splits_created": False,
        "evaluation_policy": "Future project train/eval must use the same normalization version. Not the official benchmark protocol.",
        "splits": normalization,
    })
    inventory = [json.loads(line) for line in (audit / "inventory.jsonl").read_text(encoding="utf-8").splitlines()]
    lookup = {row["key"]: row for row in inventory}
    integrity = [json.loads(line) for line in (audit / "image_integrity.jsonl").read_text(encoding="utf-8").splitlines()]
    night = [row for row in integrity if lookup[row["key"]]["period"] == "Night"]
    brightest = sorted(night, key=lambda row: (-row["mean_gray"], row["key"]))[:16]
    remaining = [row for row in night if row not in brightest]
    coverage = sorted(remaining, key=lambda row: hashlib.sha256(row["key"].encode("utf-8")).hexdigest())[:8]
    candidates = brightest + coverage
    write_json(audit / "period_review_candidates.json", {
        "method": "16 brightest Night-tagged images and 8 deterministic hash-ordered coverage samples",
        "not_automatic_relabeling": True,
        "candidates": [{"key": row["key"], "mean_gray": row["mean_gray"]} for row in candidates],
    })
    output_dir = Path("artifacts/data_review")
    contact_sheet([row["key"] for row in candidates], lookup, raw, output_dir / "night_attribute_review.jpg", [f"{row['key']} Night-tag; mean gray={row['mean_gray']:.1f}" for row in candidates])
    pairs = json.loads((audit / "duplicates.json").read_text(encoding="utf-8"))["near_duplicate_candidates"]
    chosen = pairs[:6]
    if len(pairs) > 6:
        chosen += [pairs[6 + i*(len(pairs)-7)//5] for i in range(6)]
    keys, labels = [], []
    for index, pair in enumerate(chosen, 1):
        for side in ("a", "b"):
            keys.append(pair[side])
            labels.append(f"Pair {index} {pair[side]} p={pair['phash_distance']} d={pair['dhash_distance']}")
    if keys:
        contact_sheet(keys, lookup, raw, output_dir / "near_duplicate_review.jpg", labels)
    write_json(audit / "duplicate_review_selection.json", {"pairs": chosen, "method": "first six closest plus six spread through sorted candidate list"})
    print(json.dumps({"normalization": normalization, "night_review_candidates": len(candidates),
                      "duplicate_review_pairs": len(chosen)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
