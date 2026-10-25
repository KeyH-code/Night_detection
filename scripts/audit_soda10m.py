"""Audit original SODA10M train/val without altering images or annotations.

Detailed image-level files and visual previews are private under data/ and
artifacts/. The shareable report contains aggregate statistics only.
No split is generated without an authoritative acquisition-group mapping.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import PIL

from prepare_soda10m import sha256, write_json

ATTRIBUTES = ("city", "location", "period", "weather")
GROUP_FIELDS = ("sequence_id", "sequence", "video_id", "video", "trip_id",
                "capture_group", "vehicle_id", "timestamp", "date", "camera_id")
EXPECTED_CATEGORIES = {1: "Pedestrian", 2: "Cyclist", 3: "Car", 4: "Truck",
                       5: "Tram", 6: "Tricycle"}
RESAMPLE = Image.Resampling.LANCZOS
COS = np.cos(np.pi / 32 * (np.arange(32) + 0.5)[None, :] * np.arange(8)[:, None])


def bit_hash(values) -> int:
    result = 0
    for flag in values:
        result = (result << 1) | bool(flag)
    return result


def inspect_image(task):
    path, record = task
    result = {"key": record["key"], "file_name": record["file_name"]}
    try:
        result["bytes"] = path.stat().st_size
        result["sha256"] = sha256(path)
        with Image.open(path) as image:
            image.load()  # Full JPEG decoding, with truncated-image loading disabled.
            result["actual_size"] = list(image.size)
            result["mode"] = image.mode
            result["format"] = image.format
            result["dimension_matches"] = image.size == (record["width"], record["height"])
            rgb = image.convert("RGB")
            result["decoded_rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
            gray = ImageOps.grayscale(rgb)
            small = np.asarray(gray.resize((32, 32), RESAMPLE), dtype=np.float64)
            coefficients = COS @ small @ COS.T
            bits = coefficients > np.median(coefficients.ravel()[1:])
            bits[0, 0] = False  # Ignore the DC brightness component.
            result["phash"] = f"{bit_hash(bits.ravel()):016x}"
            dh = np.asarray(gray.resize((9, 8), RESAMPLE), dtype=np.int16)
            result["dhash"] = f"{bit_hash((dh[:, 1:] > dh[:, :-1]).ravel()):016x}"
            result["mean_gray"] = round(float(small.mean()), 3)
            result["exif_keys"] = [str(k) for k in image.getexif()]
        result["ok"] = True
    except Exception as exc:
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return result


def duplicate_groups(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if row.get("ok"):
            groups[row[field]].append(row["key"])
    return [v for v in groups.values() if len(v) > 1]


def near_duplicate_pairs(rows):
    """Exhaustive hash screening, not proof of sequence identity or independence."""
    valid = [r for r in rows if r.get("ok")]
    hashes = np.array([int(r["phash"], 16) for r in valid], dtype=np.uint64)
    dhashes = np.array([int(r["dhash"], 16) for r in valid], dtype=np.uint64)
    pairs = []
    for i, row in enumerate(valid[:-1]):
        distances = np.bitwise_count(hashes[i + 1:] ^ hashes[i])
        candidates = np.flatnonzero(distances <= 6) + i + 1
        for j in candidates.tolist():
            dhdist = int((int(dhashes[i]) ^ int(dhashes[j])).bit_count())
            if dhdist <= 10:
                pairs.append({"a": row["key"], "b": valid[j]["key"],
                              "phash_distance": int(distances[j-i-1]),
                              "dhash_distance": dhdist,
                              "byte_identical": row["sha256"] == valid[j]["sha256"],
                              "decoded_identical": row["decoded_rgb_sha256"] == valid[j]["decoded_rgb_sha256"]})
    return sorted(pairs, key=lambda p: (p["phash_distance"], p["dhash_distance"], p["a"], p["b"]))


def make_preview(records, annotations, categories, raw, output):
    """Deterministic attribute coverage preview; does not use model predictions."""
    chosen = []
    # One representative per observed source/period/location group, bounded to 18.
    groups = defaultdict(list)
    for row in records:
        groups[(row["official_split"], row["period"], row["location"])].append(row)
    for _, values in sorted(groups.items()):
        values = sorted(values, key=lambda r: hashlib.sha256(r["key"].encode("utf-8")).hexdigest())
        chosen.extend(values[:2])
    chosen = chosen[:18]
    width, height = 640, 410
    sheet = Image.new("RGB", (width * 3, height * math.ceil(len(chosen)/3)), "#101722")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    palette = {1: "#ffda51", 2: "#64e89c", 3: "#4fc3ff", 4: "#ff9a55", 5: "#d6a5ff", 6: "#ff75af"}
    for index, row in enumerate(chosen):
        x, y = (index % 3)*width, (index//3)*height
        with Image.open(raw / row["official_split"] / row["file_name"]) as image:
            thumbnail = image.convert("RGB").resize((640, 360), RESAMPLE)
        td = ImageDraw.Draw(thumbnail)
        for ann in annotations[row["key"]]:
            bx, by, bw, bh = ann["bbox"]
            sx, sy = 640/row["width"], 360/row["height"]
            td.rectangle((bx*sx, by*sy, (bx+bw)*sx, (by+bh)*sy), outline=palette[ann["category_id"]], width=2)
        sheet.paste(thumbnail, (x, y))
        draw.text((x+8, y+362), f"{row['key']}  {row['period']}  {row['location']}", font=font, fill="white")
        draw.text((x+8, y+385), f"{row['city']} / {row['weather']}  boxes={len(annotations[row['key']])}", font=font, fill="#bbcada")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return [r["key"] for r in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/soda10m"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.data_root.resolve()
    raw = root / "raw/SSLAD-2D/labeled"
    audit = root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records, annotations = [], defaultdict(list)
    summaries, issues, json_hashes = {}, [], {}
    image_lookup = {}
    all_category_maps = []
    for split in ("train", "val"):
        path = raw / "annotations" / f"instance_{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        json_hashes[split] = sha256(path)
        categories = {c["id"]: c["name"] for c in data["categories"]}
        all_category_maps.append(categories)
        if categories != EXPECTED_CATEGORIES:
            issues.append({"split": split, "type": "unexpected_categories", "actual": categories})
        images = data["images"]
        ids = [im["id"] for im in images]
        if len(ids) != len(set(ids)):
            issues.append({"split": split, "type": "duplicate_image_ids"})
        filenames = [im["file_name"] for im in images]
        if len(filenames) != len(set(filenames)):
            issues.append({"split": split, "type": "duplicate_filenames"})
        by_id = {}
        for im in images:
            row = dict(im, official_split=split, key=f"{split}:{im['id']}")
            by_id[im["id"]] = row
            records.append(row)
            image_lookup[row["key"]] = row
        box_counts = Counter()
        box_counts_by_period = defaultdict(Counter)
        size_counts = defaultdict(Counter)
        seen_ids, seen_boxes = set(), set()
        for ann in data["annotations"]:
            if ann["id"] in seen_ids:
                issues.append({"split": split, "type": "duplicate_annotation_id", "id": ann["id"]})
            seen_ids.add(ann["id"])
            im = by_id.get(ann["image_id"])
            if not im:
                issues.append({"split": split, "type": "orphan_annotation", "annotation_id": ann["id"]})
                continue
            category = categories.get(ann["category_id"])
            if category is None:
                issues.append({"split": split, "type": "unknown_category", "annotation_id": ann["id"]})
                continue
            bbox = ann.get("bbox", [])
            if len(bbox) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
                issues.append({"split": split, "type": "invalid_bbox", "annotation_id": ann["id"]})
                continue
            x, y, w, h = bbox
            for condition, kind in [
                (w <= 0 or h <= 0, "nonpositive_box"),
                (x < 0 or y < 0 or x+w > im["width"] or y+h > im["height"], "box_out_of_bounds"),
                (not math.isclose(ann.get("area", -1), w*h, rel_tol=1e-6, abs_tol=1e-6), "area_mismatch"),
                (ann.get("iscrowd") != 0, "nonzero_or_missing_iscrowd"),
            ]:
                if condition:
                    issues.append({"split": split, "type": kind, "annotation_id": ann["id"], "image_key": im["key"], "bbox": bbox})
            identity = (ann["image_id"], ann["category_id"], *bbox)
            if identity in seen_boxes:
                issues.append({"split": split, "type": "duplicate_same_class_box", "annotation_id": ann["id"], "image_key": im["key"]})
            seen_boxes.add(identity)
            annotations[im["key"]].append(ann)
            box_counts[category] += 1
            box_counts_by_period[im["period"]][category] += 1
            area = w*h
            size_counts[category]["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
        expected_files = set(filenames)
        actual_files = {p.name for p in (raw / split).iterdir() if p.is_file()}
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        for name in missing:
            issues.append({"split": split, "type": "missing_image", "file_name": name})
        for name in extra:
            issues.append({"split": split, "type": "unreferenced_file", "file_name": name})
        suffix_attributes = defaultdict(set)
        suffix_counts = Counter()
        for im in images:
            suffix = "_".join(Path(im["file_name"]).stem.split("_")[-2:])
            suffix_counts[suffix] += 1
            suffix_attributes[suffix].add(tuple(im[k] for k in ATTRIBUTES))
        summaries[split] = {
            "images": len(images), "boxes": len(data["annotations"]),
            "categories": categories, "boxes_by_class": dict(box_counts),
            "boxes_by_period": {k: dict(v) for k, v in box_counts_by_period.items()},
            "box_size_by_class_original_pixels": {k: dict(v) for k, v in size_counts.items()},
            "image_attributes": {k: dict(Counter(im.get(k, "MISSING") for im in images)) for k in ATTRIBUTES},
            "image_fields": dict(Counter(k for im in images for k in im)),
            "explicit_group_fields": {k: sum(k in im for im in images) for k in GROUP_FIELDS},
            "empty_annotation_images": sum(not annotations[f"{split}:{im['id']}"] for im in images),
            "images_with_class": {name: sum(any(a["category_id"] == cid for a in annotations[f"{split}:{im['id']}"]) for im in images) for cid, name in categories.items()},
            "night_images_with_class": {name: sum(any(a["category_id"] == cid for a in annotations[f"{split}:{im['id']}"]) for im in images if im["period"] == "Night") for cid, name in categories.items()},
            "missing_images": len(missing), "unreferenced_files": len(extra),
            "filename_suffix_counts": dict(suffix_counts),
            "filename_suffix_attribute_tuples": {k: [dict(zip(ATTRIBUTES, a)) for a in sorted(v)] for k, v in sorted(suffix_attributes.items())},
        }
        print(f"Annotations audited: {split}, {len(images)} images, {len(data['annotations'])} boxes", flush=True)
    tasks = [(raw / im["official_split"] / im["file_name"], im) for im in records]
    rows = []
    cache = audit / "image_integrity.jsonl"
    # Recompute every image so a previous audit cannot hide changed raw files.
    with ThreadPoolExecutor(max_workers=args.workers) as executor, cache.open("w", encoding="utf-8", newline="\n") as output:
        for index, result in enumerate(executor.map(inspect_image, tasks), 1):
            rows.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 1000 == 0:
                print(f"Fully decoded and hashed {index}/{len(tasks)} images ({time.monotonic()-started:.1f}s)", flush=True)
    failures = [r for r in rows if not r.get("ok")]
    dimension_mismatches = [r for r in rows if r.get("ok") and not r["dimension_matches"]]
    exact = duplicate_groups(rows, "sha256")
    decoded = duplicate_groups(rows, "decoded_rgb_sha256")
    near_pairs = near_duplicate_pairs(rows)
    write_json(audit / "annotation_issues.json", issues)
    write_json(audit / "duplicates.json", {"byte_identical": exact, "decoded_identical": decoded, "near_duplicate_candidates": near_pairs})
    # Inventory contains GT-derived counts and must never be an input to selection.
    with (audit / "inventory.jsonl").open("w", encoding="utf-8", newline="\n") as output:
        for row in records:
            output.write(json.dumps({**row, "box_count": len(annotations[row["key"]]),
                                     "boxes_by_class": dict(Counter(EXPECTED_CATEGORIES[a["category_id"]] for a in annotations[row["key"]]))}, ensure_ascii=False) + "\n")
    preview_path = Path("artifacts/data_review/attribute_overview.jpg").resolve()
    selected = make_preview(records, annotations, EXPECTED_CATEGORIES, raw, preview_path)
    write_json(audit / "preview_selection.json", {"keys": selected, "method": "two SHA256(key)-ordered images per official_split/period/location; no model-based selection"})
    receipt = json.loads((root / "sources/acquisition.json").read_text(encoding="utf-8"))
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Official SODA10M labeled train/val; no official test, unlabeled or external dataset",
        "archive": {k: receipt[k] for k in ("archive_name", "archive_bytes", "archive_sha256", "drive_file_id")},
        "annotation_sha256": json_hashes, "splits": summaries,
        "image_checks": {"attempted": len(rows), "successfully_decoded": len(rows)-len(failures),
                         "failed": len(failures), "dimension_mismatches": len(dimension_mismatches),
                         "bytes": sum(r.get("bytes", 0) for r in rows),
                         "actual_dimensions": dict(Counter("x".join(map(str, r["actual_size"])) for r in rows if r.get("ok"))),
                         "images_with_exif": sum(bool(r.get("exif_keys")) for r in rows),
                         "byte_duplicate_groups": len(exact), "decoded_duplicate_groups": len(decoded),
                         "near_duplicate_candidate_pairs": len(near_pairs),
                         "near_duplicate_method": "63-bit AC pHash distance <=6 AND 64-bit dHash distance <=10; all pairs; screening only"},
        "annotation_issue_counts": dict(Counter(i["type"] for i in issues)),
        "status": "audit_complete_split_blocked",
        "split_gate": {"passed": False, "reason": "No authoritative sequence/car-day mapping in annotations. Filename suffixes are not validated acquisition groups. Do not create formal image-random splits.",
                       "no_formal_split_created": True, "no_final_test_frozen": True},
        "runtime": {"python": sys.version, "executable": sys.executable, "pillow": PIL.__version__,
                    "numpy": np.__version__, "platform": platform.platform(), "workers": args.workers,
                    "elapsed_seconds": round(time.monotonic()-started, 1)},
        "private_artifacts": {"audit_directory": str(audit), "preview": str(preview_path)},
        "limitations": ["Structural box checks do not prove semantic annotation completeness.",
                        "Perceptual hashes screen for near duplicates, not acquisition sequence independence.",
                        "Global audit uses ground truth; inventory and previews are forbidden inputs to future selection.",
                        "No model trained, final test evaluated, or deployment tested."],
    }
    write_json(audit / "summary.json", summary)
    public = json.loads(json.dumps(summary))
    public.pop("private_artifacts")
    for split in public["splits"].values():
        split.pop("filename_suffix_counts")
        split.pop("filename_suffix_attribute_tuples")
    write_json(Path("reports/data_preparation_summary.json"), public)
    print(json.dumps({"status": summary["status"], "image_checks": summary["image_checks"],
                      "annotation_issue_counts": summary["annotation_issue_counts"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
