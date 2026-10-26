"""Check extracted bytes against archive and bind the audit to raw annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from datetime import datetime, timezone

from prepare_soda10m import sha256, write_json
from normalize import clip_box
from make_splits import frozen_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/soda10m"))
    args = parser.parse_args()
    root = args.data_root.resolve()
    summary = json.loads((root / "audit/summary.json").read_text(encoding="utf-8"))
    integrity = [json.loads(line) for line in (root / "audit/image_integrity.jsonl").read_text(encoding="utf-8").splitlines()]
    if summary["image_checks"]["failed"] or summary["image_checks"]["dimension_mismatches"]:
        raise RuntimeError("Image integrity gate failed")
    indexed = {}
    for row in integrity:
        split = row["key"].split(":", 1)[0]
        name = f"SSLAD-2D/labeled/{split}/{row['file_name']}"
        if name in indexed:
            raise RuntimeError(f"Duplicate integrity entry: {name}")
        indexed[name] = row["sha256"]
    raw_annotations = root / "raw/SSLAD-2D/labeled/annotations"
    for split in ("train", "val"):
        path = raw_annotations / f"instance_{split}.json"
        digest = sha256(path)
        if digest != summary["annotation_sha256"][split]:
            raise RuntimeError("Raw annotation changed after audit")
        indexed[f"SSLAD-2D/labeled/annotations/{path.name}"] = digest
    checked = set()
    with tarfile.open(root / "archives/labeled_trainval.tar", "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name not in indexed:
                raise RuntimeError(f"Unaccounted archive member: {member.name}")
            digest = hashlib.sha256()
            with archive.extractfile(member) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != indexed[member.name]:
                raise RuntimeError(f"Extracted file differs from archive: {member.name}")
            checked.add(member.name)
            if len(checked) % 2000 == 0:
                print(f"Verified archived bytes for {len(checked)}/{len(indexed)} files", flush=True)
    if checked != set(indexed):
        raise RuntimeError("Some audited files were not found in archive")
    category_config = json.loads(Path("configs/soda10m_categories.json").read_text(encoding="utf-8"))
    mappings = category_config["category_mapping"]
    if [row["training_id"] for row in mappings] != list(range(6)):
        raise RuntimeError("Training category IDs are not contiguous 0..5")
    for split in ("train", "val"):
        observed = summary["splits"][split]["categories"]
        for mapping in mappings:
            if observed[str(mapping["official_id"])] != mapping["official_name"]:
                raise RuntimeError("Category configuration differs from source annotation")
    normalization = json.loads((root / "audit/normalization.json").read_text(encoding="utf-8"))
    normalized_boxes = 0
    changed_boxes = 0
    for split in ("train", "val"):
        original = json.loads((raw_annotations / f"instance_{split}.json").read_text(encoding="utf-8"))
        normalized_path = root / "normalized/annotations" / f"instance_{split}.json"
        normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
        if sha256(normalized_path) != normalization["splits"][split]["normalized_sha256"]:
            raise RuntimeError("Normalized annotation changed after preparation")
        if original["images"] != normalized["images"] or original["categories"] != normalized["categories"]:
            raise RuntimeError("Normalization changed image metadata or category definitions")
        if len(original["annotations"]) != len(normalized["annotations"]):
            raise RuntimeError("Normalization added/dropped annotations")
        by_id = {im["id"]: im for im in original["images"]}
        for source, result in zip(original["annotations"], normalized["annotations"]):
            im = by_id[source["image_id"]]
            expected_box = clip_box(source["bbox"], im["width"], im["height"])
            expected = dict(source, bbox=expected_box, area=expected_box[2]*expected_box[3])
            if result != expected:
                raise RuntimeError("Normalization changed more than bbox/area")
            normalized_boxes += 1
            changed_boxes += source["bbox"] != result["bbox"]
    receipt = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_files_verified": len(checked), "all_extracted_hashes_match_archive": True,
        "annotation_hashes_unchanged": True, "category_mapping_matches_both_annotations": True,
        "normalized_boxes_verified": normalized_boxes, "clipped_boxes_verified": changed_boxes,
        "normalization_preserves_image_metadata_and_all_annotation_ids": True,
        **frozen_state(root),
        "scripts_sha256": {p.name: sha256(p) for p in sorted(Path("scripts").glob("*.py"))},
    }
    prior = root / "audit/verification.json"
    if prior.exists():
        history = root / "audit/verification_history" / f"{sha256(prior)}.json"
        if not history.exists():
            history.parent.mkdir(parents=True, exist_ok=True)
            history.write_bytes(prior.read_bytes())
    write_json(prior, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
