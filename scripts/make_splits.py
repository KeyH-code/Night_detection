"""Freeze deterministic SODA10M purpose splits; no model/framework dependency.

generate stages files under a new version directory and publishes READY last.
verify is read-only, including when an input or frozen file is invalid.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import sys

SEED = 20261025
PROTOCOL_VERSION = "similar_scene_purpose_v1"
TARGETS = {"source": {"source_train": 4000, "source_dev": 500, "source_test": 500},
           "night": {"night_pool": 994, "night_dev": 331, "night_test": 331}}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_path(root, relative):
    p = PurePosixPath(relative)
    require(isinstance(relative, str) and "\\" not in relative and ":" not in relative
            and not p.is_absolute() and ".." not in p.parts and p.parts, "Unsafe relative path")
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root.resolve()), "Path escaped version root")
    return resolved


def read_inputs(root, category_path, protocol_path):
    audit = root / "audit"
    inventory = [json.loads(s) for s in (audit / "inventory.jsonl").read_text(encoding="utf-8").splitlines()]
    integrity = {r["key"]: r for r in map(json.loads, (audit / "image_integrity.jsonl").read_text(encoding="utf-8").splitlines())}
    norm = read_json(audit / "normalization.json")
    mapping = read_json(category_path)["category_mapping"]
    require([m["training_id"] for m in mapping] == list(range(6)), "Mapping must be IDs 0..5")
    inputs = {"inventory_sha256": sha256(audit / "inventory.jsonl"),
              "integrity_sha256": sha256(audit / "image_integrity.jsonl"),
              "raw_annotation_sha256": {}, "normalized_annotation_sha256": {}}
    annotations = {}
    images = {}
    for split in ("train", "val"):
        raw_path = root / f"raw/SSLAD-2D/labeled/annotations/instance_{split}.json"
        normalized_path = root / f"normalized/annotations/instance_{split}.json"
        original, normalized = read_json(raw_path), read_json(normalized_path)
        inputs["raw_annotation_sha256"][split] = sha256(raw_path)
        inputs["normalized_annotation_sha256"][split] = sha256(normalized_path)
        require(inputs["raw_annotation_sha256"][split] == norm["splits"][split]["original_sha256"], "Raw annotation changed")
        require(inputs["normalized_annotation_sha256"][split] == norm["splits"][split]["normalized_sha256"], "Normalized annotation changed")
        require(original["images"] == normalized["images"], "Image metadata changed")
        require({m["official_id"]: m["official_name"] for m in mapping} == {c["id"]: c["name"] for c in normalized["categories"]}, "Category mismatch")
        for im in normalized["images"]:
            images[f"{split}:{im['id']}"] = im
        by_image = defaultdict(list)
        for a in normalized["annotations"]:
            by_image[a["image_id"]].append(a)
        for im in normalized["images"]:
            annotations[f"{split}:{im['id']}"] = sorted(by_image[im["id"]], key=lambda a: a["id"])
    require(len(inventory) == len(images) == len(integrity) == 10000, "Inventory count mismatch")
    for r in inventory:
        require(all(r[k] == images[r["key"]][k] for k in ("file_name", "width", "height", "period", "city", "location", "weather")), "Inventory metadata mismatch")
        require(r["box_count"] == len(annotations[r["key"]]), "Inventory annotation count mismatch")
    protocol = read_json(protocol_path)
    require(protocol["fingerprint_components"] == inputs, "Protocol input fingerprint mismatch")
    require(protocol["normalization_version"] == norm["version"], "Protocol normalization mismatch")
    require(protocol["seed"] == SEED and protocol["version"] == PROTOCOL_VERSION, "Unsupported protocol version/seed")
    for relative, expected in protocol["evidence_sha256"].items():
        evidence = safe_path(root, relative)
        require(sha256(evidence) == expected, f"Protocol evidence changed: {relative}")
        content = read_json(evidence)
        if "source_export_sha256" in content:
            require(sha256(evidence.parent / "export.original.json") == content["source_export_sha256"], "Human original export changed")
        for name, expected_hash in content.get("input_sha256", {}).items():
            require(sha256(audit / name) == expected_hash, f"Scene input changed: {name}")
    return inventory, integrity, annotations, mapping, protocol


def group_rows(rows, protocol):
    parent = {r["key"]: r["key"] for r in rows}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for g in protocol["selected_groups"]:
        inside = [k for k in g["keys"] if k in parent]
        require(not inside or len(inside) == len(g["keys"]), f"Group crosses official period/source pools: {g['id']}")
        for k in inside[1:]:
            a, b = find(inside[0]), find(k)
            parent[max(a, b)] = min(a, b)
    buckets = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["key"]):
        buckets[find(r["key"])].append(r)
    return [{"id": "scene-" + digest([r["key"] for r in members])[:16], "rows": members} for _, members in sorted(buckets.items())]


def features(groups):
    dimensions = sorted({f"{a}={r[a]}" for g in groups for r in g["rows"] for a in ("city", "location", "weather")})
    def vector(g):
        rows = g["rows"]
        return [len(rows), sum(r["boxes_by_class"].get("Pedestrian", 0) > 0 for r in rows)] + [sum(f"{a}={r[a]}" == d for r in rows for a in ("city", "location", "weather")) for d in dimensions] + [sum(r["boxes_by_class"].get(c, 0) > 0 for r in rows) for c in ("Cyclist", "Car", "Truck", "Tram", "Tricycle")]
    return [vector(g) for g in groups], len(dimensions)


def allocate(groups, targets, night=False):
    require(sum(len(g["rows"]) for g in groups) == sum(targets.values()), "Pool total does not match targets")
    rng = random.Random(SEED)
    vectors, nattrs = features(groups)
    purposes = list(targets)
    totals = [sum(v[i] for v in vectors) for i in range(len(vectors[0]))]
    expected = [[x * targets[p] / totals[0] for x in totals] for p in purposes]
    state = [[0] * len(totals) for _ in purposes]
    assignment = {}
    order = list(range(len(groups)))
    rng.shuffle(order)
    order.sort(key=lambda i: -vectors[i][0])
    def contribution(s, j):
        errs = [(s[k] - expected[j][k]) ** 2 / max(expected[j][k], 1) for k in range(len(s))]
        return (errs[0], errs[1], sum(errs[2:2+nattrs]), sum(errs[2+nattrs:]))
    for i in order:
        v = vectors[i]
        options = [j for j, p in enumerate(purposes) if state[j][0] + v[0] <= targets[p]]
        if not options:
            options = [j for j, p in enumerate(purposes) if state[j][0] + v[0] <= targets[p] + 5]
        require(options, f"Cannot place intact group {groups[i]['id']} (size {v[0]}) without exceeding targets")
        def delta(j):
            before = contribution(state[j], j)
            after = contribution([a+b for a, b in zip(state[j], v)], j)
            return tuple(round(b-a, 10) for a, b in zip(before, after))
        j = min(options, key=lambda j: (delta(j), j))
        assignment[i] = j
        state[j] = [a+b for a, b in zip(state[j], v)]
    by_size = defaultdict(list)
    for i, v in enumerate(vectors):
        by_size[v[0]].append(i)
    sizes = sorted(k for k, ids in by_size.items() if len(ids) > 1)
    for _ in range(8):
        if not sizes:
            break
        improvements = 0
        for _ in range(6000):
            a, b = rng.sample(by_size[rng.choice(sizes)], 2)
            ja, jb = assignment[a], assignment[b]
            if ja == jb:
                continue
            sa = [s-va+vb for s, va, vb in zip(state[ja], vectors[a], vectors[b])]
            sb = [s-vb+va for s, va, vb in zip(state[jb], vectors[a], vectors[b])]
            if night and any(purposes[j] in ("night_dev", "night_test") and st[1] < 40 for j, st in ((ja, sa), (jb, sb))):
                continue
            old = tuple(round(x+y, 10) for x, y in zip(contribution(state[ja], ja), contribution(state[jb], jb)))
            new = tuple(round(x+y, 10) for x, y in zip(contribution(sa, ja), contribution(sb, jb)))
            if new < old:
                assignment[a], assignment[b] = jb, ja
                state[ja], state[jb] = sa, sb
                improvements += 1
        if not improvements:
            break
    result = {p: [] for p in purposes}
    for i, g in enumerate(groups):
        for r in g["rows"]:
            result[purposes[assignment[i]]].append(dict(r, group_id=g["id"]))
    validate_assignment(result, targets, night)
    return {p: sorted(rs, key=lambda r: r["key"]) for p, rs in result.items()}


def validate_assignment(result, targets, night):
    seen, group_purpose = set(), {}
    for p, rows in result.items():
        require(abs(len(rows) - targets[p]) <= 5, f"{p}: actual {len(rows)} target {targets[p]} tolerance 5")
        for r in rows:
            require(r["key"] not in seen, "Duplicate/cross-purpose image")
            seen.add(r["key"])
            require(group_purpose.setdefault(r["group_id"], p) == p, "Related group crosses purposes")
        if night and p in ("night_dev", "night_test"):
            count = sum(r["boxes_by_class"].get("Pedestrian", 0) > 0 for r in rows)
            require(count >= 40, f"{p}: pedestrian images {count} < 40")
    require(len(seen) == sum(targets.values()), "Total not conserved")


def label_bytes(row, annotations, mapping):
    ids = {m["official_id"]: m["training_id"] for m in mapping}
    lines = []
    for a in annotations:
        x, y, w, h = a["bbox"]
        width, height = row["width"], row["height"]
        require(all(math.isfinite(v) for v in (x, y, w, h)) and x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= width and y+h <= height, "Invalid normalized box")
        values = ((x+w/2)/width, (y+h/2)/height, w/width, h/height)
        formatted = [f"{v:.12f}" for v in values]
        cx, cy, nw, nh = map(float, formatted)
        recovered = ((cx-nw/2)*width, (cy-nh/2)*height, nw*width, nh*height)
        require(max(abs(a-b) for a, b in zip(recovered, (x, y, w, h))) < 1e-6, "YOLO roundtrip failed")
        lines.append(str(ids[a["category_id"]]) + " " + " ".join(formatted))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def describe(rows):
    return {"images": len(rows), "groups": len({r["group_id"] for r in rows}),
            "pedestrian_images": sum(r["boxes_by_class"].get("Pedestrian", 0) > 0 for r in rows),
            "empty_images": sum(r["box_count"] == 0 for r in rows),
            "boxes": dict(sum((Counter(r["boxes_by_class"]) for r in rows), Counter())),
            **{a: dict(sorted(Counter(r[a] for r in rows).items())) for a in ("city", "location", "weather")}}


def prepare(root, category_path, stage, protocol_path=None):
    protocol_path = protocol_path or root / "splits" / f"{stage}_v1/protocol.json"
    require(Path(protocol_path).is_file(), "Supply --protocol for a new data version; frozen versions use their own protocol.json")
    inventory, integrity, annotations, mapping, protocol = read_inputs(root, category_path, protocol_path)
    selected = [r for r in inventory if (r["official_split"] == "train" if stage == "source" else r["official_split"] == "val" and r["period"] == "Night")]
    groups = group_rows(selected, protocol)
    allocation = allocate(groups, TARGETS[stage], stage == "night")
    if stage == "night":
        cross = [r for r in inventory if r["official_split"] == "val" and r["period"] == "Daytime"]
        allocation["daytime_cross_scene_test"] = sorted([dict(r, group_id=g["id"]) for g in group_rows(cross, protocol) for r in g["rows"]], key=lambda r: r["key"])
    return allocation, integrity, annotations, mapping, protocol


def build_records(root, version, allocation, integrity, annotations, mapping, protocol, write=False):
    files, manifests, held_out = [], {}, {}
    components = protocol["fingerprint_components"]
    for purpose, rows in allocation.items():
        records = []
        for row in rows:
            key, split = row["key"], row["official_split"]
            source = root / "raw/SSLAD-2D/labeled" / split / row["file_name"]
            image = f"images/{purpose}/{row['file_name']}"
            label_dir = "hidden_ground_truth/night_pool" if purpose == "night_pool" else f"labels/{purpose}"
            label = f"{label_dir}/{Path(row['file_name']).stem}.txt"
            content = label_bytes(row, annotations[key], mapping)
            image_hash = integrity[key]["sha256"]
            record = {"key": key, "purpose": purpose, "group_id": row["group_id"], "protocol_version": PROTOCOL_VERSION,
                      "image": image, "label": label, "width": row["width"], "height": row["height"],
                      "official_split": split, "official_period": row["period"],
                      "classes": sorted({m["training_id"] for m in mapping if m["official_name"] in row["boxes_by_class"]}),
                      "image_sha256": image_hash, "label_sha256": hashlib.sha256(content).hexdigest(),
                      "annotation_sha256": digest(annotations[key]),
                      "raw_annotation_sha256": components["raw_annotation_sha256"][split],
                      "normalized_annotation_sha256": components["normalized_annotation_sha256"][split],
                      "box_count": row["box_count"], "boxes_by_class": row["boxes_by_class"],
                      **{a: row[a] for a in ("city", "location", "weather")}}
            records.append(record)
            if write:
                require(sha256(source) == image_hash, f"Raw image changed: {key}")
                image_dest, label_dest = safe_path(version, image), safe_path(version, label)
                image_dest.parent.mkdir(parents=True, exist_ok=True)
                label_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, image_dest)
                label_dest.write_bytes(content)
            if purpose in ("source_train", "source_dev"):
                files.extend([{"path": image, "sha256": image_hash, "purpose": purpose, "kind": "image"},
                              {"path": label, "sha256": record["label_sha256"], "purpose": purpose, "kind": "label"}])
        manifest = {"purpose": purpose, "records": records}
        path = f"manifests/{purpose}.json"
        receipt = {"path": path, "sha256": hashlib.sha256(canonical(manifest)+b"\n").hexdigest(), "count": len(rows)}
        if write:
            write_json(version / path, manifest)
        (manifests if purpose in ("source_train", "source_dev") else held_out)[purpose] = receipt
        if purpose == "night_pool" and write:
            write_json(version / "selection/night_pool_images.json", {"purpose": "night_pool_selection_images_only", "records": [{"key": r["key"], "image": r["image"], "image_sha256": r["image_sha256"], "group_id": r["group_id"]} for r in records]})
    return files, manifests, held_out


def candidate_ready(root, version, stage, prepared, write=False):
    allocation, integrity, annotations, mapping, protocol = prepared
    files, manifests, held_out = build_records(root, version, allocation, integrity, annotations, mapping, protocol, write)
    cats = {"category_mapping": mapping, "names": [m["name"] for m in mapping]}
    if write:
        write_json(version / "protocol.json", protocol)
        write_json(version / "categories.json", cats)
    return {"schema_version": 1, "version": f"{stage}_v1", "status": "frozen", "seed": SEED,
            "data_fingerprint": digest(protocol["fingerprint_components"]), "normalization_version": protocol["normalization_version"],
            "grouping_protocol": {"path": "protocol.json", "sha256": hashlib.sha256(canonical(protocol)+b"\n").hexdigest()},
            "categories": {"path": "categories.json", "sha256": hashlib.sha256(canonical(cats)+b"\n").hexdigest(), "names": cats["names"]},
            "category_mapping": mapping, "counts": {p: len(r) for p, r in allocation.items()},
            "manifests": manifests, "held_out": held_out, "files": files,
            "distributions": {p: describe(r) for p, r in allocation.items()},
            "selection_manifest": "selection/night_pool_images.json" if stage == "night" else None,
            "final_test_executed": False}


def validate_version(root, version, stage, category_path, ready=None, prepared=None):
    ready = ready if ready is not None else read_json(version / "READY.json")
    prepared = prepared if prepared is not None else prepare(root, category_path, stage)
    expected = candidate_ready(root, version, stage, prepared)
    require(ready == expected, "Frozen metadata/assignment differs from deterministic recomputation")
    for receipt in [ready["grouping_protocol"], ready["categories"], *ready["manifests"].values(), *ready["held_out"].values()]:
        require(sha256(safe_path(version, receipt["path"])) == receipt["sha256"], f"Metadata hash mismatch: {receipt['path']}")
    counts, empty, keys, group_purposes = Counter(), Counter(), set(), {}
    for purpose, receipt in {**ready["manifests"], **ready["held_out"]}.items():
        manifest = read_json(safe_path(version, receipt["path"]))
        for r in manifest["records"]:
            require(r["key"] not in keys, "Duplicate image key")
            keys.add(r["key"])
            require(group_purposes.setdefault(r["group_id"], purpose) == purpose, "Related group crosses uses")
            for field, hash_field in (("image", "image_sha256"), ("label", "label_sha256")):
                require(sha256(safe_path(version, r[field])) == r[hash_field], f"Derived {field} hash mismatch {r['key']}")
            label = safe_path(version, r["label"]).read_bytes()
            # Expected annotation hash and exact label bytes are recomputed, not trusted from manifest.
            annotations = prepared[2][r["key"]]
            require(label == label_bytes(r, annotations, prepared[3]), "YOLO bytes differ from normalized boxes")
            require(digest(annotations) == r["annotation_sha256"], "Per-image annotation hash mismatch")
            counts[purpose] += 1
            empty[purpose] += not label
        require(counts[purpose] == receipt["count"], "Manifest length mismatch")
    allowed = {r["path"] for r in ready["files"]}
    require(len(allowed) == len(ready["files"]), "Duplicate whitelist paths")
    require(all(r["purpose"] in ("source_train", "source_dev") for r in ready["files"]), "Test data in training whitelist")
    require(dict(counts) == ready["counts"], "Count mismatch")
    if stage == "night":
        selection = read_json(version / ready["selection_manifest"])
        pool = read_json(version / ready["held_out"]["night_pool"]["path"])["records"]
        require(selection["records"] == [{"key": r["key"], "image": r["image"], "image_sha256": r["image_sha256"], "group_id": r["group_id"]} for r in pool], "Selection manifest exposes labels or differs")
    return {"status": "passed", "counts": dict(counts), "empty_images": dict(empty), "checks": ["conservation", "purpose_disjointness", "group_isolation", "deterministic_recomputation", "metadata_and_all_image_label_hashes", "category_mapping", "normalized_bbox_roundtrip", "empty_labels_retained", "final_test_excluded_from_training", "selection_label_isolation"], "final_test_executed": False}


def frozen_state(root):
    """Read current manifests; preserve historical audit receipts elsewhere."""
    versions = {}
    for stage in ("source", "night"):
        v = root / "splits" / f"{stage}_v1"
        if (v / "READY.json").exists():
            r = read_json(v / "READY.json")
            require(r["status"] == "frozen", "Invalid READY status")
            for rec in {**r["manifests"], **r["held_out"]}.values():
                require(sha256(safe_path(v, rec["path"])) == rec["sha256"], "Frozen manifest changed")
            versions[r["version"]] = {"ready_sha256": sha256(v / "READY.json"), "counts": r["counts"]}
    return {"formal_split_created": bool(versions), "final_test_frozen": "source_v1" in versions and "night_v1" in versions, "frozen_versions": versions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("generate", "verify", "plan"))
    p.add_argument("--stage", choices=("source", "night"), required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/soda10m"))
    p.add_argument("--categories", type=Path, default=Path(__file__).resolve().parents[1] / "configs/soda10m_categories.json")
    p.add_argument("--protocol", type=Path, help="Explicit reviewed protocol for generation/planning")
    args = p.parse_args()
    root = args.data_root.resolve()
    version = root / "splits" / f"{args.stage}_v1"
    if args.command == "verify":
        print(json.dumps(validate_version(root, version, args.stage, args.categories), ensure_ascii=False, indent=2))
        return
    prepared = prepare(root, args.categories, args.stage, args.protocol)
    if args.command == "plan":
        print(json.dumps({p: describe(r) for p, r in prepared[0].items()}, ensure_ascii=False, indent=2))
        return
    # mkdir is the exclusive publisher lock. Never resume/overwrite a partial or frozen version.
    version.mkdir(parents=True, exist_ok=False)
    write_json(version / "BUILDING.json", {"owner": "data_split_session", "stage": args.stage, "seed": SEED})
    try:
        ready = candidate_ready(root, version, args.stage, prepared, write=True)
        validation = validate_version(root, version, args.stage, args.categories, ready, prepared)
        write_json(version / "validation.json", validation)
        write_json(version / "READY.pending.json", ready)
        (version / "BUILDING.json").unlink()
        (version / "READY.pending.json").rename(version / "READY.json")
        print(json.dumps({"ready": str(version / "READY.json"), "ready_sha256": sha256(version / "READY.json"), "validation": validation}, ensure_ascii=False, indent=2))
    except Exception as exc:
        write_json(version / "FAILED.json", {"error": str(exc), "ready_published": False})
        raise


if __name__ == "__main__":
    main()
