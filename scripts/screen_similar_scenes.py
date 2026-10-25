"""Create a reproducible human-review sample; never remove or split images."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_soda10m import sha256, write_json
from serve_data_review import connected_groups


VERSION = "scene-review-20261025-v2"
SEED = 20261025
LIMIT = 50


def rank(value):
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def expanded_pairs(rows, excluded_keys):
    """Exhaustively compare hashes outside existing strict candidate components.

    This is a bounded recall probe, not a new duplicate decision threshold.
    No city/period/split restriction: conflicting metadata must remain visible.
    """
    rows = sorted((r for r in rows if r["key"] not in excluded_keys), key=lambda r: r["key"])
    p = np.array([int(r["phash"], 16) for r in rows], dtype=np.uint64)
    d = np.array([int(r["dhash"], 16) for r in rows], dtype=np.uint64)
    result = []
    for start in range(0, len(rows), 128):
        stop = min(start + 128, len(rows))
        pd = np.bitwise_count(p[start:stop, None] ^ p[None, :])
        dd = np.bitwise_count(d[start:stop, None] ^ d[None, :])
        hits = (pd <= 10) & (dd <= 16)
        hits &= np.arange(len(rows))[None, :] > np.arange(start, stop)[:, None]
        for i, j in zip(*np.nonzero(hits)):
            result.append({"a": rows[start + int(i)]["key"], "b": rows[int(j)]["key"],
                           "phash_distance": int(pd[i, j]), "dhash_distance": int(dd[i, j])})
    return result


def describe_group(group, images, integrity, origin):
    keys = group["keys"]
    comparisons = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            comparisons.append({"a": a, "b": b,
                "phash_distance": (int(integrity[a]["phash"], 16) ^ int(integrity[b]["phash"], 16)).bit_count(),
                "dhash_distance": (int(integrity[a]["dhash"], 16) ^ int(integrity[b]["dhash"], 16)).bit_count()})
    farthest = max(comparisons, key=lambda e: (e["phash_distance"] + e["dhash_distance"], e["a"], e["b"]))
    closest = min(group["edges"], key=lambda e: (e["phash_distance"] + e["dhash_distance"], e["a"], e["b"]))
    attributes = {name: sorted({images[k][name] for k in keys}) for name in ("official_split", "period", "city", "location", "weather")}
    return {**group, "origin": origin, "attributes": attributes,
            "closest_pair": closest, "farthest_pair": farthest,
            "chain_warning": any(e["phash_distance"] > 6 or e["dhash_distance"] > 10 for e in comparisons),
            "edge_density": round(len(group["edges"]) / len(comparisons), 4),
            "selection_reasons": [], "previous_review": None}


def sample_groups(groups, reviewed_ids, limit=LIMIT):
    """Disjoint risk/coverage quotas, excluding previously reviewed components."""
    if not 0 <= limit <= LIMIT:
        raise ValueError("Human review is capped at 50 groups")
    eligible = [g for g in groups if g["id"] not in reviewed_ids]
    selected, reasons = [], {}

    def take(pool, count, label, order=None):
        available = [g for g in pool if g["id"] not in reasons]
        available.sort(key=order or (lambda g: rank(g["id"])))
        for g in available[:min(count, limit - len(selected))]:
            selected.append(g["id"])
            reasons[g["id"]] = label

    strict = [g for g in eligible if g["origin"] == "strict"]
    take([g for g in strict if "Night" in g["attributes"]["period"]], 18, "夜间覆盖")
    take([g for g in strict if len(g["keys"]) >= 5], 10, "大组优先",
         lambda g: (-len(g["keys"]), rank(g["id"])))
    take([g for g in strict if g["chain_warning"]], 8, "链式关系核对")
    buckets = defaultdict(list)
    for g in strict:
        signature = tuple(tuple(g["attributes"][a]) for a in ("official_split", "period", "city", "location", "weather"))
        buckets[signature].append(g)
    coverage = []
    for bucket in buckets.values():
        bucket.sort(key=lambda g: rank(g["id"]))
    while any(buckets.values()):
        for signature in sorted(buckets):
            if buckets[signature]:
                coverage.append(buckets[signature].pop(0))
    # The coverage order deliberately round-robins attribute strata.
    coverage_order = {g["id"]: i for i, g in enumerate(coverage)}
    take(coverage, 8, "场景属性覆盖", lambda g: coverage_order[g["id"]])
    take([g for g in eligible if g["origin"] == "expanded"], 6, "原候选之外补查")
    take(eligible, limit - len(selected), "补足覆盖")
    # Keep the review sequence reproducible and alternate risk categories.
    selected.sort(key=rank)
    return selected, reasons


def build_screening(project):
    project = Path(project)
    audit = project / "data/soda10m/audit"
    inventory = [json.loads(line) for line in (audit / "inventory.jsonl").read_text(encoding="utf-8").splitlines()]
    images = {r["key"]: r for r in inventory}
    rows = [json.loads(line) for line in (audit / "image_integrity.jsonl").read_text(encoding="utf-8").splitlines()]
    integrity = {r["key"]: r for r in rows}
    strict_pairs = json.loads((audit / "duplicates.json").read_text(encoding="utf-8"))["near_duplicate_candidates"]
    strict = connected_groups(strict_pairs)
    involved = {k for g in strict for k in g["keys"]}
    extra_pairs = expanded_pairs(rows, involved)
    expanded = connected_groups(extra_pairs)
    groups = [describe_group(g, images, integrity, "strict") for g in strict]
    groups += [describe_group({**g, "id": "probe-" + g["id"].removeprefix("group-")}, images, integrity, "expanded") for g in expanded]
    raw = project / "data/soda10m/raw/SSLAD-2D/labeled/annotations"
    annotation_hashes = {s: sha256(raw / f"instance_{s}.json") for s in ("train", "val")}
    fingerprint = hashlib.sha256(json.dumps(annotation_hashes, sort_keys=True).encode("utf-8")).hexdigest()
    previous_path = audit / "human_reviews/2026-10-25/interpretation.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if previous["dataset_fingerprint"] != fingerprint or previous["annotation_sha256"] != annotation_hashes:
        raise RuntimeError("Previous human review belongs to another dataset version")
    previous_groups = {g["id"]: g for g in previous["group_sample"]["groups"]}
    group_ids = {g["id"] for g in groups}
    if not set(previous_groups) <= group_ids:
        raise RuntimeError("Historical candidate membership changed")
    selected, reasons = sample_groups(groups, set(previous_groups))
    for g in groups:
        if g["id"] in reasons:
            g["selection_reasons"] = [reasons[g["id"]]]
        g["previous_review"] = previous_groups.get(g["id"])
    chosen = [g for g in groups if g["id"] in selected]
    summary = {
        "strict_pairs": len(strict_pairs), "strict_groups": len(strict), "strict_involved_images": len(involved),
        "expanded_pairs": len(extra_pairs), "expanded_groups": len(expanded),
        "outside_strict_images_scanned": len(rows) - len(involved),
        "sample_groups": len(selected), "historical_groups_excluded": len(previous_groups),
        "sample_images": len({k for g in chosen for k in g["keys"]}),
        "reason_counts": dict(Counter(reasons.values())),
        "sample_night_groups": sum("Night" in g["attributes"]["period"] for g in chosen),
        "sample_max_group_size": max((len(g["keys"]) for g in chosen), default=0),
        "sample_cross_official_split_groups": sum(len(g["attributes"]["official_split"]) > 1 for g in chosen),
    }
    result = {"schema_version": 2, "campaign_id": VERSION, "seed": SEED, "human_review_limit": LIMIT,
        "dataset_fingerprint": fingerprint, "annotation_sha256": annotation_hashes,
        "input_sha256": {name: sha256(audit / name) for name in ("inventory.jsonl", "image_integrity.jsonl", "duplicates.json")},
        "previous_interpretation_sha256": sha256(previous_path),
        "policy": {"period": "official period unchanged", "raw_images_deleted": 0,
                   "formal_split_created": False, "automatic_duplicate_verdicts": False,
                   "sampling": "risk and attribute coverage, not prevalence estimation",
                   "expanded_scope": "all image pairs outside existing strict candidate groups; pHash <=10 and dHash <=16; recall probe only"},
        "summary": summary, "sample_group_ids": selected, "historical_group_ids": list(previous_groups), "groups": groups}
    write_json(audit / "scene_screening.json", result)
    return result


if __name__ == "__main__":
    result = build_screening(Path(__file__).resolve().parents[1])
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
