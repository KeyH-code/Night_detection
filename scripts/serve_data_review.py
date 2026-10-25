"""Loopback-only, read-only SODA10M review UI bound to verified raw data."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import random
import urllib.parse

from PIL import Image
from prepare_soda10m import sha256, write_json


def connected_groups(pairs):
    """Connected components retain explicit edges; do not imply all-pairs similarity."""
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for pair in pairs:
        a, b = find(pair["a"]), find(pair["b"])
        if a != b:
            parent[max(a, b)] = min(a, b)
    nodes, edges = defaultdict(list), defaultdict(list)
    for key in sorted(parent):
        nodes[find(key)].append(key)
    for pair in pairs:
        edges[find(pair["a"])].append(pair)
    groups = []
    for root, keys in sorted(nodes.items()):
        group_id = "group-" + hashlib.sha256("|".join(keys).encode("utf-8")).hexdigest()[:12]
        groups.append({"id": group_id, "keys": keys, "edges": edges[root]})
    return groups


def stratified_sample(rows, total=24, seed=20261025):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["key"]):
        buckets[(row["city"], row["location"])].append(row["key"])
    for bucket in buckets.values():
        rng.shuffle(bucket)
    chosen = []
    while len(chosen) < total:
        added = False
        for group in sorted(buckets):
            if buckets[group] and len(chosen) < total:
                chosen.append(buckets[group].pop())
                added = True
        if not added:
            break
    return chosen


def build_catalog(project):
    root = project / "data/soda10m"
    alignment = json.loads((project / "reports/source_alignment.json").read_text(encoding="utf-8"))
    samples = alignment.get("fresh_remote_image_samples", [])
    if not alignment.get("fresh_remote_annotations_match_raw") or len(samples) != 8 or not all(s["local_match"] for s in samples):
        raise RuntimeError("Source alignment gate not satisfied; run verify_official_source.py first")
    integrity = {r["key"]: r for r in map(json.loads, (root / "audit/image_integrity.jsonl").read_text(encoding="utf-8").splitlines())}
    comparisons = {row["official_split"]: row for row in alignment["comparisons"]}
    images, paths, hashes = {}, {}, {}
    raw = root / "raw/SSLAD-2D/labeled"
    for split in ("train", "val"):
        source = raw / "annotations" / f"instance_{split}.json"
        hashes[split] = sha256(source)
        if hashes[split] != comparisons[split]["remote_sha256"]:
            raise RuntimeError("Raw annotations changed after official-source check")
        data = json.loads(source.read_text(encoding="utf-8"))
        if {im["file_name"] for im in data["images"]} != {p.name for p in (raw / split).iterdir() if p.is_file()}:
            raise RuntimeError("Raw files do not match official split")
        for record in data["images"]:
            key = f"{split}:{record['id']}"
            check = integrity.get(key)
            if not check or check["file_name"] != record["file_name"] or not check["ok"]:
                raise RuntimeError("Image key/path differs from audited original")
            path = (raw / split / record["file_name"]).resolve()
            if not path.is_relative_to(raw.resolve()):
                raise RuntimeError("Image path escaped raw dataset")
            images[key] = {**record, "key": key, "official_split": split,
                "raw_path": str(path.relative_to(project)).replace("\\", "/"),
                "annotation_file": f"instance_{split}.json", "sha256": check["sha256"],
                "mean_gray": check["mean_gray"]}
            paths[key] = path
    pairs = json.loads((root / "audit/duplicates.json").read_text(encoding="utf-8"))["near_duplicate_candidates"]
    for pair in pairs:
        if pair["a"] not in images or pair["b"] not in images:
            raise RuntimeError("Candidate refers to unknown official image")
    groups = connected_groups(pairs)
    night = [row for row in images.values() if row["official_split"] == "val" and row["period"] == "Night"]
    daytime = [row for row in images.values() if row["official_split"] == "val" and row["period"] == "Daytime"]
    bright = sorted(night, key=lambda row: (-row["mean_gray"], row["key"]))[:24]
    visual = json.loads((root / "audit/visual_review.json").read_text(encoding="utf-8"))
    flagged = visual["night_attribute_review"]["daytime_appearing_night_tagged_keys"]
    if not all(key in images and images[key]["period"] == "Night" for key in flagged):
        raise RuntimeError("Agent flags are not a subset of official Night labels")
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    screening_path = root / "audit/scene_screening.json"
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    if screening["dataset_fingerprint"] != fingerprint or screening["annotation_sha256"] != hashes:
        raise RuntimeError("Scene screening refers to another dataset version")
    for name, digest in screening["input_sha256"].items():
        if name not in {"inventory.jsonl", "image_integrity.jsonl", "duplicates.json"} or sha256(root / "audit" / name) != digest:
            raise RuntimeError("Scene screening inputs changed; regenerate the review sample")
    if sha256(root / "audit/human_reviews/2026-10-25/interpretation.json") != screening["previous_interpretation_sha256"]:
        raise RuntimeError("Historical review changed; regenerate the review sample")
    groups = screening["groups"]
    sample_groups = screening["sample_group_ids"]
    group_ids = {g["id"] for g in groups}
    if len(sample_groups) > 50 or len(set(sample_groups)) != len(sample_groups) or not set(sample_groups) <= group_ids:
        raise RuntimeError("Invalid scene-review sample or more than 50 groups")
    if set(sample_groups) & set(screening["historical_group_ids"]):
        raise RuntimeError("This round must exclude previously reviewed groups")
    for group in groups:
        if not set(group["keys"]) <= images.keys() or any(e["a"] not in group["keys"] or e["b"] not in group["keys"] for e in group["edges"]):
            raise RuntimeError("Scene group refers to unknown images or edges")
    catalog = {
        "schema_version": 2, "dataset_fingerprint": fingerprint,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "alignment": alignment, "annotation_hashes": hashes, "images": images,
        "groups": groups, "pair_count": len(pairs),
        "sample_sets": {"sample": stratified_sample(night), "flagged": flagged,
            "bright": [im["key"] for im in bright], "all": [im["key"] for im in night],
            "day": stratified_sample(daytime)},
        "sample_group_ids": sample_groups,
        "historical_group_ids": screening["historical_group_ids"],
        "scene_screening": {k: screening[k] for k in ("campaign_id", "seed", "human_review_limit", "summary", "policy")},
        "sampling": {"seed": screening["seed"], "night": "official period, read-only", "groups": "50 new groups, risk and attribute coverage; 12 prior groups excluded; not prevalence-weighted"},
    }
    write_json(root / "review/catalog.json", catalog)
    return catalog, paths


class ReviewHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, project, catalog, paths, **kwargs):
        self.project, self.catalog, self.paths = project, catalog, paths
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(fmt, *args)

    def send_bytes(self, payload, content_type, status=200, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def error(self, message, status=404):
        self.send_bytes(json.dumps({"error": message}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self):
        allowed_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if self.headers.get("Host") not in allowed_hosts:
            return self.error("仅支持本机访问", 403)
        url = urllib.parse.urlsplit(self.path)
        if url.path == "/api/health":
            return self.send_bytes(json.dumps({"status": "ready", "fingerprint": self.catalog["dataset_fingerprint"]}).encode(), "application/json")
        if url.path == "/api/catalog":
            return self.send_bytes(json.dumps(self.catalog, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        if url.path in ("/", "/styles.css", "/app.js"):
            name = "index.html" if url.path == "/" else url.path[1:]
            types = {"index.html": "text/html; charset=utf-8", "styles.css": "text/css; charset=utf-8", "app.js": "text/javascript; charset=utf-8"}
            return self.send_bytes((self.project / "review_app" / name).read_bytes(), types[name])
        if url.path.startswith(("/image/", "/thumb/")):
            kind, key = url.path.strip("/").split("/", 1)
            key = urllib.parse.unquote(key)
            if key not in self.paths:
                return self.error("图片不在已核验清单中")
            try:
                path = self.paths[key]
                payload = path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != self.catalog["images"][key]["sha256"]:
                    return self.error("图片已变化，来源校验失败；请重新审计", 409)
                if kind == "thumb":
                    with Image.open(io.BytesIO(payload)) as image:
                        image.thumbnail((960, 540), Image.Resampling.LANCZOS)
                        output = io.BytesIO()
                        image.convert("RGB").save(output, "JPEG", quality=87)
                        payload = output.getvalue()
                return self.send_bytes(payload, "image/jpeg", cache=True)
            except (OSError, ValueError):
                return self.error("图片读取失败，请检查本地数据", 500)
        return self.error("没有这个页面或文件")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8731)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    catalog, paths = build_catalog(project)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(ReviewHandler, project=project, catalog=catalog, paths=paths))
    write_json(project / "artifacts/data_review/server.json", {"pid": os.getpid(), "port": args.port, "url": f"http://127.0.0.1:{args.port}", "dataset_fingerprint": catalog["dataset_fingerprint"]})
    print(f"Local review ready: http://127.0.0.1:{args.port} / {len(catalog['groups'])} candidate groups", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
