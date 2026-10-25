"""Recheck the official link and compare fresh remote annotation bytes to raw.

Only a small archive prefix is fetched. Existing acquisition evidence is kept.
This confirms annotation/source alignment; it is not a publisher signature.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time
import urllib.parse
import urllib.request

from prepare_soda10m import FILE_ID, resolve_url, sha256, write_json


def retry(operation):
    for attempt in range(3):
        try:
            return operation()
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    root = Path("data/soda10m").resolve()
    local_date = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    evidence = root / f"sources/recheck_{local_date}"
    evidence.mkdir(parents=True, exist_ok=True)
    page_url = "https://soda-2d.github.io/download.html"
    with urllib.request.urlopen(page_url, timeout=45) as response:
        page = response.read()
    (evidence / "download.html").write_bytes(page)
    section = re.search(r"Downloadlink for SODA10M Trainval:(.*?)Downloadlink for SODA10M Test:", page.decode("utf-8"), re.S)
    if not section:
        raise RuntimeError("Could not identify official trainval section")
    links = re.findall(r'href=[\"\x27]([^\"\x27]+)', section[1])
    drive = next(url for url in links if "drive.google.com" in url)
    match = re.search(r"/d/([^/]+)", drive)
    if not match or match[1] != FILE_ID:
        raise RuntimeError("Official trainval file ID differs from acquisition")
    archive_path = root / "archives/labeled_trainval.tar"
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        json_members = [m for m in members if m.isfile() and m.name.endswith(".json")]
        prefix_size = max(m.offset_data + m.size for m in json_members)
    download_url = retry(resolve_url)
    request = urllib.request.Request(download_url, headers={"Range": f"bytes=0-{prefix_size-1}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if "text/" in response.headers.get("Content-Type", ""):
            raise RuntimeError("Remote returned HTML rather than archive")
        http_status = response.status
        response_headers = {k: response.headers.get(k) for k in ("Content-Length", "Content-Range", "ETag", "Last-Modified", "Content-Disposition")}
        if http_status == 206 and not response.headers.get("Content-Range", "").startswith("bytes 0-"):
            raise RuntimeError("Range did not start at archive offset 0")
        prefix = response.read(prefix_size)
    if len(prefix) != prefix_size:
        raise RuntimeError("Remote prefix incomplete")
    (evidence / "official_archive_prefix.bin").write_bytes(prefix)
    comparisons = []
    remote_images = {}
    with tarfile.open(fileobj=io.BytesIO(prefix), mode="r:") as remote:
        for member in remote:
            if not member.name.endswith(".json"):
                continue
            content = remote.extractfile(member).read()
            local = root / "raw" / member.name
            remote_hash = hashlib.sha256(content).hexdigest()
            equal = remote_hash == sha256(local)
            if not equal:
                raise RuntimeError(f"Remote/local JSON mismatch: {member.name}")
            data = json.loads(content)
            split = "train" if "instance_train" in member.name else "val"
            images = data["images"]
            remote_images.update({f"{split}:{im['id']}": im for im in images})
            image_by_id = {im["id"]: im for im in images}
            # Compare exact filenames, not image IDs across splits (IDs overlap).
            disk_files = {p.name for p in (root / "raw/SSLAD-2D/labeled" / split).iterdir() if p.is_file()}
            expected = {im["file_name"] for im in images}
            if disk_files != expected:
                raise RuntimeError(f"Raw {split} images do not match official JSON filenames")
            suffixes = {Path(im["file_name"]).name.split("_")[1] for im in images}
            comparisons.append({"official_split": split, "member": member.name,
                "remote_sha256": remote_hash, "local_sha256": sha256(local),
                "byte_identical": equal, "images": len(images), "boxes": len(data["annotations"]),
                "filename_split_markers": sorted(suffixes), "files_match_exactly": disk_files == expected,
                "period_counts": dict(Counter(im["period"] for im in images)),
                "image_fields": sorted(set(k for im in images for k in im)),
                "daytime_appearing_example": image_by_id.get(3746) if split == "val" else None})
            if len(comparisons) == len(json_members):
                break
    if len(comparisons) != 2:
        raise RuntimeError("Both official annotation files were not compared")
    sample_keys = ("val:3746", "val:395", "val:4395", "val:75", "train:1300", "train:3775", "val:742", "val:3537")
    sample_paths = {f"SSLAD-2D/labeled/{key.split(':')[0]}/{remote_images[key]['file_name']}" for key in sample_keys}
    sampled_members = [m for m in members if m.name in sample_paths]
    if len(sampled_members) != len(sample_paths):
        raise RuntimeError("A source-check sample filename was not found")

    def compare_image(member):
        start, end = member.offset_data, member.offset_data + member.size - 1
        req = urllib.request.Request(download_url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=90) as response:
            expected_range = f"bytes {start}-{end}/"
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected_range):
                raise RuntimeError("Server did not honor the image byte range")
            content = response.read(member.size + 1)
        if len(content) != member.size:
            raise RuntimeError("Remote image member length mismatch")
        digest = hashlib.sha256(content).hexdigest()
        if digest != sha256(root / "raw" / member.name):
            raise RuntimeError("Remote image bytes differ from extracted local raw image")
        return {"member": member.name, "bytes": len(content), "remote_sha256": digest, "local_match": True}

    with ThreadPoolExecutor(max_workers=4) as executor:
        image_comparisons = list(executor.map(compare_image, sampled_members))
    archive_files = [m.name for m in members if m.isfile()]
    if any("/night/" in name.lower() for name in archive_files):
        raise RuntimeError("Unexpected physical night folder in archive")
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_download_page": page_url, "official_trainval_link": drive,
        "file_id_matches_original_acquisition": True,
        "remote_http_status": http_status, "remote_headers": response_headers,
        "fresh_remote_prefix_bytes": len(prefix),
        "fresh_remote_annotations_match_raw": True, "comparisons": comparisons,
        "fresh_remote_image_samples": image_comparisons,
        "local_archive_sha256": sha256(archive_path),
        "archive_members": len(members), "archive_file_count": len(archive_files),
        "non_image_files": [m.name for m in members if m.isfile() and not m.name.lower().endswith(".jpg")],
        "physical_night_folder_exists": False,
        "night_definition_used_by_audit": "official raw instance_val.json images[].period == 'Night'",
        "boundary": "Official train/val retained; no custom split or night-directory copying performed. Test deliberately excluded, not a missing trainval part.",
        "signature_limit": "Fresh HTTPS byte comparison and local hashes; no independent publisher signature available.",
    }
    write_json(evidence / "source_alignment.json", result)
    # Aggregate evidence for local review UI; no image-level annotations here.
    public = {k: v for k, v in result.items() if k not in ("comparisons", "remote_headers")}
    public["comparisons"] = [{k: v for k, v in row.items() if k != "daytime_appearing_example"} for row in comparisons]
    write_json(Path("reports/source_alignment.json"), public)
    print(json.dumps(public, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
