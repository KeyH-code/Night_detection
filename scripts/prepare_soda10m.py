"""Download the official labeled train/val archive; never downloads test/unlabeled.

Raw data, source snapshots and detailed provenance stay under ignored data/.
Uses the Python standard library; supports partial downloads and safe extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

FILE_ID = "1oSJ0rbqNHLmlOOzpmQqXLDraCCQss4Q4"
OFFICIAL = "https://soda-2d.github.io/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class DownloadForm(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = ""
        self.fields = {}
        self.in_form = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action", "")
            self.in_form = True
        elif self.in_form and tag == "input" and attrs.get("name"):
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.in_form = False


def resolve_url() -> str:
    url = "https://drive.google.com/uc?" + urllib.parse.urlencode(
        {"export": "download", "id": FILE_ID})
    with urllib.request.urlopen(url, timeout=45) as response:
        if "text/html" not in response.headers.get("Content-Type", ""):
            return response.url
        parser = DownloadForm()
        parser.feed(response.read().decode("utf-8"))
    parsed = urllib.parse.urlparse(parser.action)
    if parsed.scheme != "https" or parsed.hostname != "drive.usercontent.google.com":
        raise RuntimeError("Official Drive download form unavailable; no fallback mirror used")
    if parser.fields.get("id") != FILE_ID:
        raise RuntimeError("Unexpected download file ID")
    return parser.action + "?" + urllib.parse.urlencode(parser.fields)


def download(root: Path) -> None:
    archives = root / "archives"
    sources = root / "sources"
    archives.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    target = archives / "labeled_trainval.tar"
    partial = target.with_suffix(".tar.part")
    receipt_path = sources / "acquisition.json"
    if target.exists():
        if not receipt_path.exists():
            raise RuntimeError("Existing archive has no acquisition receipt; inspect before reuse")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if target.stat().st_size != receipt["archive_bytes"] or sha256(target) != receipt["archive_sha256"]:
            raise RuntimeError("Existing archive does not match acquisition receipt")
        for snapshot in receipt["source_snapshots"]:
            path = sources / urllib.parse.urlparse(snapshot["url"]).path.rsplit("/", 1)[-1]
            if sha256(path) != snapshot["sha256"]:
                raise RuntimeError("Source snapshot differs from original acquisition receipt")
        print("Verified existing archive", target, flush=True)
        return
    snapshots = []
    for page in ("download.html", "about.html", "documentation.html"):
        snapshot_path = sources / page
        with urllib.request.urlopen(OFFICIAL + page, timeout=45) as response:
            snapshot_path.write_bytes(response.read())
        snapshots.append({"url": OFFICIAL + page, "sha256": sha256(snapshot_path)})
    last_error = None
    for attempt in range(1, 6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            url = resolve_url()
            headers = {"User-Agent": "NightDetection-data-preparation/0.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                if "text/" in response.headers.get("Content-Type", ""):
                    raise RuntimeError("Received a web page instead of archive bytes")
                if offset and response.status != 206:
                    raise RuntimeError("Server did not honor Range; partial preserved for inspection")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Content-Range does not match local partial")
                length = response.headers.get("Content-Length")
                expected = offset + int(length) if length else None
                print(json.dumps({"attempt": attempt, "offset": offset,
                                  "expected_bytes": expected, "status": response.status}), flush=True)
                count = offset
                started = time.monotonic()
                last_report = started
                with partial.open("ab" if offset else "wb") as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
                        count += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            print(f"Downloaded {count / 2**20:.1f} MiB"
                                  f" at {(count-offset)/2**20/(now-started):.2f} MiB/s", flush=True)
                            last_report = now
                if expected is not None and count != expected:
                    raise RuntimeError(f"Incomplete archive: {count} != {expected}")
            # Check archive structure before promoting a partial file to completed.
            with tarfile.open(partial, "r:") as archive:
                members = archive.getmembers()
                if not members:
                    raise RuntimeError("Empty archive")
            archive_hash = sha256(partial)
            partial.rename(target)
            write_json(receipt_path, {
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                "official_download_page": OFFICIAL + "download.html",
                "drive_file_id": FILE_ID, "archive_name": target.name,
                "archive_bytes": count, "archive_sha256": archive_hash,
                "source_snapshots": snapshots, "scope": "official labeled train + val only",
                "checksum_note": "Locally computed SHA-256; no publisher checksum supplied",
            })
            print(f"Download complete: {count} bytes SHA256={archive_hash}", flush=True)
            return
        except Exception as exc:
            last_error = exc
            print(f"Attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            if attempt < 5:
                time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"Download failed; partial retained: {last_error}")


def extract(root: Path) -> None:
    target = root / "archives" / "labeled_trainval.tar"
    destination = root / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((root / "sources" / "acquisition.json").read_text(encoding="utf-8"))
    if target.stat().st_size != receipt["archive_bytes"] or sha256(target) != receipt["archive_sha256"]:
        raise RuntimeError("Archive checksum mismatch")
    with tarfile.open(target, "r:") as archive:
        members = archive.getmembers()
        total = sum(m.size for m in members if m.isfile())
        if shutil.disk_usage(destination).free < total + 2 * 2**30:
            raise RuntimeError("Insufficient disk space for safe extraction")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name or ":" in member.name:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise RuntimeError(f"Link or special file rejected: {member.name}")
            if "test" in path.parts or "Unlabel" in path.parts:
                raise RuntimeError("Archive unexpectedly includes out-of-scope data")
            resolved = (destination / member.name).resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise RuntimeError("Archive path escaped extraction directory")
        print(f"Validated {len(members)} members; {total} extracted bytes", flush=True)
        for index, member in enumerate(members):
            output = destination / member.name
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if output.exists():
                raise RuntimeError(f"Will not overwrite existing extracted file: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, output.open("xb") as sink:
                shutil.copyfileobj(source, sink, 4 * 1024 * 1024)
            if index % 2000 == 0:
                print(f"Extracted {index + 1}/{len(members)} members", flush=True)
    write_json(root / "sources" / "extraction.json", {
        "archive_sha256": receipt["archive_sha256"], "member_count": len(members),
        "extracted_bytes": total, "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print("Extraction complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "extract"))
    parser.add_argument("--data-root", type=Path, default=Path("data/soda10m"))
    args = parser.parse_args()
    root = args.data_root.resolve()
    if args.command == "download":
        download(root)
    else:
        extract(root)


if __name__ == "__main__":
    main()
