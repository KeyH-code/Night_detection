"""Small offline checks for archive safety and image audit failure handling."""
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_soda10m import download, extract, sha256, write_json
from audit_soda10m import duplicate_groups, inspect_image, near_duplicate_pairs
from normalize import clip_box, normalize


class ArchiveSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="中文路径_")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def archive(self, name, content=b"test", symlink=False):
        target = self.root / "archives/labeled_trainval.tar"
        target.parent.mkdir(parents=True)
        with tarfile.open(target, "w") as archive:
            member = tarfile.TarInfo(name)
            if symlink:
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside"
                archive.addfile(member)
            else:
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        write_json(self.root / "sources/acquisition.json", {
            "archive_bytes": target.stat().st_size, "archive_sha256": sha256(target)})

    def test_chinese_path_extracts_exact_bytes(self):
        self.archive("SSLAD-2D/labeled/train/image.jpg", b"unaltered-bytes")
        extract(self.root)
        self.assertEqual((self.root / "raw/SSLAD-2D/labeled/train/image.jpg").read_bytes(), b"unaltered-bytes")

    def test_reusing_archive_preserves_source_receipt_without_network(self):
        self.archive("SSLAD-2D/labeled/train/image.jpg")
        snapshot = self.root / "sources/about.html"
        snapshot.write_text("original terms snapshot", encoding="utf-8")
        receipt_path = self.root / "sources/acquisition.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["source_snapshots"] = [{"url": "https://soda-2d.github.io/about.html", "sha256": sha256(snapshot)}]
        write_json(receipt_path, receipt)
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("Network must not be used")):
            download(self.root)
        self.assertEqual(snapshot.read_text(encoding="utf-8"), "original terms snapshot")

    def test_parent_path_rejected(self):
        self.archive("../outside.txt")
        with self.assertRaisesRegex(RuntimeError, "Unsafe archive path"):
            extract(self.root)

    def test_symlink_rejected(self):
        self.archive("SSLAD-2D/labeled/train/link.jpg", symlink=True)
        with self.assertRaisesRegex(RuntimeError, "Link or special file"):
            extract(self.root)

    def test_out_of_scope_test_rejected(self):
        self.archive("SSLAD-2D/labeled/test/image.jpg")
        with self.assertRaisesRegex(RuntimeError, "out-of-scope"):
            extract(self.root)

    def test_existing_raw_file_is_not_overwritten(self):
        self.archive("SSLAD-2D/labeled/train/image.jpg")
        target = self.root / "raw/SSLAD-2D/labeled/train/image.jpg"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"keep-existing")
        with self.assertRaisesRegex(RuntimeError, "Will not overwrite"):
            extract(self.root)
        self.assertEqual(target.read_bytes(), b"keep-existing")

    def test_corruption_and_dimension_mismatch_recorded(self):
        metadata = {"key": "train:1", "file_name": "image.jpg", "width": 20, "height": 10}
        path = self.root / "image.jpg"
        path.write_bytes(b"not-an-image")
        self.assertFalse(inspect_image((path, metadata))["ok"])
        Image.new("RGB", (10, 10), "red").save(path)
        result = inspect_image((path, metadata))
        self.assertTrue(result["ok"])
        self.assertFalse(result["dimension_matches"])

    def test_duplicate_screening_keeps_split_qualified_ids(self):
        rows = [dict(key=key, ok=True, sha256="same", decoded_rgb_sha256="same", phash="0"*16, dhash="0"*16)
                for key in ("train:1", "val:1")]
        self.assertEqual(duplicate_groups(rows, "sha256"), [["train:1", "val:1"]])
        pairs = near_duplicate_pairs(rows)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["decoded_identical"])

    def test_clip_box_preserves_opposite_corner_and_original(self):
        original = [-1, 4, 12, 8]
        self.assertEqual(clip_box(original, 10, 10), [0, 4, 10, 6])
        self.assertEqual(original, [-1, 4, 12, 8])
        with self.assertRaisesRegex(ValueError, "empty box"):
            clip_box([11, 2, 1, 3], 10, 10)

    def test_normalization_refuses_frozen_or_existing_version(self):
        output = self.root / 'normalized/annotations'
        output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            normalize(self.root)
        ready = self.root / 'splits/source_v1/READY.json'
        ready.parent.mkdir(parents=True)
        ready.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Frozen data'):
            normalize(self.root)


if __name__ == "__main__":
    unittest.main()
