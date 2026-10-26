"""Semantic data tests; standard library only, including Chinese relocation."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import make_splits as s


class SplitTests(unittest.TestCase):
    def row(self, i, ped=True, empty=False):
        return {"key": f"train:{i}", "official_split": "train", "period": "Daytime", "file_name": f"图_{i}.jpg", "width": 100, "height": 80, "city": "上海", "location": "道路", "weather": "晴天", "box_count": 0 if empty else 1, "boxes_by_class": {} if empty else {"Pedestrian" if ped else "Car": 1}}

    def test_allocation_deterministic_group_integrity_conservation(self):
        rows = [self.row(i, i % 2 == 0) for i in range(30)]
        protocol = {"selected_groups": [{"id": "related", "keys": ["train:0", "train:1", "train:2"]}]}
        groups = s.group_rows(rows, protocol)
        first = s.allocate(groups, {"a": 20, "b": 5, "c": 5})
        self.assertEqual(first, s.allocate(groups, {"a": 20, "b": 5, "c": 5}))
        self.assertEqual(sum(map(len, first.values())), 30)
        assignments = {r["key"]: p for p, rs in first.items() for r in rs}
        self.assertEqual(len({assignments[f"train:{i}"] for i in range(3)}), 1)

    def test_explicitly_removed_relation_has_no_constraint(self):
        groups = s.group_rows([self.row(0), self.row(1)], {"selected_groups": []})
        self.assertEqual(len(groups), 2)

    def test_unplaceable_group_fails_without_split(self):
        rows = [self.row(i) for i in range(20)]
        groups = s.group_rows(rows, {"selected_groups": [{"id": "unplaceable", "keys": [r["key"] for r in rows]}]})
        with self.assertRaisesRegex(ValueError, "Cannot place intact group"):
            s.allocate(groups, {"a": 10, "b": 10})

    def test_cross_pool_relation_is_explicit_failure(self):
        with self.assertRaisesRegex(ValueError, "crosses official"):
            s.group_rows([self.row(0)], {"selected_groups": [{"id": "cross", "keys": ["train:0", "val:0"]}]})

    def test_night_minimum_is_hard_constraint(self):
        rows = [dict(self.row(i), group_id=str(i)) for i in range(10)]
        with self.assertRaisesRegex(ValueError, "pedestrian images 10 < 40"):
            s.validate_assignment({"night_dev": rows}, {"night_dev": 10}, True)

    def test_yolo_roundtrip_empty_and_unknown_category(self):
        mapping = [{"official_id": 3, "training_id": 2}]
        self.assertEqual(s.label_bytes(self.row(0), [], mapping), b"")
        self.assertEqual(s.label_bytes(self.row(0), [{"category_id": 3, "bbox": [0, 0, 100, 80]}], mapping), b"2 0.500000000000 0.500000000000 1.000000000000 1.000000000000\n")
        with self.assertRaisesRegex(ValueError, "Invalid normalized"):
            s.label_bytes(self.row(0), [{"category_id": 3, "bbox": [-1, 0, 2, 2]}], mapping)

    def fixture(self, base):
        root, version = base / "数据", base / "冻结"
        mapping = [{"official_id": i+1, "official_name": name.title(), "training_id": i, "name": name} for i, name in enumerate(["pedestrian", "cyclist", "car", "truck", "tram", "tricycle"])]
        allocation, integrity, annotations = {}, {}, {}
        for i, purpose in enumerate(("source_train", "source_dev", "source_test")):
            row = dict(self.row(i, empty=i == 1), group_id=f"g{i}")
            allocation[purpose] = [row]
            raw = root / "raw/SSLAD-2D/labeled/train" / row["file_name"]
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(f"original-image-{i}".encode())
            integrity[row["key"]] = {"sha256": s.sha256(raw)}
            annotations[row["key"]] = [] if i == 1 else [{"id": i, "category_id": 1, "bbox": [3, 4, 12, 15]}]
        protocol = {"normalization_version": "test", "fingerprint_components": {"raw_annotation_sha256": {"train": "raw"}, "normalized_annotation_sha256": {"train": "norm"}}}
        prepared = allocation, integrity, annotations, mapping, protocol
        ready = s.candidate_ready(root, version, "source", prepared, write=True)
        s.write_json(version / "READY.json", ready)
        return root, version, prepared, ready

    def test_chinese_relocation_hashes_whitelist_empty_and_read_only_verify(self):
        with tempfile.TemporaryDirectory(prefix="中文划分_") as temp:
            root, version, prepared, ready = self.fixture(Path(temp))
            before = {p: (s.sha256(p), p.stat().st_mtime_ns) for p in version.rglob("*") if p.is_file()}
            result = s.validate_version(root, version, "source", None, prepared=prepared)
            self.assertEqual(result["empty_images"]["source_dev"], 1)
            self.assertEqual(len(ready["files"]), 4)
            self.assertFalse(any(f["purpose"] == "source_test" for f in ready["files"]))
            self.assertEqual(before, {p: (s.sha256(p), p.stat().st_mtime_ns) for p in before})
            renamed = version.with_name("移动后的冻结目录")
            version.rename(renamed)
            self.assertEqual(s.validate_version(root, renamed, "source", None, prepared=prepared)["status"], "passed")
            target = renamed / ready["files"][0]["path"]
            target.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                s.validate_version(root, renamed, "source", None, prepared=prepared)

    def test_freeze_rejects_metadata_or_test_injection(self):
        with tempfile.TemporaryDirectory(prefix="中文_") as temp:
            root, version, prepared, ready = self.fixture(Path(temp))
            modified = copy.deepcopy(ready)
            modified["files"].append({"path": "labels/source_test/test.txt", "sha256": "bad", "kind": "label", "purpose": "source_test"})
            with self.assertRaisesRegex(ValueError, "deterministic recomputation"):
                s.validate_version(root, version, "source", None, ready=modified, prepared=prepared)

    def test_paths_and_exclusive_publisher(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for invalid in ("../secret", "C:/secret", "a\\b", "/secret"):
                with self.assertRaises(ValueError):
                    s.safe_path(root, invalid)
            version = root / "source_v1"
            version.mkdir()
            with self.assertRaises(FileExistsError):
                version.mkdir(exist_ok=False)

    def test_legacy_verifier_uses_manifest_derived_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse(s.frozen_state(root)["formal_split_created"])
            version = root / "splits/source_v1"
            version.mkdir(parents=True)
            s.write_json(version / "manifest.json", {"records": []})
            s.write_json(version / "READY.json", {"status": "frozen", "version": "source_v1", "counts": {}, "manifests": {"source_train": {"path": "manifest.json", "sha256": s.sha256(version / "manifest.json")}}, "held_out": {}})
            self.assertTrue(s.frozen_state(root)["formal_split_created"])
            self.assertFalse(s.frozen_state(root)["final_test_frozen"])


if __name__ == "__main__":
    unittest.main()
