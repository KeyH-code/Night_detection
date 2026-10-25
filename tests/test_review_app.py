"""Review provenance and HTTP boundaries; no writes to original dataset."""
from functools import partial
import hashlib
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serve_data_review import ReviewHandler, build_catalog, connected_groups, stratified_sample
from screen_similar_scenes import expanded_pairs, sample_groups


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path(__file__).resolve().parents[1]
        cls.catalog, cls.paths = build_catalog(cls.project)

    def test_night_subset_uses_raw_val_period_not_numeric_id_join(self):
        images = self.catalog["images"]
        self.assertEqual(len(images), 10000)
        self.assertNotEqual(self.paths["train:1"], self.paths["val:1"])
        self.assertEqual(len(self.catalog["sample_sets"]["all"]), 1656)
        for key in self.catalog["sample_sets"]["all"]:
            self.assertEqual((images[key]["official_split"], images[key]["period"]), ("val", "Night"))

    def test_remote_provenance_and_local_display_path_match(self):
        images = self.catalog["images"]
        example = images["val:3746"]
        self.assertEqual(example["file_name"], "HT_VAL_003746_GZ_231.jpg")
        self.assertEqual(example["period"], "Night")
        remote = {row["member"]: row["remote_sha256"] for row in self.catalog["alignment"]["fresh_remote_image_samples"]}
        for key, path in self.paths.items():
            member = "SSLAD-2D/labeled/" + key.split(":")[0] + "/" + images[key]["file_name"]
            if member in remote:
                self.assertEqual(images[key]["sha256"], remote[member])

    def test_graph_keeps_actual_edges_not_transitive_pair_claim(self):
        pairs = [{"a": "val:1", "b": "val:2", "phash_distance": 1, "dhash_distance": 2}, {"a": "val:2", "b": "val:3", "phash_distance": 3, "dhash_distance": 4}]
        group = connected_groups(pairs)[0]
        self.assertEqual(len(group["keys"]), 3)
        self.assertEqual(len(group["edges"]), 2)
        self.assertEqual(connected_groups(list(reversed(pairs)))[0]["id"], group["id"])

    def test_all_candidate_edges_represented_once_and_samples_reproducible(self):
        c = self.catalog
        strict = [g for g in c["groups"] if g["origin"] == "strict"]
        self.assertEqual(sum(len(group["edges"]) for group in strict), 1324)
        self.assertEqual(len(strict), 320)
        chosen = c["sample_sets"]["sample"]
        self.assertEqual(len(set(chosen)), 24)
        rows = [im for im in c["images"].values() if im["key"] in c["sample_sets"]["all"]]
        self.assertEqual(chosen, stratified_sample(list(reversed(rows))))

    def test_scene_sample_cap_history_exclusion_and_reproducibility(self):
        c = self.catalog
        self.assertEqual(len(c["sample_group_ids"]), 50)
        self.assertEqual(len(set(c["sample_group_ids"])), 50)
        self.assertEqual(len(c["historical_group_ids"]), 12)
        self.assertFalse(set(c["sample_group_ids"]) & set(c["historical_group_ids"]))
        chosen, reasons = sample_groups(list(reversed(c["groups"])), set(c["historical_group_ids"]))
        self.assertEqual(chosen, c["sample_group_ids"])
        self.assertEqual(len(reasons), 50)
        self.assertEqual(len(sample_groups(c["groups"], set(), limit=7)[0]), 7)
        with self.assertRaises(ValueError):
            sample_groups(c["groups"], set(), limit=51)
        self.assertFalse(c["scene_screening"]["policy"]["formal_split_created"])
        for group in c["groups"]:
            if group["id"] in chosen:
                self.assertIsNone(group["previous_review"])
                self.assertTrue(group["selection_reasons"])
            for pair in (group["closest_pair"], group["farthest_pair"]):
                self.assertIn(pair["a"], group["keys"])
                self.assertIn(pair["b"], group["keys"])

    def test_expanded_probe_does_not_reuse_strict_images_or_restrict_source(self):
        rows = [{"key": "train:1", "phash": "0", "dhash": "0"},
                {"key": "val:1", "phash": "f", "dhash": "f"},
                {"key": "val:2", "phash": "ff", "dhash": "ff"},
                {"key": "val:3", "phash": "ffffffffffffffff", "dhash": "ffffffffffffffff"}]
        pairs = expanded_pairs(rows, {"val:2"})
        self.assertEqual(pairs, [{"a": "train:1", "b": "val:1", "phash_distance": 4, "dhash_distance": 4}])


class HTTPBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "review_app").mkdir()
        (root / "review_app/index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
        self.image = root / "image.jpg"
        Image.new("RGB", (20, 10), "blue").save(self.image)
        digest = hashlib.sha256(self.image.read_bytes()).hexdigest()
        catalog = {"dataset_fingerprint": "test", "images": {"val:1": {"sha256": digest}}}
        handler = partial(ReviewHandler, project=root, catalog=catalog, paths={"val:1": self.image})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()
        self.tmp.cleanup()

    def test_raw_and_thumbnail_serve_only_allowlisted_images(self):
        with urllib.request.urlopen(self.url + "/image/val%3A1") as response:
            self.assertEqual(response.read(), self.image.read_bytes())
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        with urllib.request.urlopen(self.url + "/thumb/val%3A1") as response:
            self.assertEqual(Image.open(io.BytesIO(response.read())).size, (20, 10))
        for route in ("/image/../image.jpg", "/data/soda10m/raw", "/api/private", "/.env"):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(self.url + route)
            self.assertEqual(error.exception.code, 404)

    def test_changed_image_is_not_displayed_as_verified(self):
        self.image.write_bytes(b"changed")
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.url + "/image/val%3A1")
        self.assertEqual(error.exception.code, 409)

    def test_unrelated_host_is_rejected(self):
        request = urllib.request.Request(self.url + "/api/catalog", headers={"Host": "unrelated.example"})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
