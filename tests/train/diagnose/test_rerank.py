"""Protect reranker candidate identity, scene isolation, crops and checkpoint selection."""
from functools import partial
from pathlib import Path
import sys

from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
import rerank as rr


def row(key, group, purpose, *, single=False):
    box = [1, 2, 8, 16]
    scores = [.40] if single else [.40, .20, .02, .019, .90, .30]
    boxes = [box] * len(scores)
    classes = [0] if single else [0, 0, 0, 0, 1, 0]
    return {'key': key, 'group_id': group, 'purpose': purpose, 'width': 24, 'height': 24,
            'image': f'{key}.png', 'image_sha256': key,
            'gt': {'cls': [0, 1], 'boxes': [box, box], 'original_boxes': [box, box]},
            'pred': {'cls': classes, 'ids': list(range(len(scores))), 'conf': scores,
                     'boxes': boxes, 'original_boxes': boxes},
            'iou_edges': [[0, 0, .90]] if single else
            [[0, 0, .90], [0, 1, .80], [1, 2, .95], [0, 3, .95], [1, 4, .95], [0, 5, .30]]}


def pool(rows):
    rows = [rr.add_ap(rr.candidate_row(r)) for r in rows]
    return rr.Pool(rows, [s for r in rows for s in rr.label_samples(r)])


def test_labels_use_only_in_pool_pedestrian_matches():
    p = pool([row('a', 'g', 'source_train')])
    assert [s.index for s in p.samples] == [0, 1, 2, 5]
    assert [s.kind for s in p.samples] == ['positive', 'duplicate', 'negative', 'gray']
    assert p.samples[0].target == pytest.approx(.90)
    assert p.samples[2].target == 0
    assert [s.index for s in rr.fitting_samples(p, {('a', 2)})] == [0]
    new = rr.with_scores(p, [.001, .0001, .1, .2])[0]
    assert len(new['pred']['ids']) == 4
    assert new['pred']['conf'][0] < .02
    assert {k: v for k, v in new['pred'].items() if k != 'conf'} == {
        k: v for k, v in p.rows[0]['pred'].items() if k != 'conf'}
    with pytest.raises(rr.ContractError, match='one score per candidate'):
        rr.with_scores(p, [.1])
    with pytest.raises(rr.ContractError, match='score identities'):
        rr.with_scores(rr.Pool(p.rows, list(reversed(p.samples))), [.2]*4)


def test_scene_split_is_deterministic_and_rejects_cross_purpose_or_duplicate_image():
    p = pool([row('a', 'same', 'source_train'), row('b', 'same', 'source_train'),
              row('c', 'other', 'source_train'), row('d', 'third', 'source_train')])
    fit, inner = rr.split_train(p)
    rr.isolate(fit, inner)
    assert {r['group_id'] for r in inner.rows} == {'other'}
    reverse = rr.Pool(list(reversed(p.rows)), list(reversed(p.samples)))
    assert {s.key for s in rr.split_train(reverse)[1].samples} == {'c'}
    r = row('outside', 'unseen', 'source_dev')
    r['image_sha256'] = 'a'
    with pytest.raises(rr.ContractError, match='image_sha256'):
        rr.isolate(fit, inner, pool([r]))
    with pytest.raises(rr.ContractError, match='only from source_train'):
        rr.split_train(pool([r]))


def test_crop_rgb_padding_aspect_and_cache(tmp_path):
    image = Image.new('RGB', (24, 24), (255, 33, 0))
    image.save(tmp_path/'crop.png')
    sample = rr.Sample('crop', 0, 'crop.png', (24, 24), (0, 0, 4, 16), 'positive', .8)
    tensor, target = rr.Crops(tmp_path, [sample])[0]
    cached, cached_target = rr.Crops(tmp_path, [sample], cache=True)[0]
    assert torch.equal(tensor, cached) and target == cached_target
    assert tensor.shape == (3, 96, 96)
    assert tensor[:, 0, 48].tolist() == pytest.approx([114/255]*3)
    assert tensor[:, 48, 48].tolist() == pytest.approx([1., 33/255, 0.])
    assert tensor[:, 48, 3].tolist() == pytest.approx([114/255]*3)


def test_no_precision_point_is_not_selected_and_ties_prefer_early_epoch():
    value = {'point': {'t_rule': None, 'recall': .8}, 'ap': {'ap50_95': .45}}
    assert rr.epoch_key(value, 2) is None
    value['point']['t_rule'] = .6
    assert rr.epoch_key(value, 2) < rr.epoch_key(value, 1)


def test_train_and_evaluate_cli_on_synthetic_images(tmp_path, monkeypatch):
    runs = tmp_path/'runs'
    runs.mkdir()
    monkeypatch.setattr(rr, 'RUNS', runs)
    monkeypatch.setattr(rr, 'SOURCE', tmp_path)
    # Fit uses positives and negatives; the internal image has positives only
    # so the synthetic fit can reliably produce a feasible validation threshold.
    train_pool = pool([row('a', 'same', 'source_train'), row('b', 'same', 'source_train'),
                       row('c', 'other', 'source_train', single=True)])
    dev_pool = pool([row('d', 'new', 'source_dev', single=True)])
    for s in train_pool.samples + dev_pool.samples:
        Image.new('RGB', (24, 24), (190, 120, 72)).save(tmp_path/s.image)
    inputs = []

    def build(purpose, identity):
        inputs.append(purpose)
        return {'source_train': train_pool, 'source_dev': dev_pool}[purpose]

    out = runs/'synthetic'
    archive = tmp_path/'r.json'
    rr.write_json(archive, {'metrics': {'per_class': {}}, 'local_reselected_workpoints': {}})
    relative = 'final_eval/source_dev/metrics.json'
    fixture = {'files': {relative: {'sha256': rr.sha256(archive)}}}, {}
    monkeypatch.setattr(rr, 'build_pool', build)
    monkeypatch.setattr(rr, 'verify_r', lambda: fixture)
    monkeypatch.setattr(rr, 'sealed_file', lambda manifest, name: archive)
    monkeypatch.setattr(rr, 'exclusions', lambda p: set())
    monkeypatch.setattr(rr, 'assert_r_baseline', lambda p, rules: rr.pedestrian_metrics(p.rows, rules)[0])
    monkeypatch.setattr(rr, 'compare', partial(rr.compare, repeats=3))
    monkeypatch.setattr(rr, 'RECIPE', dict(rr.RECIPE, epochs=1))
    rr.main(['train', '--output', str(out)])
    assert inputs == ['source_train']
    record = rr.read_json(out/'run.json')
    assert record['status'] == 'complete'
    assert record['best']['epoch'] == 1 and record['epochs'][0]['loss'] > 0
    assert record['fit']['groups_used'] == ['same']
    assert (out/'last.pt').exists()
    rr.main(['evaluate', '--run', str(out)])
    assert inputs == ['source_train', 'source_dev']
    result = out/'source_dev'
    scores = rr.dm.read_rows(result/'scores.jsonl.gz')
    assert scores[0]['key'] == 'd' and scores[0]['index'] == 0 and scores[0]['box'] == [1, 2, 8, 16]
    metrics = rr.read_json(result/'metrics.json')
    assert metrics['arms']['R']['point']['tp'] == 1
    assert metrics['arms']['rerank']['point']['tp'] == 1
    assert metrics['unchanged_classes'] == rr.NAMES[1:]
    assert metrics['transitions'] == []
    assert (result/'pr.npz').exists()
    # An interrupted or unqualified fit cannot provide a model for source_dev.
    rr.write_json(out/'run.json', dict(record, status='no_feasible_inner_point', best=None))
    with pytest.raises(rr.ContractError, match='training run status'):
        rr.load_selected(out, torch.device('cpu'), rr.load_rules(rr.RULES))
