import base64
import io
import json
import math
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from geoguessbench.dataset import Dataset, safe_path
from geoguessbench.engine import Episode
from geoguessbench.geo import bearing, geo_score, haversine
from geoguessbench.metrics import cluster_ci, summarize
from geoguessbench.render import perspective
from geoguessbench.schema import Action, Protocol

@pytest.mark.parametrize('a,b,expected', [((0, 0), (0, 0), 0), ((0, 0), (0, 180), math.pi * 6371), ((0, 179.9), (0, -179.9), 22.238985), ((90, 0), (90, 100), 0)])
def test_haversine(a, b, expected):
    assert haversine(a, b) == pytest.approx(expected, abs=1e-05)
    assert haversine(a, b) == pytest.approx(haversine(b, a))

def test_score_and_bearing():
    assert geo_score(0) == 5000
    assert geo_score(1492.7) == pytest.approx(5000 / math.e)
    assert bearing((0, 0), (0, 1)) == pytest.approx(90)
    assert bearing((0, 0), (1, 0)) == pytest.approx(0)

@pytest.mark.parametrize('d', [-1, float('nan'), float('inf')])
def test_invalid_score(d):
    with pytest.raises(ValueError):
        geo_score(d)

@pytest.mark.parametrize('action', [{'type': 'submit', 'lat': 91, 'lon': 0}, {'type': 'submit', 'lat': 0, 'lon': 181}, {'type': 'submit', 'lat': float('nan'), 'lon': 0}, {'type': 'submit', 'lat': 0}, {'type': 'move', 'exit': '0'}, {'type': 'move', 'exit': -1}, {'type': 'turn'}, {'type': 'zoom', 'fov': 10}, {'type': 'look', 'degrees': 61}, {'type': 'abstain', 'lat': 0}, {'type': 'submit', 'lat': 0, 'lon': 0, 'confidence_25km': 1.1}, {'type': 'search', 'query': 'hidden'}, {'type': 'abstain', 'extra': 'unexpected'}])
def test_invalid_actions(action):
    with pytest.raises(ValidationError):
        Action.model_validate(action)

def test_valid_zero_coordinate():
    assert Action(type='submit', lat=0, lon=0).lat == 0

def test_fixture_refused(dataset_factory):
    with pytest.raises(ValueError, match='fixture'):
        Dataset(dataset_factory())

@pytest.mark.parametrize('name', ['../secret', '/tmp/secret', 'C:\\secret', 'images/../../secret', '..\\secret'])
def test_path_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        safe_path(tmp_path, name)

@pytest.mark.parametrize('mutation,match', [(lambda m: m['nodes'][0].update(sha256='0' * 64), 'hash mismatch'), (lambda m: m['nodes'][0].update(neighbors=['absent']), 'missing'), (lambda m: m['rounds'][0].update(start='absent'), 'missing'), (lambda m: m['nodes'].append(m['nodes'][0].copy()), 'duplicate'), (lambda m: m['rounds'][1].update(split='train'), 'split leakage'), (lambda m: m['nodes'][0].update(heading=None), 'heading')])
def test_dataset_validation(dataset_factory, mutation, match):
    with pytest.raises(ValueError, match=match):
        Dataset(dataset_factory(mutate=mutation), allow_fixture=True)

def test_spatial_split_leakage(dataset_factory):

    def mutate(m):
        m['nodes'][2].update(lat=10, lon=20.0001)
        m['rounds'].append({'id': 'other', 'start': 'secret-c', 'split': 'train', 'group': 'other'})
    with pytest.raises(ValueError, match='geographic buffer'):
        Dataset(dataset_factory(mutate=mutate), allow_fixture=True)

def test_dateline_split_leakage(dataset_factory):

    def mutate(m):
        for node in m['nodes'][:2]:
            node.update(lat=0, lon=179.9999)
        m['nodes'][2].update(lat=0, lon=-179.9999)
        m['rounds'].append({'id': 'other', 'start': 'secret-c', 'split': 'train', 'group': 'other'})
    with pytest.raises(ValueError, match='geographic buffer'):
        Dataset(dataset_factory(mutate=mutate), allow_fixture=True)

def test_flat_track_refused(dataset_factory):
    data = Dataset(dataset_factory(panorama=False), allow_fixture=True)
    data.validate_track('nmpz', 'test')
    for track in ['nm', 'moving']:
        with pytest.raises(ValueError, match='panoramas'):
            data.validate_track(track, 'test')
    obs = Episode(data, Protocol(width=128, height=128), 'round-secret-1').observation()
    assert obs['heading'] is None and obs['fov'] is None

def test_projection_changes_real_view():
    raw = np.zeros((100, 200, 3), dtype='uint8')
    raw[:, :, 0] = np.arange(200)[None, :]
    raw[:, :, 1] = np.arange(100)[:, None]
    image = Image.fromarray(raw)
    a = np.array(perspective(image, 0, 0, 40, 128, 128))
    b = np.array(perspective(image, 90, 0, 40, 128, 128))
    up = np.array(perspective(image, 0, 30, 40, 128, 128))
    assert b[64, 64, 0] > a[64, 64, 0] + 40
    assert up[64, 64, 1] < a[64, 64, 1] - 10
    assert np.array_equal(np.array(perspective(image, 360, 0, 40, 128, 128)), a)

def test_observation_no_labels_or_metadata(dataset):
    obs = Episode(dataset, Protocol(width=128, height=128), 'round-secret-1').observation()
    text = json.dumps({k: v for k, v in obs.items() if k != 'image'})
    for forbidden in ['secret-a', 'source_url', 'sequence', 'round-secret', 'lat', 'lon', 'country', 'target', 'XX']:
        assert forbidden not in text
    image = Image.open(io.BytesIO(base64.b64decode(obs['image'].split(',')[1])))
    assert not image.getexif()
    assert image.size == (128, 128)

def test_move_scores_original_start(dataset):
    ep = Episode(dataset, Protocol(track='moving', width=128, height=128), 'round-secret-1')
    first = ep.observation()
    ep.step({'type': 'move', 'exit': 0}, 0)
    second = ep.observation()
    assert ep.state['node'] == 'secret-b' and first['image'] != second['image']
    ep.step({'type': 'submit', 'lat': 10, 'lon': 20}, 1)
    assert ep.state['result']['score'] == 5000
    assert ep.state['result']['cost'] == 2
    assert 'target' not in ep.observation()

def test_forbidden_move_consumes_budget(dataset):
    ep = Episode(dataset, Protocol(track='nm', max_actions=2, width=128, height=128), 'round-secret-1')
    ep.step({'type': 'move', 'exit': 0}, 0)
    assert ep.state['node'] == 'secret-a' and ep.state['actions'] == 1 and (ep.state['cost'] == 1)
    ep.step({'type': 'turn', 'degrees': 30}, 1)
    assert ep.observation()['final_only']
    ep.step({'type': 'submit', 'lat': 10, 'lon': 20}, 2)
    assert ep.state['result']['status'] == 'ok'

def test_budget_cannot_be_bypassed(dataset):
    ep = Episode(dataset, Protocol(track='moving', max_cost=1, width=128, height=128), 'round-secret-1')
    ep.step({'type': 'move', 'exit': 0}, 0)
    assert ep.state['node'] == 'secret-a' and ep.final_only
    ep.step({'type': 'turn', 'degrees': 30}, 1)
    assert ep.state['status'] == 'invalid' and ep.state['cost'] == 1

def test_nmpz_no_extra_views(dataset):
    ep = Episode(dataset, Protocol(track='nmpz', width=128, height=128), 'round-secret-1')
    ep.step({'type': 'zoom', 'fov': 40}, 0)
    assert ep.state['status'] == 'invalid' and ep.state['result']['score'] == 0

def test_stale_and_double_submit(dataset):
    ep = Episode(dataset, Protocol(track='moving', width=128, height=128), 'round-secret-1')
    ep.step({'type': 'turn', 'degrees': 30}, 0)
    with pytest.raises(ValueError, match='stale'):
        ep.step({'type': 'turn', 'degrees': 30}, 0)
    assert ep.state['actions'] == 1
    ep.step({'type': 'submit', 'lat': 10, 'lon': 20}, 1)
    with pytest.raises(ValueError, match='finished'):
        ep.step({'type': 'submit', 'lat': 0, 'lon': 0}, 2)

def test_timeout(dataset):
    now = [100]
    ep = Episode(dataset, Protocol(time_limit_s=5, width=128, height=128), 'round-secret-1', clock=lambda: now[0])
    now[0] = 106
    assert ep.observation()['status'] == 'timeout'
    assert ep.state['result']['score'] == 0 and ep.state['result']['elapsed_s'] == 5

def test_metrics_failures_in_denominator(dataset):
    p = Protocol(width=128, height=128)
    good = Episode(dataset, p, 'round-secret-1')
    good.step({'type': 'submit', 'lat': 10, 'lon': 20}, 0)
    bad = Episode(dataset, p, 'round-secret-2')
    bad.step({'type': 'abstain'}, 0)
    rows = [good.state['result'], bad.state['result']]
    report = summarize(rows, samples=100)
    assert report['mean_score'] == 2500 and report['accuracy_at_km']['25'] == 0.5
    assert report['mean_error_km_valid_only'] == 0
    assert report['mean_error_km_failure_penalty'] == pytest.approx(math.pi * 6371 / 2)
    assert report['brier_25km'] is None and report['confidence_coverage'] == 0
    assert report['mean_score_ci95']['clusters'] == 1
    assert report['status_counts'] == {'abstain': 1, 'ok': 1}
    assert cluster_ci(rows, [1, 0], 100, 42) == cluster_ci(rows, [1, 0], 100, 42)


def test_usage_missing_tokens_are_not_zero():
    from geoguessbench.metrics import telemetry_summary
    traces = [{"events": [
        {"telemetry_self_reported": {"prompt_tokens": 25, "completion_tokens": 3}},
        {"telemetry_self_reported": {"error": "HTTPStatusError", "prompt_tokens": None}},
    ]}]
    result = telemetry_summary(traces)
    assert result["calls"] == 2 and result["provider_errors"] == 1
    assert result["prompt_tokens"] is None
    assert result["prompt_tokens_known_sum"] == 25
    assert result["prompt_tokens_coverage"] == 0.5


def test_usage_complete_tokens_are_summed():
    from geoguessbench.metrics import telemetry_summary
    result = telemetry_summary([{"events": [
        {"telemetry_self_reported": {"prompt_tokens": 25, "completion_tokens": 3}},
        {"telemetry_self_reported": {"prompt_tokens": 40, "completion_tokens": 4}},
    ]}])
    assert result["prompt_tokens"] == 65 and result["completion_tokens"] == 7
    assert result["prompt_tokens_coverage"] == 1
