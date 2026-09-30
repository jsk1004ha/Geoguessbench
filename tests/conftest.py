"""Synthetic fixtures are restricted to tests and marked in every manifest/report."""
import hashlib
import json
import numpy as np
import pytest
from PIL import Image
from geoguessbench.dataset import Dataset

@pytest.fixture
def dataset_factory(tmp_path):
    number = 0

    def create(*, panorama=True, fixture=True, mutate=None):
        nonlocal number
        number += 1
        root = tmp_path / f'fixture-{number}'
        root.mkdir()
        nodes = []
        for i, (key, lat, lon) in enumerate((('secret-a', 10, 20), ('secret-b', 10, 20.0005), ('secret-c', -20, 110))):
            image = np.zeros((128, 256, 3), dtype=np.uint8)
            image[:, :, 0] = (np.arange(256)[None, :] + i * 31) % 256
            image[:, :, 1] = np.arange(128)[:, None]
            image[:, :, 2] = 50 + i * 50
            path = root / f'{key}.png'
            Image.fromarray(image).save(path)
            nodes.append({'id': key, 'image': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'lat': lat, 'lon': lon, 'panorama': panorama, 'heading': 0 if panorama else None, 'sequence': 'seq-ab' if i < 2 else 'seq-c', 'neighbors': ['secret-b'] if i == 0 else ['secret-a'] if i == 1 else [], 'source_url': 'https://example.invalid/secret-source'})
        manifest = {'schema_version': 1, 'name': 'TEST FIXTURE - NOT REAL IMAGERY', 'version': '1', 'source': 'synthetic unit test', 'license': 'test only', 'fixture': fixture, 'nodes': nodes, 'rounds': [{'id': 'round-secret-1', 'start': 'secret-a', 'split': 'test', 'group': 'group-ab', 'country': 'XX'}, {'id': 'round-secret-2', 'start': 'secret-b', 'split': 'test', 'group': 'group-ab', 'country': 'XX'}]}
        if mutate:
            mutate(manifest)
        path = root / 'manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        return path
    return create

@pytest.fixture
def dataset(dataset_factory):
    return Dataset(dataset_factory(), allow_fixture=True)


@pytest.fixture(autouse=True)
def close_evaluator_connections(monkeypatch):
    from geoguessbench.server import Evaluator
    original = Evaluator.__init__
    instances = []

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        instances.append(self)

    monkeypatch.setattr(Evaluator, "__init__", initialize)
    yield
    for evaluator in instances:
        evaluator.close()
