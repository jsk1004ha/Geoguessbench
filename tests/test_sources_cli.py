import csv
import io
import sys
import types
import zipfile
import httpx
import pytest
from PIL import Image
from geoguessbench.cli import main
from geoguessbench.dataset import Dataset
from geoguessbench.sources import RangeReader, fetch_mapillary, fetch_osv, import_csv, write_image

def png(red=10):
    stream = io.BytesIO()
    Image.new('RGB', (256, 128), (red, 20, 30)).save(stream, 'PNG')
    return stream.getvalue()

def range_transport(raw):

    def handler(request):
        header = request.headers.get('range', '')
        start, end = map(int, header.removeprefix('bytes=').split('-'))
        return httpx.Response(206, headers={'content-range': f'bytes {start}-{end}/{len(raw)}'}, content=raw[start:end + 1])
    return httpx.MockTransport(handler)

def test_remote_zip_selective_read():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('path/123.png', png())
    with httpx.Client(transport=range_transport(buffer.getvalue())) as client:
        with RangeReader(client, 'https://data.example/archive.zip') as remote, zipfile.ZipFile(remote) as archive:
            assert archive.read('path/123.png') == png()
            assert remote.seekable() and remote.readable()
            remote.seek(0, 2)
            assert remote.read() == b''

@pytest.mark.parametrize('status,headers,body', [(200, {}, b'huge'), (206, {'content-range': 'wrong'}, b'x'), (206, {'content-range': 'bytes 0-0/10'}, b'xx')])
def test_range_fails_closed(status, headers, body):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(status, headers=headers, content=body))) as client:
        with pytest.raises(ValueError):
            RangeReader(client, 'https://data.example/archive.zip')

def test_image_import_strips_exif(tmp_path):
    stream = io.BytesIO()
    image = Image.new('RGB', (20, 20))
    exif = Image.Exif()
    exif[270] = 'SECRET GPS LOCATION'
    image.save(stream, 'JPEG', exif=exif)
    path, sha = write_image(stream.getvalue(), tmp_path)
    with Image.open(tmp_path / path) as clean:
        assert not clean.getexif()
    assert len(sha) == 64 and b'SECRET' not in (tmp_path / path).read_bytes()

def test_csv_import(tmp_path):
    (tmp_path / 'real-input.png').write_bytes(png())
    table = tmp_path / 'input.csv'
    table.write_text('id,image,lat,lon\nA,real-input.png,10,20\n')
    path = import_csv(table, tmp_path, tmp_path / 'out', name='test ingestion only', license_name='test license')
    dataset = Dataset(path)
    assert dataset.nodes['A'].lat == 10 and len(dataset.rounds) == 1
    with pytest.raises(ValueError):
        import_csv(table, tmp_path, tmp_path / 'out', name='x', license_name='x')

def test_osv_downloader_with_http_fixture(tmp_path, monkeypatch):
    metadata = tmp_path / 'test.csv'
    with metadata.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'latitude', 'longitude', 'country'])
        writer.writeheader()
        writer.writerows([{'id': str(i), 'latitude': 10 + i, 'longitude': 20 + i, 'country': 'XX'} for i in range(3)])
    hub = types.SimpleNamespace(HfApi=lambda: types.SimpleNamespace(dataset_info=lambda *a, **k: types.SimpleNamespace(sha='a' * 40)), hf_hub_download=lambda *a, **k: str(metadata))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for i in range(3):
            archive.writestr(f'{i}.jpg', png(10 + i))
    client_type = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client_type(transport=range_transport(buffer.getvalue()), **kwargs))
    path = fetch_osv(tmp_path / 'osv', limit=2, seed=42)
    data = Dataset(path)
    assert len(data.rounds) == 2 and data.manifest.provenance['hf_revision'] == 'a' * 40
    assert data.manifest.source.endswith('(subset)')
    assert all((not node.panorama for node in data.nodes.values()))
    with pytest.raises(ValueError):
        data.validate_track('moving', 'test')

def test_mapillary_real_route_adapter_with_http_fixture(tmp_path, monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.host == 'images.example':
            return httpx.Response(200, content=png(10 + int(request.url.path[1:])))
        assert request.headers['authorization'] == 'OAuth test-token'
        if request.url.path == '/image_ids':
            return httpx.Response(200, json={'data': [{'id': '2'}, {'id': '1'}]})
        key = int(request.url.path[1:])
        return httpx.Response(200, json={'id': str(key), 'captured_at': key * 1000, 'is_pano': True, 'computed_geometry': {'coordinates': [20 + key * 0.0001, 10]}, 'computed_compass_angle': 0, 'thumb_2048_url': f'https://images.example/{key}', 'creator': {'username': 'test author'}})
    client_type = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs))
    path = fetch_mapillary(tmp_path / 'm', ['sequence'], 'test-token', max_frames=2)
    data = Dataset(path)
    data.validate_track('moving', 'test')
    assert data.nodes['1'].neighbors == ['2'] and data.nodes['2'].neighbors == ['1']
    assert 'test author' in data.nodes['1'].attribution
    assert all(('authorization' not in r.headers for r in seen if r.url.host == 'images.example'))
    with pytest.raises(ValueError, match='TOKEN'):
        fetch_mapillary(tmp_path / 'x', ['a'], '')

def test_cli_refuses_missing_data_and_fixture(dataset_factory, capsys):
    assert main(['validate', '--data', 'this-file-does-not-exist.json']) == 2
    assert main(['validate', '--data', str(dataset_factory())]) == 2
    assert 'fixture' in capsys.readouterr().err

def test_cli_nonlocal_requires_token(monkeypatch, capsys):
    monkeypatch.delenv('BENCH_TOKEN', raising=False)
    assert main(['serve', '--host', '0.0.0.0', '--data', 'not-read', '--config', 'not-read']) == 2
    assert 'BENCH_TOKEN' in capsys.readouterr().err
