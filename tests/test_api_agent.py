import json
import httpx
import pytest
from fastapi.testclient import TestClient
from geoguessbench.agent import ChatAgent, parse_action, run_agent, trim_images
from geoguessbench.cli import export_run, leaderboard
from geoguessbench.metrics import compare
from geoguessbench.schema import Protocol
from geoguessbench.server import create_app

@pytest.fixture
def service(dataset, tmp_path):
    db = tmp_path / 'runs.sqlite'
    protocol = Protocol(track='moving', width=128, height=128, bootstrap_samples=100)
    app = create_app(dataset, protocol, db)
    with TestClient(app) as client:
        yield (client, app, db)
    app.state.evaluator.close()

def start(client):
    response = client.post('/api/runs', json={'model': 'TEST-MOCK-NOT-A-REAL-MODEL', 'settings': {}})
    assert response.status_code == 200
    return response.json()['run_id']

def send(client, run_id, obs, action):
    return client.post(f'/api/runs/{run_id}/step', json={'episode': obs['episode'], 'step': obs['step'], 'action': action})

def test_api_hides_answers_and_requires_completion(service):
    client, app, db = service
    run = start(client)
    current = client.post(f'/api/runs/{run}/next').json()
    obs = current['observation']
    assert client.get(f'/api/runs/{run}/report').status_code == 409
    assert client.get('/data/manifest.json').status_code == 404
    assert client.get('/private/runs.sqlite').status_code == 404
    assert client.get('/openapi.json').status_code == 404
    assert client.get('/').headers['content-security-policy'].startswith("default-src 'self'")
    assert client.get('/app.js').status_code == 200
    assert client.get('/health').json()['status'] == 'ok'
    assert send(client, run, obs, {'type': 'turn', 'degrees': 30}).status_code == 200
    assert send(client, run, obs, {'type': 'turn', 'degrees': 30}).status_code == 409
    assert 'target' not in json.dumps(current)
    assert app.state.evaluator.load(run)['active']['actions'] == 1

def test_token_required(dataset, tmp_path):
    app = create_app(dataset, Protocol(width=128, height=128), tmp_path / 'token.sqlite', token='test-only-key')
    with TestClient(app) as client:
        assert client.get('/api/config').status_code == 401
        assert client.get('/api/config', headers={'Authorization': 'Bearer test-only-key'}).status_code == 200
        assert client.post('/api/runs', json={'model': 'mock'}).status_code == 401
    app.state.evaluator.close()

def test_state_survives_restart(service, dataset):
    client, app, db = service
    run = start(client)
    obs = client.post(f'/api/runs/{run}/next').json()['observation']
    send(client, run, obs, {'type': 'turn', 'degrees': 45})
    other = create_app(dataset, app.state.evaluator.protocol, db)
    with TestClient(other) as second:
        restored = second.post(f'/api/runs/{run}/next').json()['observation']
        assert restored['step'] == 1 and restored['heading'] == 45
        assert restored['episode'] == obs['episode']
    other.state.evaluator.close()

def test_old_episode_cannot_replay_into_new_round(service):
    client, _, _ = service
    run = start(client)
    old = client.post(f'/api/runs/{run}/next').json()['observation']
    send(client, run, old, {'type': 'abstain'})
    new = client.post(f'/api/runs/{run}/next').json()['observation']
    assert old['step'] == new['step'] == 0 and old['episode'] != new['episode']
    assert send(client, run, old, {'type': 'submit', 'lat': 0, 'lon': 0}).status_code == 409

def test_invalid_payload_budget_and_unknown_run(service):
    client, app, _ = service
    assert client.post('/api/runs/unknown/next').status_code == 404
    run = start(client)
    obs = client.post(f'/api/runs/{run}/next').json()['observation']
    response = send(client, run, obs, {'type': 'invented-tool', 'value': 'anything'})
    assert response.status_code == 200
    assert app.state.evaluator.load(run)['active']['actions'] == 1

@pytest.mark.parametrize('text', [None, 'bad JSON', '[]', '{"type":"submit","lat":NaN,"lon":0}', 'x' * 16001])
def test_model_parser_never_fakes_coordinates(text):
    assert parse_action(text) == {'type': 'invalid'}

def test_model_parser_fences():
    assert parse_action('```json\n{"type":"abstain"}\n```') == {'type': 'abstain'}

def test_image_memory_policy():
    messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': str(i)}}]} for i in range(5)]
    result = trim_images(messages, 2)
    assert sum((part['type'] == 'image_url' for message in result for part in message['content'])) == 2
    assert all((message['content'][0]['type'] == 'image_url' for message in messages))

def mock_agent(calls, *, status=200):

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(status, json={'model': 'test-mock', 'choices': [{'message': {'content': '{"type":"submit","lat":10,"lon":20,"confidence_25km":0.8}'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 123, 'completion_tokens': 20}})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ChatAgent('test-mock', 'https://model.example/v1', 'SECRET-ONLY-IN-HEADER', client=client)

def test_model_request_has_pixels_not_labels(service):
    client, _, _ = service
    run = start(client)
    obs = client.post(f'/api/runs/{run}/next').json()['observation']
    calls = []
    agent = mock_agent(calls)
    action, usage = agent.act(obs)
    assert action['type'] == 'submit' and usage['prompt_tokens'] == 123
    body = json.dumps(calls[0])
    for forbidden in ['secret-a', 'round-secret', 'SECRET-ONLY-IN-HEADER', 'image_sha256', obs['episode']]:
        assert forbidden not in body
    assert calls[0]['messages'][-1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert 'temperature' not in calls[0]
    assert 'max_completion_tokens' in calls[0]

def test_provider_error_abstains(service):
    client, _, _ = service
    run = start(client)
    obs = client.post(f'/api/runs/{run}/next').json()['observation']
    action, usage = mock_agent([], status=401).act(obs)
    assert action['type'] == 'abstain' and 'lat' not in action
    assert usage['prompt_tokens'] is None and usage['error'] == 'HTTPStatusError'

def test_full_runner_and_private_export(service, tmp_path):
    client, _, db = service
    calls = []
    agent = mock_agent(calls)
    path = tmp_path / 'report.json'
    report = run_agent('http://evaluator', agent, path, client=client)
    assert len(calls) == 2 and report['metrics']['n'] == 2
    assert report['metrics']['accuracy_at_km']['25'] == 1
    assert report['metrics']['mean_score'] > 4999
    assert report['fixture'] is True
    assert 'rows' not in report and 'target' not in json.dumps(report)
    assert not path.with_suffix('.checkpoint.json').exists()
    private = export_run(db, report['id'], True)
    assert len(private['rows']) == 2 and private['rows'][0]['target']['lat'] == 10
    delta = compare(private, private, samples=100)
    assert delta['mean_score_difference'] == 0 and delta['ci95']['low'] == 0
    with pytest.raises(ValueError, match='fixture'):
        leaderboard([path])
    altered = {**private, 'protocol_sha256': 'different'}
    with pytest.raises(ValueError, match='incomparable'):
        compare(private, altered)

@pytest.mark.parametrize('already_sent', [False, True])
def test_exact_pending_action_resume(service, tmp_path, already_sent):
    client, _, _ = service
    calls = []
    agent = mock_agent(calls)
    run = client.post('/api/runs', json={'model': agent.model, 'settings': agent.settings()}).json()['run_id']
    first = client.post(f'/api/runs/{run}/next').json()
    obs = first['observation']
    action, telemetry = agent.act(obs)
    pending = {'episode': obs['episode'], 'step': obs['step'], 'action': action, 'telemetry': telemetry}
    path = tmp_path / f'resume-{already_sent}.json'
    path.with_suffix('.checkpoint.json').write_text(json.dumps({'run_id': run, 'settings': agent.settings(), 'round_index': first['round_index'], 'messages': agent.messages, 'pending': pending}))
    if already_sent:
        client.post(f'/api/runs/{run}/step', json=pending)
    resumed = mock_agent(calls)
    report = run_agent('http://evaluator', resumed, path, client=client, resume=run)
    assert report['metrics']['n'] == 2 and len(calls) == 2

def test_settings_do_not_expose_key():
    agent = mock_agent([])
    assert 'SECRET-ONLY-IN-HEADER' not in json.dumps(agent.settings())
    with pytest.raises(ValueError):
        ChatAgent('m', 'https://name:secret@example.com/v1')
    with pytest.raises(ValueError):
        ChatAgent('m', 'http://remote.example/v1')
