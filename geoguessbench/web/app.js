'use strict';
const $ = id => document.getElementById(id);
let current = null, runId = null, busy = false, latestReport = null, deadline = 0;
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
async function api(path, body, method) {
  const headers = { 'Content-Type': 'application/json' };
  if ($('token').value) headers.Authorization = 'Bearer ' + $('token').value;
  const response = await fetch(path, { method: method || (body === undefined ? 'GET' : 'POST'), headers,
    ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '요청 형식을 확인하세요.');
  return data;
}
async function task(fn) {
  if (busy) return;
  busy = true;
  try { await fn(); } catch (error) { message(error.message, true); }
  finally { busy = false; updateControls(); }
}
function updateControls() {
  const active = current && !current.done && !busy;
  document.querySelectorAll('#controls button, #exits button').forEach(button => {
    button.disabled = !active || current.final_only;
  });
  $('submit').disabled = !active; $('abstain').disabled = !active;
  $('start').disabled = busy; $('resume').disabled = busy;
}
async function config() {
  const data = await api('/api/config');
  $('connection').textContent = data.fixture ? '테스트 픽스처 · 실제 결과 아님' : '평가 서버 연결됨';
  $('track').textContent = data.protocol.track.toUpperCase();
  $('protocol-details').textContent = `${data.rounds} rounds · ${data.protocol.max_actions} actions · ${data.protocol.time_limit_s}s / round`;
  $('fingerprint').textContent = 'DATASET / ' + data.dataset_sha256.slice(0, 16);
}
function showObservation(obs, index, total) {
  current = obs;
  if (obs.done) { updateControls(); return; }
  $('empty').hidden = true; $('view').hidden = false; $('view').src = obs.image;
  if (index !== undefined) $('round').textContent = `ROUND ${index + 1} / ${total}`;
  $('heading').textContent = obs.heading == null ? '정보 없음' : `${Math.round(obs.heading)}°`;
  $('fov').textContent = obs.fov == null ? '고정 화면' : `${obs.fov}°`;
  $('budget').textContent = obs.actions_remaining; $('cost').textContent = obs.cost_remaining;
  deadline = Date.now() + obs.time_remaining_s * 1000;
  $('exits').replaceChildren();
  for (const exit of obs.exits) {
    const button = document.createElement('button');
    button.textContent = `이동 ${exit.exit + 1} · ${exit.bearing}°`;
    button.addEventListener('click', () => task(() => act({ type: 'move', exit: exit.exit })));
    $('exits').append(button);
  }
  message(obs.error ? '허용되지 않은 행동입니다. 남은 예산을 확인하세요.' :
    obs.final_only ? '탐색 없이 출발 위치를 제출하거나 기권하세요.' : '화면을 탐색하고 출발 위치의 좌표를 제출하세요.', Boolean(obs.error));
  updateControls();
}
async function next() {
  const data = await api(`/api/runs/${runId}/next`, {});
  if (data.complete) {
    current = null; deadline = 0; $('round').textContent = '평가 완료';
    showReport(await api(`/api/runs/${runId}/report`));
    message('모든 문항의 평가가 완료되었습니다.'); return;
  }
  if (data.observation.done) return next();
  showObservation(data.observation, data.round_index, data.total);
}
async function act(action) {
  if (!current || current.done) return;
  const data = await api(`/api/runs/${runId}/step`, { episode: current.episode, step: current.step, action });
  if (data.observation.done) {
    $('lat').value = ''; $('lon').value = ''; $('confidence').value = '';
    await next();
  } else showObservation(data.observation);
}
$('start').addEventListener('click', () => task(async () => {
  await config();
  const data = await api('/api/runs', { model: $('participant').value.trim() || 'human', settings: { adapter: 'human-console', external_tools: false } });
  runId = data.run_id; $('run-id').value = runId; $('results').hidden = true;
  await next();
}));
$('resume').addEventListener('click', () => task(async () => {
  runId = $('run-id').value.trim();
  if (!/^[a-f0-9]{32}$/.test(runId)) throw new Error('올바른 Run ID를 입력하세요.');
  await config(); await next();
}));
$('controls').addEventListener('click', event => {
  const target = event.target.closest('button[data-type]'); if (!target || target.disabled) return;
  const action = { type: target.dataset.type };
  for (const key of ['degrees', 'fov']) if (target.dataset[key] !== undefined) action[key] = Number(target.dataset[key]);
  task(() => act(action));
});
$('submit').addEventListener('click', () => task(async () => {
  if (!$('lat').value.trim() || !$('lon').value.trim()) throw new Error('위도와 경도를 모두 입력하세요.');
  const lat = Number($('lat').value), lon = Number($('lon').value);
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180)
    throw new Error('위도는 −90~90, 경도는 −180~180 범위여야 합니다.');
  const action = { type: 'submit', lat, lon };
  if ($('confidence').value !== '') {
    action.confidence_25km = Number($('confidence').value);
    if (!Number.isFinite(action.confidence_25km) || action.confidence_25km < 0 || action.confidence_25km > 1)
      throw new Error('확률은 0~1 범위여야 합니다.');
  }
  await act(action);
}));
$('abstain').addEventListener('click', () => task(() => act({ type: 'abstain' })));
function showReport(report) {
  if (!report.metrics || typeof report.metrics.mean_score !== 'number' || !report.metrics.accuracy_at_km)
    throw new Error('완료된 Geoguessbench 결과 JSON이 아닙니다.');
  latestReport = report; $('results').hidden = false;
  $('report-label').textContent = `${report.fixture ? '[합성 테스트 · 벤치마크 결과 아님] ' : ''}${report.model} · ${report.protocol.track.toUpperCase()} · Run ${report.id}`;
  const m = report.metrics;
  const stats = [['평균 위치 점수 / 5000', m.mean_score.toFixed(1)], ['25 km 이내 정확도', (m.accuracy_at_km['25'] * 100).toFixed(1) + '%'],
    ['유효 답변 / 전체 문항', `${m.valid} / ${m.n}`], ['평균 탐색 행동', m.mean_actions.toFixed(1)]];
  $('summary').replaceChildren();
  for (const [label, value] of stats) {
    const block = document.createElement('div'); block.className = 'stat';
    const small = document.createElement('span'), strong = document.createElement('strong');
    small.textContent = label; strong.textContent = value; block.append(small, strong); $('summary').append(block);
  }
  $('accuracy').replaceChildren();
  for (const [radius, value] of Object.entries(m.accuracy_at_km)) {
    const tr = document.createElement('tr'), a = document.createElement('td'), b = document.createElement('td');
    a.textContent = `< ${Number(radius).toLocaleString()} km`; b.textContent = (value * 100).toFixed(2) + '%';
    tr.append(a, b); $('accuracy').append(tr);
  }
}
$('load-report').addEventListener('click', () => $('report-file').click());
$('report-file').addEventListener('change', () => task(async () => {
  const file = $('report-file').files[0]; if (!file) return;
  if (file.size > 10 * 1024 * 1024) throw new Error('공개 결과 JSON만 여세요. 최대 크기는 10 MB입니다.');
  showReport(JSON.parse(await file.text()));
}));
$('download-report').addEventListener('click', () => {
  if (!latestReport) return;
  const blob = new Blob([JSON.stringify(latestReport, null, 2)], { type: 'application/json' });
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'geoguessbench-report.json';
  link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
setInterval(() => { $('timer').textContent = deadline && current && !current.done ? `${Math.max(0, Math.ceil((deadline - Date.now()) / 1000))}s` : '—'; }, 250);
config().catch(error => { $('connection').textContent = '토큰 또는 연결 확인'; message(error.message, true); });
