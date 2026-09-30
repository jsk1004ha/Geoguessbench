# Geoguessbench

**실제 거리 이미지로 멀티모달 AI의 위치 추론과 능동 탐색을 평가하는 실행 가능한 벤치마크.**

사진이나 점수를 만들어 보여주는 데모가 아닙니다. 실제 이미지를 가져오고 → 모델이 이미지와 허용 행동만 관측하고 → 출발 좌표를 제출하면 → 별도 평가 서버가 정답과 비교합니다. Python 3.11 이상에서 동작하며, 모델 학습은 필요하지 않습니다.

## 어떤 방식인가요?

| 트랙 | 모델이 할 수 있는 일 | 필요한 데이터 |
|---|---|---|
| **NMPZ** | 고정 화면 한 장을 보고 좌표 제출 | OSV-5M 테스트 이미지 또는 실제 사진 |
| **NM** | 제자리 회전·상하 시선·확대 후 좌표 제출 | 방향이 알려진 실제 360° 파노라마 |
| **Moving** | 회전·확대·실제 연결 프레임 이동 후 **출발 위치** 제출 | 실제 파노라마 시퀀스/검증된 이동 그래프 |
| **Moving + mask** | Moving과 동일하되 원본 이미지 하단 일부를 마스킹 | 같은 Moving 데이터; 메타 단서 민감도 실험 |

GeoGuessr의 NMPZ/No Move/Moving 구분에 대응하는 독립 평가 환경입니다. **GeoGuessr 사이트에 접속해 게임을 자동 플레이하는 봇도, 공식 제휴 제품도 아닙니다.** OSV-5M 공개 구현과 동일한 거리 기반 점수식과 반경 정확도를 사용하지만, 이 저장소의 탐색 예산·실패 처리·이미지 전처리는 명시적으로 버전 관리하는 자체 프로토콜입니다. 공식 GeoGuessr 점수나 WanderBench 전체 실험을 그대로 재현했다는 뜻이 아닙니다.

## 빠르게 시작하기: 실제 사진 100문항

### 1. 설치

```bash
git clone https://github.com/jsk1004ha/Geoguessbench.git
cd Geoguessbench
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux / WSL / macOS: source .venv/bin/activate
python -m pip install -e ".[data,dev]"
```

### 2. 실제 데이터 다운로드

```bash
geoguessbench fetch-osv --out data/osv-test100 --limit 100 --seed 42
geoguessbench validate --data data/osv-test100/manifest.json --track nmpz
```

공식 `osv5m/osv5m`의 **test.csv 전체에서 seed 고정 균등 표본**을 선택합니다. Hugging Face 데이터 리비전을 커밋 SHA로 고정하고, 테스트 ZIP의 필요한 파일만 HTTP Range로 읽습니다. 100문항 결과는 **OSV-5M 테스트 부분집합 결과**이지 전체 테스트셋 공식 성적이 아닙니다. 파일이 빠지거나 호스트가 Range를 지원하지 않으면 다른 사진으로 교체하거나 가짜 데이터로 진행하지 않고 실패합니다. 메타데이터 다운로드와 ZIP 색인 읽기도 필요하므로 다운로드 크기를 임의로 보장하지 않습니다.

### 3. 평가 서버 실행

```bash
geoguessbench serve --data data/osv-test100/manifest.json --config configs/nmpz.json
```

브라우저에서 `http://127.0.0.1:8000`을 열면 사람도 동일한 관측·행동·채점 규칙으로 참여할 수 있습니다. 정답 파일은 서버에만 둡니다. 서버를 외부 인터페이스에 바인딩하려면 `BENCH_TOKEN`을 설정해야 합니다.

### 4. 별도 터미널에서 실제 모델 연결

같은 가상환경을 활성화한 다음 API 키를 환경변수에 설정합니다. `.env.example`은 설명용이며 **자동으로 읽지 않습니다**.

```bash
# Linux / WSL / macOS
export MODEL_API_KEY='YOUR_API_KEY'
# Windows PowerShell에서는 대신:
# $env:MODEL_API_KEY='YOUR_API_KEY'

geoguessbench evaluate --model YOUR_VISION_MODEL_ID --out results/model-nmpz.json
```

`YOUR_VISION_MODEL_ID`는 해당 서비스 계정에서 실제 사용할 수 있는 **이미지 입력 모델 ID**로 바꿉니다. API 사용료는 해당 제공자의 정책에 따라 발생할 수 있습니다. 이 명령이 모델을 자동으로 다운로드하거나 학습하지는 않습니다.

기본 주소는 OpenAI의 Chat Completions API입니다. 같은 요청 형식을 지원하는 로컬 VLM 서버도 사용할 수 있습니다.

```bash
geoguessbench evaluate --model YOUR_LOCAL_VISION_MODEL_ID --base-url http://127.0.0.1:8001/v1 --token-parameter max_tokens --out results/local-nmpz.json
```

자체 호스팅 엔드포인트가 키를 요구하지 않으면 `MODEL_API_KEY`를 비워 두세요. 이미지 data URL과 Chat Completions 응답 형식 지원 여부는 서버에서 확인해야 합니다. **네이티브 Anthropic/Gemini API 어댑터는 이 버전에 포함되지 않습니다.** 추론 모델과의 호환성을 위해 temperature는 지정하지 않으면 보내지 않습니다.

## 실제로 이동하는 평가

OSV-5M의 평면 사진을 가짜 360° 화면으로 취급하지 않습니다. Moving/NM에는 **실제 2:1 equirectangular 파노라마와 카메라 방향**이 필요합니다.

```bash
# Linux / WSL / macOS; PowerShell은 $env:MAPILLARY_ACCESS_TOKEN='...'
export MAPILLARY_ACCESS_TOKEN='YOUR_MAPILLARY_ACCESS_TOKEN'
geoguessbench fetch-mapillary --sequence REAL_PANORAMIC_SEQUENCE_ID --out data/routes
geoguessbench validate --data data/routes/manifest.json --track moving
geoguessbench serve --data data/routes/manifest.json --config configs/moving.json --db private/moving.sqlite
```

`--sequence`를 반복해 여러 실제 시퀀스를 넣을 수 있습니다. 각 시퀀스에서 최대 40개 프레임을 확인하고, 파노라마를 시간순으로 정렬한 뒤 기본 75m 이내의 인접 프레임만 연결합니다. 실제 도로의 모든 교차로를 복원하는 것은 아니며, **해당 시퀀스의 제한된 이동 경로**입니다. 파노라마가 없거나 연결 가능한 경로가 없으면 명확한 오류로 종료합니다. 국가 균형과 난이도는 시퀀스를 수집하는 평가자가 설계해야 합니다.

자체 보유한 실제 이미지도 CSV로 가져올 수 있습니다.

```bash
geoguessbench import-csv --csv locations.csv --images raw-images --out data/custom --name custom-v1 --license YOUR_IMAGE_LICENSE
```

형식과 방향 규약은 [데이터 문서](docs/DATASET.md)를 확인하세요.

## 결과 및 비교

실제 평가가 모두 끝나면 JSON을 저장합니다. 실패 문항을 제외해 성적이 높아지는 방식은 사용하지 않습니다.

- **위치 성능:** 평균 거리 점수, Acc@1/25/200/750/2500km, 유효 답변의 평균·중앙 거리, 실패에 지구 반둘레를 적용한 별도 거리 지표.
- **탐색·불확실성:** 행동 수·비용·시간, 시퀀스/그래프 단위 cluster bootstrap 95% 구간, 선택적으로 제출한 25km 이내 확률의 Brier score와 제출률.
- **재현성:** 데이터·프로토콜·구현 SHA-256, 패키지/의존성 버전, 프롬프트 해시, 실제 설정, 모델 응답의 사용량. 제공자가 토큰 수를 주지 않으면 `null`과 coverage로 표시합니다.

```bash
# 중단 후 동일 출력 경로·모델 설정으로 이어가기: RUN_ID는 실행 시 출력됨
geoguessbench evaluate --model YOUR_VISION_MODEL_ID --out results/model-nmpz.json --resume RUN_ID

# 서버 DB에서 완료 결과 다시 내보내기
geoguessbench export --db private/runs.sqlite --run RUN_ID --out results/public.json

# 비교 가능한 설정끼리 묶는 로컬 리더보드 JSON
geoguessbench leaderboard results/model-a.json results/model-b.json --out results/leaderboard.json

# 평가자만: paired comparison용 정답/추정/전체 로그 내보내기
geoguessbench export --db private/runs.sqlite --run RUN_ID --out private/model-a.json --private
geoguessbench compare --left private/model-a.json --right private/model-b.json --out results/comparison.json
```

공개 API는 평가 도중 정답·중간 점수를 공개하지 않습니다. `--private` 파일과 SQLite DB에는 정답이 있으므로 에이전트나 공개 저장소에 제공하면 안 됩니다. 체크포인트에는 관측 이미지와 모델 문맥이 들어 있으므로 데이터 이용 조건에 맞게 관리하세요. 로컬 리더보드는 **서명·심사를 갖춘 공인 공개 리더보드가 아닙니다**.

## 구현 범위와 검증 범위

완성된 기능은 데이터 유효성 검사, 실제 이미지 수집 경로, 파노라마 투영, 제한된 탐색 환경, 별도 평가 API, VLM 실행기, 영속 저장/재개, 채점·통계·비교, 반응형 브라우저 콘솔입니다. 정답 좌표, 원본 ID, 이미지 URL, EXIF/XMP는 모델 관측에 전달하지 않습니다. API 오류를 다른 모델이나 임의 좌표로 대체하지 않습니다.

단위·API·어댑터 테스트는 **합성 픽스처를 명시적으로 표시**하고 실행합니다. 일반 CLI는 픽스처 데이터셋을 거부하며, 픽스처 결과는 리더보드에서 제외됩니다. 자동 테스트 통과는 실제 모델의 위치 추정 성능을 검증했다는 뜻이 아닙니다. 초기 구현 환경에서는 외부 네트워크/API 키를 사용한 OSV-5M·Mapillary 다운로드 및 유료 모델 평가를 끝까지 실행하지 못했습니다. 해당 경로는 모의 HTTP 응답 기반 계약 테스트로 확인했습니다. 실제 성능표는 사용자가 동일 고정 데이터로 평가를 실행한 뒤 생성됩니다.

다음은 의도적으로 보장하지 않습니다: 학습 데이터 오염의 완전한 부재, 실제 지리 이해와 암기의 완전한 분리, 모든 메타 단서 제거, 실제 GeoGuessr 모든 지도/기능 복제, 검색 도구 트랙, 악의적인 참가자를 방어하는 공개 대회 수준의 격리. 경쟁 평가에서는 별도 컨테이너/호스트와 네트워크 정책으로 **에이전트의 정답 파일 접근 및 비허용 검색을 차단**해야 합니다.

## 개발

```bash
python -m pip install -e ".[dev]"
python -m pytest --cov=geoguessbench --cov-report=term-missing
python -m ruff check .
node --check geoguessbench/web/app.js
```

GitHub Actions는 Linux/Windows × Python 3.11/3.12에서 설치·lint·테스트를 실행하도록 구성되어 있습니다. 실제 통과 여부는 저장소 Actions 결과를 확인하세요.

- [평가 규약](docs/PROTOCOL.md)
- [데이터 제작과 보안 경계](docs/DATASET.md)
- 참고: [OSV-5M 공식 코드](https://github.com/gastruc/osv5m), [거리 점수 구현](https://github.com/gastruc/osv5m/blob/main/metrics/distance_based.py), [공식 데이터 다운로드](https://github.com/gastruc/osv5m/blob/main/DATASET.md), [Learning to Wander / WanderBench](https://arxiv.org/abs/2603.10463).

코드는 기존 저장소의 MIT 라이선스를 유지합니다. 데이터·이미지·지도에는 원 제공자의 별도 이용 조건이 적용됩니다. API 키와 데이터 원본은 저장소에 커밋하지 마세요.
