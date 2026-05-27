# PORMAKE 테스트 기반 도입 설계

- 작성일: 2026-05-28
- 작성자: Sangwon Lee (with Claude)
- 상태: Draft → User Review

## 배경

PORMAKE는 위상(topology)과 빌딩블록(building block)을 조합해 다공성 물질을 생성하는 Python 라이브러리이지만, 현재 테스트 코드가 0개다. CI(`.github/workflows/pip-test.yaml`)는 `pip install -e .`만 실행해 "설치가 되는가" 수준만 검증한다. 그 결과:

- jax/ase/pymatgen 같은 무거운 의존성 업그레이드 시 깨짐을 빠르게 감지하지 못한다.
- 내부 코드 정리·리팩토링을 시도할 안전망이 없다.
- 실사용자 시나리오(HKUST-1 빌드 등)가 깨져도 머지 후에야 발견된다.

본 설계는 이 세 문제를 동시에 해결하는 최소·실용적 테스트 기반을 정의한다.

## 목표

1. **회귀 방지** — README의 핵심 시나리오가 항상 동작함을 보장
2. **의존성 호환성 검증** — Python·OS·라이브러리 매트릭스에서 자산 로딩과 빌드가 깨지지 않음
3. **리팩토링 안전망** — 협력자 적은 핵심 로직에 단위 테스트를 두어 내부 정리에 자신감 확보

비목표: 100% 라인 커버리지, mock 기반 격리, property-based 테스트, `experimental/decomposer`·`app/` 테스트, pre-commit 도구 업그레이드.

## 사용자 결정사항 요약

| 항목 | 결정 |
|------|------|
| 일차 목표 | 회귀 + 의존성 + 리팩토링 셋 다 |
| 실행 모델 | 마커 분리 없이 `pytest` 한 방에 |
| 수치 검증 | 구조적 불변량 + `pytest.approx` 허용 오차 |
| CI 매트릭스 | Python 3.10/3.11/3.12 × Ubuntu/macOS, 설치 + 전체 pytest |

## 1. 테스트 아키텍처 (3계층)

```
tests/
├── unit/          # 협력자 적은 순수 로직 (빠름, 다수)
├── integration/   # 빌드 시나리오 end-to-end (느림, 핵심만)
└── smoke/         # 데이터 자산·import 스모크 (얕고 빠름)
```

| 계층 | 검증 대상 | 협력자 | 예상 테스트 수 | 1회 실행 |
|------|-----------|--------|----------------|---------|
| unit | xyz/cgd 파싱, RMSD 계산, 격자 변환, 토폴로지 메타데이터 | 적음 | 20~30개 | <5초 |
| integration | HKUST-1 빌드, edge BB 삽입, low-symmetry MOF, chimera MOF, CIF write/read 왕복 | 많음 (JAX 포함) | 5~8개 | 30~90초 |
| smoke | `import pormake`, 대표 BB/topology 로드 | DB만 | 4~5개 (parametrize 다수) | <10초 |

전부 `pytest`로 일괄 실행. 마커 분리 없음. 무거운 객체(`Database()`, `Builder()`, `Locator()`)는 session-scoped fixture로 단 한 번만 생성.

**범위 밖**: `src/pormake/experimental/decomposer`, `src/pormake/app/`, 867개 BB / 2479개 topology 전수 테스트.

## 2. 디렉토리 구조 + 픽스처

```
tests/
├── conftest.py
├── helpers.py
├── unit/
│   ├── test_building_block.py
│   ├── test_topology.py
│   ├── test_locator.py
│   ├── test_scaler.py
│   ├── test_database.py
│   └── test_framework.py
├── integration/
│   ├── test_build_hkust1.py
│   ├── test_build_with_edge_bb.py
│   ├── test_build_low_symmetry.py
│   ├── test_build_chimera.py
│   └── test_cif_roundtrip.py
└── smoke/
    ├── test_import.py
    ├── test_database_samples.py
    └── test_topology_samples.py
```

### conftest.py 핵심 픽스처

```python
import pytest
import pormake

@pytest.fixture(scope="session")
def database():
    return pormake.Database()

@pytest.fixture(scope="session")
def builder():
    return pormake.Builder()

@pytest.fixture(scope="session")
def locator():
    return pormake.Locator()
```

### 결정 근거

- **session 스코프**: `Database()`는 디렉토리 스캔, `Builder()`는 JAX 초기화 비용이 크다. 불변 객체로 취급해 테스트마다 새로 만들지 않는다.
- **mock/patch 없음**: 모든 게 실제 객체. JAX·ASE·pymatgen 모두 진짜로 호출. (testing 스킬 "Real Object First")
- **co-location 안 함**: PORMAKE는 외부 사용자용 라이브러리. `src/`는 깔끔하게 유지하고 테스트는 루트 `tests/`에 집결.
- **golden 파일 없음**: 사용자 결정대로 불변량 + 허용 오차만.

## 3. 수치 검증 전략

JAX 최적화/플랫폼 BLAS 차이로 좌표는 일정 부분 흔들린다. "정확히 같아야 할 것"과 "근사적으로 같으면 될 것"을 명확히 구분한다.

### Exact (정확히 일치)

| 항목 | 검증 방법 |
|------|----------|
| 원자 개수 | `len(atoms) == 624` |
| 원소 구성 | `Counter(atoms.get_chemical_symbols()) == {...}` |
| 공간군 | `topology.space_group == "Fm-3m"` |
| 슬롯 수 / 노드 타입 수 | `topology.n_slots`, `len(unique_node_types)` |
| 연결성 | networkx 그래프 `n_edges` |
| `bb.copy()` 동등성 | 원소·연결점 인덱스 일치 |

### Approximate (`pytest.approx`)

| 항목 | 허용 오차 | 근거 |
|------|----------|------|
| RMSD (대칭 노드) | `rel=1e-2`, `abs=0.05` | 대칭 매칭은 안정적 |
| RMSD (비대칭 노드) | `rel=5e-2` | 정렬 결과가 더 흔들림 |
| 격자 파라미터 | `rel=1e-3` | scaler 수렴 오차 |
| 좌표 통계량 | `rel=1e-3` | 좌표 자체 비교는 안 함 |

### 의도적 비대상

- 좌표 element-wise 비교 — 원자 순서가 빌드 알고리즘에 따라 바뀔 수 있어 false positive 다발
- CIF 텍스트 비교 — 부동소수점 출력 포맷 차이로 깨짐. 항상 `ase.io.read`로 다시 읽어 객체 단위 비교
- golden CIF 파일

### 헬퍼

```python
# tests/helpers.py
from collections import Counter
import pytest

def assert_structure_invariants(atoms, *, n_atoms, composition, lattice=None):
    assert len(atoms) == n_atoms
    assert Counter(atoms.get_chemical_symbols()) == composition
    if lattice is not None:
        assert atoms.cell.cellpar() == pytest.approx(lattice, rel=1e-3)
```

### 기준값 채취

대표 시나리오(HKUST-1 등)는 PR 시작 시 `main` 기준으로 한 번 실행해 `n_atoms`, `composition`, `cellpar`를 채취하고 테스트 코드에 하드코드한다. 채취값·채취 환경(OS·Python·jax 버전)을 PR 설명에 기록해 이력을 남긴다. 이후 CI에서 다른 환경의 결과가 달라지면 허용 오차 범위에서 흡수되거나, 흡수 안 되면 그 시점에 원인 분석 후 기준값을 갱신한다.

## 4. 의존성·CI 매트릭스 정리

### pyproject.toml

```toml
[project]
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[dependency-groups]
dev = [
    "notebook>=7.3.2",
    "pytest>=8.0",
    "pytest-xdist>=3.5",
]
```

- `pytest-cov`/`hypothesis` 미포함 (YAGNI).
- `pytest-xdist`만 예외 추가: 통합 테스트가 각 10초 이상 걸리므로 `pytest -n auto`로 병렬화.

### CI: `.github/workflows/test.yaml`

```yaml
name: Test
on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
        os: ["ubuntu-22.04", "macos-14"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync --all-extras --dev
      - run: uv run pytest -n auto -v
```

### 기존 `pip-test.yaml` 대비 변경

| 항목 | Before | After | 이유 |
|------|--------|-------|------|
| 트리거 브랜치 | main, test | main, dev | `test` 미사용, `dev` 활성 |
| Python | 3.8/3.9/3.10 | 3.10/3.11/3.12 | jax 0.4.13/pymatgen 최신이 3.10+ |
| OS | ubuntu-20.04, 22.04 | ubuntu-22.04, macos-14 | 20.04 deprecated, macOS는 jax CPU 빌드 검증 |
| 실행 | `pip install -e .` | `uv sync` + `pytest` | 설치+동작 일원화 |
| 패키지 매니저 | pip | uv | 프로젝트 표준에 정렬 |

### Windows 제외

README가 "jax는 Linux/macOS만 지원, Windows는 WSL"이라고 명시. CI에도 Windows job 없음.

### 노이즈 정리 (Stage 1에 포함)

- `runtime.log` (13MB) `.gitignore` 추가 + `git rm`
- 루트 `HKUST1.cif` `.gitignore` + `git rm` (예시는 `example/` 안에서 생성됨)

## 5. 단계별 도입 계획

테스트 0개 → 전체 매트릭스 통과까지 한 PR로 묶지 않고 **4단계 PR**로 점진 도입.

### Stage 1 — Foundation (PR #1, S)

- `pyproject.toml`: pytest/pytest-xdist 추가, Python 3.10+ 상향
- `tests/conftest.py` (session fixtures)
- `tests/smoke/test_import.py` 1개 — `import pormake` + 모든 public symbol 접근
- `.github/workflows/test.yaml` 신규, `pip-test.yaml` 삭제
- `runtime.log`, 루트 `HKUST1.cif` `.gitignore` + `git rm`

**완료 기준**: 매트릭스 6칸(3 Python × 2 OS) 전부 녹색.

### Stage 2 — Smoke (PR #2, S)

- `test_database_samples.py` — 20개 대표 BB parametrize 로드
- `test_topology_samples.py` — 20개 대표 topology parametrize 로드 + `unique_local_structures` 접근
- 대표 샘플 선정 기준: README에 등장하는 모든 이름 + 각 CN별 1개씩

**완료 기준**: 의존성 한 줄 변경으로 깨질 항목이 모두 감지됨.

### Stage 3 — Unit (PR #3, M)

- `test_building_block.py` — xyz 파서(bond block 있을 때/없을 때), `copy()`, connection point 추출
- `test_topology.py` — cgd 로드, `node_types`/`edge_types` 길이, `n_slots` 일관성
- `test_locator.py` — `calculate_rmsd` (acs/N198 ≈ 0.02, pcu/N198 ≈ 0.42)
- `test_database.py` — 없는 키 → 적절한 에러
- `test_framework.py` — `write_cif` 후 `ase.io.read` 왕복

**완료 기준**: 20~30개 단위 테스트, 전체 <5초.

### Stage 4 — Integration (PR #4, M)

- `test_build_hkust1.py` — README 예제 1
- `test_build_with_edge_bb.py` — README 예제 2 (E41)
- `test_build_low_symmetry.py` — README 예제 5 (ith + N3 + N114)
- `test_build_chimera.py` — README 예제 3
- `test_cif_roundtrip.py` — write → read → 조성/격자 일치

**완료 기준**: 5개 시나리오 통과, 전체 <90초.

### 단계 간 의존성

```
Stage 1 (인프라) ──> Stage 2 (smoke)
                ╰──> Stage 3 (unit)         } 병렬 가능
                ╰──> Stage 4 (integration)
```

Stage 2/3/4는 Stage 1 머지 이후 병렬 가능.

### 명시적 비범위

- pre-commit 도구 업그레이드 (black/isort/flake8 → ruff)
- `experimental/decomposer` 테스트
- `app/`(streamlit) 테스트
- 커버리지 측정·뱃지
- Hypothesis property-based 테스트

## 위험 요소와 대응

| 위험 | 대응 |
|------|------|
| jax 버전 차이로 RMSD 값이 README와 다르게 측정 | 채취값을 README가 아닌 **현재 main 실제 실행 결과**로 고정. README 값은 참고만 |
| macOS CI에서 jax 설치 실패 | Stage 1 머지 전 실제 매트릭스 실행으로 확인. 실패 시 macOS-14 → macOS-13 fallback |
| session-scoped fixture 공유로 테스트 간 오염 | `bb.copy()`를 의무화. fixture 객체를 직접 변경하지 않는 컨벤션 도큐먼트화 |
| Stage 4 통합 테스트가 90초 초과 | `pytest-xdist`로 병렬. 그래도 초과 시 `slow` 마커 도입 (Stage 4 시점 재검토) |

## 비범위 / 후속 작업

본 설계 외 작업은 별도 spec/PR로 진행:

- pre-commit 도구 현대화
- `experimental/decomposer` 테스트
- `app/` 웹 UI 테스트
- 커버리지 측정 도입
- property-based 테스트 도입 검토
