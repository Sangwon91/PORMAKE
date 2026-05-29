# Testing Foundation — Stage 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PORMAKE에 pytest가 동작하는 최소 인프라를 도입하고 CI를 모든 매트릭스(Python 3.10/3.11/3.12 × Ubuntu/macOS)에서 녹색으로 만든다.

**Architecture:** 루트에 `tests/` 디렉토리 신설(`unit/`, `integration/`, `smoke/`는 후속 Stage에서 채움). `conftest.py`는 무거운 객체(`Database`, `Builder`, `Locator`)를 session 스코프로 단 한 번 초기화. Stage 1에서는 smoke 레벨의 import 테스트 한 개만 추가해 인프라가 살아 있음을 증명한다. CI는 `uv sync` + `uv run pytest -n auto`로 일원화.

**Tech Stack:** Python 3.10+, uv, pytest 8+, pytest-xdist, GitHub Actions (ubuntu-22.04, macos-14).

**Spec:** `docs/superpowers/specs/2026-05-28-testing-foundation-design.md`

**Branch:** `docs/testing-foundation` (이미 origin/main 위에서 spec 커밋 1개 보유). 이 plan의 모든 커밋은 이 브랜치 위에 쌓고, PR로 origin/main에 머지.

---

## File Structure

| 경로 | 작업 | 책임 |
|------|------|------|
| `.gitignore` | Modify | `test*` 패턴 제거 — `tests/`가 ignored 되는 문제 수정 |
| `pyproject.toml` | Modify | `requires-python`을 `>=3.10`으로, classifiers·dev deps에 pytest 추가 |
| `tests/conftest.py` | Create | session-scoped `database`, `builder`, `locator` 픽스처 |
| `tests/smoke/test_import.py` | Create | `import pormake` 성공 + public symbol 접근 검증 |
| `.github/workflows/test.yaml` | Create | 새 CI: 매트릭스 + `uv sync` + `pytest -n auto` |
| `.github/workflows/pip-test.yaml` | Delete | 구 CI 폐기 (install-only) |
| `runtime.log` | Delete (git rm) | 13MB 잘못 커밋된 로그 파일 |
| `HKUST1.cif` | Delete (git rm) | 루트의 산출물 — 예시는 `example/` 안에서 생성 |

각 변경은 task 단위로 분리해 작은 커밋으로 쌓는다.

---

## Task 1: `.gitignore`의 `tests/` ignore 문제 수정

배경: 현재 `.gitignore`에 `test*` 패턴이 있어 `tests/` 디렉토리가 통째로 ignored 됨. 이걸 먼저 고치지 않으면 이후 task에서 `git add tests/`가 무시된다.

**Files:**
- Modify: `/Users/lsw91/Workspace/PORMAKE/.gitignore`

- [ ] **Step 1: 현재 `.gitignore`가 `tests/`를 ignore하는지 확인**

Run:
```bash
git check-ignore -v tests tests/conftest.py 2>&1
```
Expected: `.gitignore:6:test*	tests` 같이 6번째 줄(`test*`)이 매칭됨.

- [ ] **Step 2: `.gitignore`에서 `test*` 라인 제거**

기존 파일:
```
*__pycache__
*.cif
build
dist
*.egg-*
test*
*.log
*.pickle
*.log
.ipynb_checkpoints
```

수정 후 (`test*` 라인을 제거하고 중복 `*.log`도 1개로 정리):
```
*__pycache__
*.cif
build
dist
*.egg-*
*.log
*.pickle
.ipynb_checkpoints
```

`*.cif`는 그대로 유지 (산출물 무시 의도) — 이 때문에 `tests/data/*.cif` 같은 fixture를 둘 수 없으니, 그런 fixture가 필요해질 때는 별도 task에서 `!tests/data/` 예외를 추가한다 (Stage 1에서는 불필요).

- [ ] **Step 3: 변경이 적용됐는지 확인**

Run:
```bash
git check-ignore -v tests tests/conftest.py 2>&1 || echo "(not ignored)"
```
Expected: `(not ignored)`

- [ ] **Step 4: 커밋**

```bash
git add .gitignore
git commit -m "chore: stop ignoring tests/ directory

The legacy 'test*' pattern in .gitignore swept up the tests/
directory itself, blocking the new test suite. Remove it and
de-duplicate *.log while we're here."
```

---

## Task 2: 트래킹된 노이즈 파일 제거

**Files:**
- Delete (git rm): `runtime.log`, `HKUST1.cif`

`.gitignore`에 이미 `*.log`, `*.cif` 패턴이 있으므로 `git rm` 후에는 재추가되지 않는다.

- [ ] **Step 1: 트래킹된 노이즈 파일 확인**

Run:
```bash
git ls-files | grep -E '^(runtime\.log|HKUST1\.cif)$'
```
Expected:
```
HKUST1.cif
runtime.log
```

- [ ] **Step 2: `git rm`으로 트래킹 해제 (작업 트리에서도 제거)**

Run:
```bash
git rm runtime.log HKUST1.cif
```
Expected:
```
rm 'HKUST1.cif'
rm 'runtime.log'
```

- [ ] **Step 3: 다시 트래킹되지 않음을 확인**

Run:
```bash
git status --short
```
Expected: `D runtime.log` / `D HKUST1.cif` (staged deletion) 외에는 본 task와 무관한 항목만.

- [ ] **Step 4: 커밋**

```bash
git commit -m "chore: remove tracked noise files

runtime.log (13MB) and the root HKUST1.cif were accidentally
committed. Both file types are already in .gitignore; they only
remained tracked from earlier commits."
```

---

## Task 3: `pyproject.toml` — Python 버전 상향 + pytest 의존성

**Files:**
- Modify: `/Users/lsw91/Workspace/PORMAKE/pyproject.toml`

- [ ] **Step 1: 현재 `pyproject.toml` 확인**

Run:
```bash
cat /Users/lsw91/Workspace/PORMAKE/pyproject.toml
```

`requires-python = ">=3.8"`, classifiers에 3.8/3.9 포함, `[dependency-groups].dev`에 notebook만 있는 상태.

- [ ] **Step 2: `requires-python` 상향**

`requires-python = ">=3.8"` → `requires-python = ">=3.10"`

- [ ] **Step 3: classifiers에서 3.8/3.9 제거하고 3.12 추가**

Before:
```toml
classifiers = [
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11"
]
```

After:
```toml
classifiers = [
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
```

- [ ] **Step 4: `[dependency-groups].dev`에 pytest와 pytest-xdist 추가**

Before:
```toml
[dependency-groups]
dev = [
    "notebook>=7.3.2",
]
```

After:
```toml
[dependency-groups]
dev = [
    "notebook>=7.3.2",
    "pytest>=8.0",
    "pytest-xdist>=3.5",
]
```

- [ ] **Step 5: `uv sync`로 lock 갱신 및 설치**

Run:
```bash
uv sync --dev
```
Expected: pytest, pytest-xdist 설치됨. 에러 없이 완료.

- [ ] **Step 6: pytest 실행 가능 확인 (테스트 0개라 통과)**

Run:
```bash
uv run pytest --version
```
Expected: `pytest 8.x.x` 출력.

- [ ] **Step 7: 커밋**

```bash
git add pyproject.toml uv.lock
git commit -m "build: drop python 3.8/3.9, add pytest dev deps

- requires-python: >=3.8 -> >=3.10 (jax 0.4.13 and pymatgen
  latest both dropped <3.10 support)
- classifiers updated to 3.10/3.11/3.12
- dev group adds pytest>=8 and pytest-xdist>=3.5"
```

---

## Task 4: `tests/conftest.py` + smoke import 테스트

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/conftest.py`
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/smoke/test_import.py`

- [ ] **Step 1: `tests/conftest.py` 작성**

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

이 fixture는 Stage 1에서는 사용되지 않으나, 후속 Stage가 즉시 활용할 수 있도록 미리 둔다. import 자체가 라이브러리 헬스 체크 역할도 한다.

- [ ] **Step 2: `tests/smoke/test_import.py` 작성**

```python
import pormake


def test_top_level_import_succeeds():
    assert pormake is not None


def test_public_symbols_available():
    expected = {
        "Builder",
        "BuildingBlock",
        "Database",
        "Locator",
        "Scaler",
        "Topology",
    }
    missing = expected - set(dir(pormake))
    assert not missing, f"missing public symbols: {missing}"


def test_database_instantiates():
    db = pormake.Database()
    assert db is not None


def test_builder_instantiates():
    builder = pormake.Builder()
    assert builder is not None
```

세 번째·네 번째 케이스는 lazy import / JAX 초기화 문제를 잡기 위한 가벼운 인스턴스화 검증이다. 빌드는 안 하고 생성만 한다.

- [ ] **Step 3: 로컬에서 pytest 실행하여 통과 확인**

Run:
```bash
uv run pytest tests/ -v
```
Expected: 4 passed.

만약 import가 실패하면 Stage 1 진행을 멈추고 원인 분석 — 이게 첫 번째 회귀 안전망 신호다.

- [ ] **Step 4: 커밋**

```bash
git add tests/
git commit -m "test: add pytest infra with smoke import tests

- tests/conftest.py: session-scoped fixtures for Database,
  Builder, Locator (used by later stages)
- tests/smoke/test_import.py: verify top-level import, public
  API surface, and Database/Builder construction succeed

First runnable test suite for the repository."
```

---

## Task 5: CI 워크플로우 교체

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/.github/workflows/test.yaml`
- Delete: `/Users/lsw91/Workspace/PORMAKE/.github/workflows/pip-test.yaml`

- [ ] **Step 1: 새 워크플로우 작성**

`/Users/lsw91/Workspace/PORMAKE/.github/workflows/test.yaml`:

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
    env:
      UV_PYTHON: ${{ matrix.python-version }}
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Install Python ${{ matrix.python-version }} via uv
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --dev

      - name: Run tests
        run: uv run pytest -n auto -v
```

설계 메모:
- `pyproject.toml`에 `python-preference = "only-managed"` 가 있어 시스템 python을 안 쓰므로 `actions/setup-python`은 일부러 빼고 `uv python install`로 일원화한다.
- `UV_PYTHON` 환경변수로 매트릭스 버전을 uv에게 전달 → `uv sync`/`uv run` 모두 해당 버전을 사용.
- `uv sync --dev`는 dev group(pytest, pytest-xdist)도 함께 설치한다. `--all-extras`는 streamlit·stmol 등 무거운 extra까지 끌어와 CI 시간이 늘어나므로 Stage 1에서는 dev만으로 충분.

- [ ] **Step 2: 구 워크플로우 삭제**

Run:
```bash
git rm .github/workflows/pip-test.yaml
```

- [ ] **Step 3: 워크플로우 YAML이 문법적으로 유효한지 확인**

Run:
```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yaml'))"
```
Expected: 에러 없이 종료 (출력 없음).

- [ ] **Step 4: 커밋**

```bash
git add .github/workflows/test.yaml
git commit -m "ci: replace install-only workflow with pytest matrix

- Python 3.10/3.11/3.12 x ubuntu-22.04/macos-14
- uv sync --dev + uv run pytest -n auto
- Triggers updated: main/dev branches (drops obsolete 'test')
- macOS added to verify jax CPU build on the maintainer's
  primary environment

The old pip-test.yaml only verified that pip install succeeded;
the new workflow exercises actual code paths via smoke tests."
```

---

## Task 6: PR 생성 및 매트릭스 녹색 확인

**Files:** (no code changes — 이 task는 운영)

- [ ] **Step 1: 브랜치 푸시**

Run:
```bash
git push -u origin docs/testing-foundation
```
Expected: 브랜치가 origin에 생성됨.

- [ ] **Step 2: PR 생성**

Run:
```bash
gh pr create --base main --title "Add testing foundation (Stage 1)" --body "$(cat <<'EOF'
## Summary

- 테스트 인프라 도입 spec(`docs/superpowers/specs/2026-05-28-testing-foundation-design.md`)의 Stage 1.
- `tests/conftest.py` + smoke import 테스트 1개로 pytest가 동작함을 증명.
- CI를 install-only → 매트릭스 + `pytest -n auto`로 교체.
- Python 3.8/3.9 지원 종료, 저장소 노이즈(`runtime.log`, `HKUST1.cif`) 제거.

자세한 설계와 단계는 `docs/superpowers/specs/2026-05-28-testing-foundation-design.md`와 `docs/superpowers/plans/2026-05-28-testing-foundation-stage1.md` 참고.

## Test plan

- [ ] CI 매트릭스 6칸(3 Python x 2 OS) 모두 녹색
- [ ] 로컬 `uv run pytest -n auto -v` 4 passed
- [ ] `git check-ignore tests/conftest.py` 미매치
EOF
)"
```
Expected: PR URL 출력.

- [ ] **Step 3: CI 매트릭스 모니터링**

Run (PR 번호 N을 실제 값으로):
```bash
gh pr checks <N> --watch
```
Expected: 6/6 success. 실패 시 로그 확인 후 fix-up 커밋을 같은 브랜치에 추가.

흔한 실패 시나리오:
- macOS jax 설치 실패 → `runs-on: macos-14`를 `macos-13`로 fallback 후 재시도.
- `uv sync` 캐시 미스로 느림 → setup-uv의 `enable-cache: true` 옵션 추가 검토 (Stage 1 비범위, 그러나 5분 이상 걸리면 추가).
- pytest discovery 실패 (`tests/`가 여전히 ignored) → Task 1이 제대로 머지됐는지 확인.

- [ ] **Step 4: 머지는 사용자 검토 후 수동**

자동 머지는 하지 않는다. 매트릭스 녹색을 확인했다고 보고하고 사용자의 머지 결정을 기다린다.

---

## Stage 1 완료 기준

- Stage 1 PR이 origin/main에 머지됨
- 머지 후 main에서 `uv run pytest -n auto -v` 실행 시 4 passed
- 새 PR을 main에 올리면 자동으로 매트릭스 6칸이 돌고 녹색이 나옴

Stage 1이 머지된 이후 Stage 2 (smoke samples), Stage 3 (unit), Stage 4 (integration) plan을 별도로 작성·실행한다. 세 Stage는 서로 독립이라 병렬 진행 가능.
