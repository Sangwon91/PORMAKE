# Testing Foundation — Stage 2 (Smoke Samples) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PORMAKE의 BB(867개) / topology(2479개) 데이터 자산이 의존성 업그레이드에도 깨지지 않음을 빠르게 감지하는 parametrized smoke 테스트를 추가한다.

**Architecture:** `tests/smoke/`에 두 개의 parametrized 테스트 파일 추가. 각 파일은 대표 샘플 20개 이름을 명시적 리스트로 박아두고 `database.get_bb` / `database.get_topo`가 정상 동작함을 확인. Stage 1의 session-scoped `database` fixture를 즉시 활용.

**Tech Stack:** pytest 9.x, pytest-xdist 3.x (Stage 1에서 도입됨)

**Spec:** `docs/superpowers/specs/2026-05-28-testing-foundation-design.md`

**Prerequisite:** Stage 1 PR(#38)이 main에 머지되어 있어야 한다. 이 plan은 Stage 1이 머지된 main 위에서 분기한 새 브랜치 `feat/testing-stage2`에서 실행한다.

---

## File Structure

| 경로 | 작업 | 책임 |
|------|------|------|
| `tests/smoke/test_database_samples.py` | Create | 대표 BB 20개를 parametrize로 로드 검증 |
| `tests/smoke/test_topology_samples.py` | Create | 대표 topology 20개를 parametrize로 로드 + `unique_local_structures` 검증 |

각 파일은 task 단위로 분리해 별도 커밋으로 쌓는다. 두 파일은 책임이 다르므로 한 commit에 묶지 않는다.

---

## 대표 샘플 선정 기준

spec §5 Stage 2: "README에 등장하는 모든 이름 + 각 CN별 1개씩". 추가로 인덱스 범위 분포(N1 ~ N900대)도 포함해 데이터 디렉토리 스캔 회귀를 잡는다.

### BB 샘플 (20개)

README 등장 7개 + 인덱스 분포 9개 + Edge BB 4개:

```python
SAMPLE_BBS = [
    # README 등장 (Stage 1 plan/spec에 명시된 빌드 시나리오의 핵심)
    "N3",     # CN 4, ith 노드
    "N10",    # CN 3, BTC linker (HKUST-1)
    "N13",    # CN 4, porphyrin (chimera MOF)
    "N114",   # CN 12, ith 노드 (저대칭)
    "N198",   # CN 6, triangular prism
    "N409",   # CN 4, Cu paddle-wheel (HKUST-1)
    "E41",    # 긴 edge BB
    # 인덱스 분포 — 디렉토리 스캔 회귀 감지용
    "N1",
    "N100",
    "N200",
    "N300",
    "N400",
    "N500",
    "N600",
    "N700",
    "N800",
    # 추가 edge
    "E1",
    "E50",
    "E100",
    "E200",
]
```

확인된 사실: 위 17개의 N{n} 이름은 `src/pormake/database/bbs/N{n}.xyz` 형식으로 모두 존재. Edge BB 4개도 확인됨 (Stage 2 작업 전 다시 확인하라).

### Topology 샘플 (20개)

README 등장 4개 + RCSR 잘 알려진 nets 16개:

```python
SAMPLE_TOPOLOGIES = [
    # README 등장
    "tbo",    # HKUST-1
    "pcu",    # primitive cubic
    "acs",    # triangular prism
    "ith",    # 저대칭 빌드 시나리오
    # 잘 알려진 RCSR nets — 모두 존재 확인됨
    "pts",
    "dia",
    "nbo",
    "srs",
    "ths",
    "qom",
    # 확장 (Stage 2 작업 전 존재 재확인)
    "bcu",
    "fcu",
    "hcb",
    "soc",
    "rht",
    "sql",
    "lon",
    "hxg",
    "hms",
    "mtn",
]
```

**Implementer 주의:** 확장 10개 중 일부가 존재하지 않으면 즉시 BLOCKED로 보고하지 말고, 작업 디렉토리에서 `ls src/pormake/database/topologies/*.cgd | head -100`로 후보를 확인해 합리적인 대체를 선택한 뒤 DONE_WITH_CONCERNS로 보고하라. 핵심은 "20개의 다양한 topology"이지 "이 정확한 20개"가 아니다.

---

## Task 1: BB 샘플 smoke 테스트

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/smoke/test_database_samples.py`

목적: `database.get_bb(name)`이 명단의 모든 이름에 대해 정상 동작하고 결과가 `BuildingBlock` 인스턴스이며 최소한의 속성(`atoms`, connection points)을 가짐을 확인.

- [ ] **Step 1: 현재 작업 디렉토리/브랜치 확인**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
git branch --show-current
```
Expected: `feat/testing-stage2`. 다른 브랜치라면 BLOCKED로 보고.

- [ ] **Step 2: 샘플 이름이 실제 파일과 일치하는지 사전 검증**

Run:
```bash
for name in N3 N10 N13 N114 N198 N409 E41 N1 N100 N200 N300 N400 N500 N600 N700 N800 E1 E50 E100 E200; do
  test -f /Users/lsw91/Workspace/PORMAKE/src/pormake/database/bbs/${name}.xyz \
    && echo "OK $name" \
    || echo "MISSING $name"
done
```
Expected: 모두 `OK ...`. 누락이 있으면 BLOCKED로 보고.

- [ ] **Step 3: `tests/smoke/test_database_samples.py` 작성**

```python
import pytest

import pormake


SAMPLE_BBS = [
    "N3",
    "N10",
    "N13",
    "N114",
    "N198",
    "N409",
    "E41",
    "N1",
    "N100",
    "N200",
    "N300",
    "N400",
    "N500",
    "N600",
    "N700",
    "N800",
    "E1",
    "E50",
    "E100",
    "E200",
]


@pytest.mark.parametrize("name", SAMPLE_BBS)
def test_bb_loads(database, name):
    bb = database.get_bb(name)
    assert isinstance(bb, pormake.BuildingBlock)
    assert bb.atoms is not None
    assert len(bb.atoms) > 0


@pytest.mark.parametrize("name", SAMPLE_BBS)
def test_bb_has_connection_points(database, name):
    bb = database.get_bb(name)
    # 'X' atoms denote connection points (per README §6)
    n_x = sum(1 for sym in bb.atoms.get_chemical_symbols() if sym == "X")
    assert n_x > 0, f"BB {name} has no connection points (X atoms)"
```

설계 메모:
- `database` fixture는 `tests/conftest.py`의 session-scoped — DB 인스턴스화는 단 한 번
- `BuildingBlock` 인스턴스 검증으로 매개변수 매핑까지 확인 (Stage 1 `test_import.py`는 `pormake.BuildingBlock` 존재만 확인)
- connection point 검증으로 xyz 파서까지 가벼운 헬스 체크. spec §3의 "Exact 검증" 원칙 따름
- 좌표·연결성 등 더 깊은 검증은 Stage 3 unit 테스트에서

- [ ] **Step 4: 로컬 실행 확인**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
uv run pytest tests/smoke/test_database_samples.py -v
```
Expected: `40 passed` (20 names × 2 tests). 통과 시간 <5초.

- [ ] **Step 5: 전체 smoke 디렉토리 회귀 확인**

Run:
```bash
uv run pytest tests/smoke/ -v
```
Expected: `44 passed` (Stage 1의 4개 + 새 40개). 회귀 없음.

- [ ] **Step 6: 커밋**

```bash
git add tests/smoke/test_database_samples.py
git commit -m "test: add building block sample smoke tests

Parametrize 20 representative BB names (README scenarios + index
distribution + edge BBs) and verify each loads through
database.get_bb, yielding a BuildingBlock with atoms and at least
one connection point (X atom).

Catches data-asset corruption and xyz parser regressions on
dependency upgrades."
```

---

## Task 2: Topology 샘플 smoke 테스트

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/smoke/test_topology_samples.py`

목적: `database.get_topo(name)`이 명단의 모든 이름에 대해 정상 동작하고 결과가 `Topology` 인스턴스이며 `unique_local_structures`가 비어있지 않음을 확인. cgd 파서 + networkx 그래프 빌드 헬스 체크.

- [ ] **Step 1: 샘플 이름이 실제 파일과 일치하는지 사전 검증**

Run:
```bash
for name in tbo pcu acs ith pts dia nbo srs ths qom bcu fcu hcb soc rht sql lon hxg hms mtn; do
  test -f /Users/lsw91/Workspace/PORMAKE/src/pormake/database/topologies/${name}.cgd \
    && echo "OK $name" \
    || echo "MISSING $name"
done
```

만약 누락이 있으면, Plan에 명시된 대체 정책에 따라 처리:
- 누락된 이름을 `ls src/pormake/database/topologies/*.cgd | head -200`에서 다양한 대안으로 교체
- 최종 20개 리스트와 교체 사유를 self-review에 기록

- [ ] **Step 2: `tests/smoke/test_topology_samples.py` 작성**

```python
import pytest

import pormake


SAMPLE_TOPOLOGIES = [
    "tbo",
    "pcu",
    "acs",
    "ith",
    "pts",
    "dia",
    "nbo",
    "srs",
    "ths",
    "qom",
    "bcu",
    "fcu",
    "hcb",
    "soc",
    "rht",
    "sql",
    "lon",
    "hxg",
    "hms",
    "mtn",
]


@pytest.mark.parametrize("name", SAMPLE_TOPOLOGIES)
def test_topology_loads(database, name):
    topo = database.get_topo(name)
    assert isinstance(topo, pormake.Topology)
    assert topo.n_slots > 0


@pytest.mark.parametrize("name", SAMPLE_TOPOLOGIES)
def test_topology_has_unique_local_structures(database, name):
    topo = database.get_topo(name)
    locals_ = topo.unique_local_structures
    assert len(locals_) > 0, f"Topology {name} has no unique local structures"
```

설계 메모:
- `unique_local_structures` 접근은 cgd 파싱 → networkx 그래프 → 노드 타입 분류까지 전체 파이프라인을 한 번 돌림. 의존성(networkx, numpy) 회귀를 폭넓게 잡음
- 깊은 검증(슬롯 수·노드 타입 정확값)은 Stage 3 unit 테스트에서

- [ ] **Step 3: 만약 Step 1에서 일부 이름을 교체했다면, 위 리스트를 그에 맞게 수정**

리스트 변경 시:
- 최종 리스트의 20개를 모두 명시 (placeholder/TODO 금지)
- 교체 이유를 commit body에 명시 (예: "replaced 'mtn' with 'mof-32' since mtn.cgd was not present in 0.2.2 database")

- [ ] **Step 4: 로컬 실행 확인**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
uv run pytest tests/smoke/test_topology_samples.py -v
```
Expected: `40 passed` (20 names × 2 tests).

- [ ] **Step 5: 전체 smoke 디렉토리 회귀 확인**

Run:
```bash
uv run pytest tests/smoke/ -v
```
Expected: `84 passed` (Stage 1의 4개 + T1의 40개 + T2의 40개). 전체 실행 시간 <15초가 합리적인 상한.

- [ ] **Step 6: 커밋**

만약 리스트를 수정하지 않았다면:
```bash
git add tests/smoke/test_topology_samples.py
git commit -m "test: add topology sample smoke tests

Parametrize 20 representative topology names (README scenarios +
well-known RCSR nets) and verify each loads through
database.get_topo, yielding a Topology with positive slot count
and at least one unique local structure.

Catches cgd parser, networkx graph build, and node-type
classification regressions on dependency upgrades."
```

만약 리스트를 수정했다면, 커밋 메시지 body에 교체 내역을 추가:
```bash
git commit -m "test: add topology sample smoke tests

Parametrize 20 representative topology names (README scenarios +
well-known RCSR nets) and verify each loads through
database.get_topo, yielding a Topology with positive slot count
and at least one unique local structure.

Catches cgd parser, networkx graph build, and node-type
classification regressions on dependency upgrades.

Substitutions from initial plan list:
- <original name> -> <replacement>: <reason>"
```

---

## Task 3: PR 생성 및 매트릭스 녹색 확인

(코드 변경 없는 운영 task)

- [ ] **Step 1: 브랜치 푸시**

Run:
```bash
git push -u origin feat/testing-stage2
```

- [ ] **Step 2: PR 생성**

Run:
```bash
gh pr create --base main --head feat/testing-stage2 --title "Add smoke samples for BB and topology (Stage 2)" --body "$(cat <<'EOF'
## Summary

테스트 인프라 도입 spec의 Stage 2.

- `tests/smoke/test_database_samples.py`: 대표 BB 20개를 parametrize로 로드
- `tests/smoke/test_topology_samples.py`: 대표 topology 20개를 parametrize로 로드 + `unique_local_structures` 접근

의존성 한 줄 변경으로 깨질 데이터 자산을 즉시 감지하는 회귀 안전망.

자세한 설계: `docs/superpowers/specs/2026-05-28-testing-foundation-design.md`
구현 계획: `docs/superpowers/plans/2026-05-28-testing-foundation-stage2.md`

## Test plan

- [ ] CI 매트릭스 6칸(3 Python × 2 OS) 전부 녹색
- [ ] 로컬 `uv run pytest tests/smoke/ -v` → 84 passed
EOF
)"
```

- [ ] **Step 3: CI 매트릭스 모니터링**

Run (PR 번호 N을 실제 값으로):
```bash
gh pr checks <N> --watch --interval 30
```
Expected: 6/6 success.

흔한 실패 시나리오:
- 누락된 BB/topology 이름 → Task 1 또는 Task 2의 Step 1 사전 검증이 catch했어야 함. 로컬에서 다시 확인 후 fix-up
- xdist worker별 fixture 재초기화 → smoke 테스트는 idempotent하므로 무영향. 시간만 약간 추가

- [ ] **Step 4: 머지는 사용자 검토 후 수동**

매트릭스 녹색을 확인했다고 보고하고 사용자의 머지 결정을 기다린다.

---

## Stage 2 완료 기준

- PR이 origin/main에 머지됨
- 머지 후 main에서 `uv run pytest tests/smoke/ -v` → 84 passed
- 후속 Stage 3 (unit) 및 Stage 4 (integration)가 병렬로 진행 가능한 상태

---

## 명시적 비범위

- `.gitignore`의 `*.cif` 패턴은 Stage 3에서 `!tests/data/` 예외와 함께 처리 (Stage 1 final review 권고 사항)
- `tests/smoke/__init__.py` 추가하지 않음 — Stage 1의 컨벤션 유지 (pytest rootdir-based discovery)
- Conftest fixture를 사용하는 패턴 도입 (`database` fixture 활용) — Stage 1의 미사용 fixture를 처음으로 실제 활용. Stage 1 review minor #4 해결
- `test_top_level_import_succeeds` 등 Stage 1 tautological 테스트 정리는 별도 PR (Stage 2 scope 밖)
- `experimental/decomposer`·`app/` 테스트는 별도 spec
