# Testing Foundation — Stage 4 (Integration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** README의 핵심 MOF 빌드 시나리오를 end-to-end로 검증해 빌드 파이프라인(locator→scaler→framework) 전체의 회귀를 잡는다.

**Architecture:** `tests/integration/`에 5개 시나리오 파일. 각 파일은 한 시나리오를 module-scoped fixture로 단 한 번 빌드하고 여러 test 함수가 그 결과를 공유한다(빌드가 비싸므로). 검증은 spec §3의 불변량 원칙을 따른다: 원자 수·원소 조성은 exact(빌드는 동일 BB를 배치하므로 결정적), cell 기하는 cross-platform jax 차이를 흡수하는 느슨한 sanity로만 검증한다.

**Tech Stack:** pytest 9.x, pytest-xdist, ase, jax(cpu), pymatgen, scipy

**Spec:** `docs/superpowers/specs/2026-05-28-testing-foundation-design.md`

**Prerequisite:** Stage 1/2/3 머지 완료. 이 plan은 그 위 main에서 분기한 `feat/testing-stage4`에서 실행한다 (이미 생성됨).

---

## 핵심 설계 결정: 무엇을 얼마나 엄격하게 검증하는가

빌드 결정성을 직접 측정했다 (현재 main `f2e0ac9`, macOS):
- 동일 플랫폼에서 HKUST-1을 2회 빌드 → 원자 수, cellpar **완전히 동일** (maxdiff 0.0). scaler/jax 최적화는 결정적이다.
- 따라서 같은 CI runner 안에서는 모든 값이 정확히 재현된다. **위험은 baseline을 채취한 플랫폼(macOS)과 다른 플랫폼(Linux) 사이의 float 차이뿐이다.**

검증 정책:
| 항목 | 엄격도 | 근거 |
|------|--------|------|
| 원자 개수 | exact (`==`) | 빌드가 위상의 모든 슬롯에 정해진 BB를 배치 → 플랫폼 무관하게 동일 |
| 원소 조성 (`Counter`) | exact (`==`) | 동일. 원소 구성은 기하 최적화와 무관 |
| cell 길이 a,b,c | 느슨 (`rel=5e-2`) | scaler 최적화 결과. 동일 플랫폼은 정확하나 cross-platform 마진 필요. 5%는 "셀이 완전히 깨짐"을 잡으면서 BLAS 차이를 흡수 |
| cell 각도 (α,β,γ) | 검증 안 함 | 90°에서 벗어난 최적화 잔차(예: 89.2°)가 cross-platform에서 가장 불안정 |
| 원자 좌표 (element-wise) | 검증 안 함 | spec §3: 원자 순서가 빌드에 따라 바뀔 수 있어 false positive 다발 |

cell 길이 sanity는 **대표 시나리오(HKUST-1)에만** 적용하고, 나머지 시나리오는 원자 수+조성만 검증한다 (cross-platform 위험 최소화 + 핵심 회귀는 충분히 포착).

---

## 채취된 baseline (현재 0.2.2, main `f2e0ac9`)

| 시나리오 | n_atoms | 조성 (Counter) | cell 길이 a,b,c (참고) |
|----------|---------|----------------|------------------------|
| HKUST-1 (tbo + N10@0 + N409@1) | 624 | C:288, H:96, Cu:48, O:192 | 27.139, 27.14, 27.14 |
| + edge E41 ((0,1)) | 1200 | C:864, H:96, Cu:48, O:192 | 71.875, 71.876, 71.876 |
| low-symmetry (ith + N3@0 + N114@1 + E41) | 294 | C:216, H:24, Ce:6, O:48 | — |
| chimera (tbo + N10/N409 + E41, 슬롯 33/38/40/49/53/55=N13) | 1320 | C:960, H:156, Cu:36, O:144, N:24 | — |
| CIF roundtrip (HKUST-1 write→ase.io.read) | 624 | C:288, H:96, Cu:48, O:192 | 27.139, 27.14, 27.14 |

조성 dict는 ASE `get_chemical_symbols()`의 `Counter` 기준. low-symmetry의 `Ce`는 N114(란타나이드 클러스터) 때문.

---

## File Structure

| 경로 | 작업 | 책임 |
|------|------|------|
| `tests/integration/test_build_hkust1.py` | Create | HKUST-1 빌드: 원자 수, 조성, cell 길이 sanity |
| `tests/integration/test_build_with_edge_bb.py` | Create | edge BB 삽입 빌드: 원자 수, 조성 |
| `tests/integration/test_build_low_symmetry.py` | Create | 저대칭 빌드(ith): 원자 수, 조성 |
| `tests/integration/test_build_chimera.py` | Create | chimera 빌드(슬롯별 다른 BB): 원자 수, 조성 |
| `tests/integration/test_cif_roundtrip.py` | Create | HKUST-1 write_cif → ase.io.read 왕복: 원자 수, 조성 보존 |

- helper 파일은 만들지 않는다. `Counter(atoms.get_chemical_symbols())` 비교는 한 줄이라 각 파일이 자립하는 편이 import 경로 문제(tests/integration이 sys.path에 없음)를 피해 더 단순하다.
- `tests/integration/__init__.py`는 만들지 않는다 (Stage 1~3의 rootdir-based discovery 컨벤션 유지).
- Stage 1 conftest의 session-scoped `database`/`builder` fixture를 재사용. conftest는 수정하지 않는다.
- 각 파일은 시나리오를 module-scoped fixture로 1회만 빌드한다 (빌드가 비싸므로). pytest-xdist 하에서 각 파일이 worker에 분산되고, 파일 내 여러 test가 한 번의 빌드를 공유한다.

---

## Task 1: test_build_hkust1.py

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/integration/test_build_hkust1.py`

- [ ] **Step 1: 브랜치 확인**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
git branch --show-current
```
Expected: `feat/testing-stage4`. 아니면 BLOCKED.

- [ ] **Step 2: 테스트 파일 작성**

```python
from collections import Counter

import pytest

import pormake


@pytest.fixture(scope="module")
def hkust1(database, builder):
    tbo = database.get_topo("tbo")
    node_bbs = {0: database.get_bb("N10"), 1: database.get_bb("N409")}
    return builder.build_by_type(topology=tbo, node_bbs=node_bbs)


def test_hkust1_atom_count(hkust1):
    assert len(hkust1.atoms) == 624


def test_hkust1_composition(hkust1):
    composition = Counter(hkust1.atoms.get_chemical_symbols())
    assert composition == Counter({"C": 288, "O": 192, "H": 96, "Cu": 48})


def test_hkust1_cell_lengths_are_sane(hkust1):
    a, b, c = hkust1.atoms.cell.cellpar()[:3]
    # Cubic HKUST-1; lengths captured at ~27.14 on the baseline platform.
    # Loose rel tolerance absorbs cross-platform jax/BLAS differences.
    assert a == pytest.approx(27.14, rel=5e-2)
    assert b == pytest.approx(27.14, rel=5e-2)
    assert c == pytest.approx(27.14, rel=5e-2)
```

설계 메모:
- `hkust1` fixture는 module-scoped — 이 파일의 3개 test가 단 한 번의 빌드를 공유
- `database`/`builder`는 Stage 1 conftest의 session fixture
- 조성은 exact `Counter` 비교 (결정적). cell 길이는 `rel=5e-2` sanity (cross-platform 마진). 각도·좌표는 비검증
- `database.get_topo("tbo")`는 tbo.pickle을 패키지 디렉토리에 씀 (gitignore됨) — README 시나리오 그대로이므로 허용

- [ ] **Step 3: 실행**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
uv run pytest tests/integration/test_build_hkust1.py -v
```
Expected: `3 passed`. 빌드 때문에 수 초 소요될 수 있음.

**실패 시:** 원자 수/조성이 baseline과 다르면 빌드 로직 회귀이므로 BLOCKED로 보고(값 임의 수정 금지). cell 길이만 `rel=5e-2` 밖이면 실제 측정값을 함께 DONE_WITH_CONCERNS로 보고.

- [ ] **Step 4: 전체 회귀**

Run:
```bash
uv run pytest -v
```
Expected: `110 passed` (Stage 1~3의 107 + 새 3).

- [ ] **Step 5: pickle이 staged되지 않았는지 확인 후 커밋**

```bash
git status --short
git add tests/integration/test_build_hkust1.py
git commit -m "test: add HKUST-1 build integration test

Build HKUST-1 (tbo + N10/N409) end-to-end and assert the
structural invariants: 624 atoms, exact composition
(C288 O192 H96 Cu48), and cubic cell lengths near 27.14
(loose rel=5e-2 to absorb cross-platform jax differences).

Baseline captured from live 0.2.2 build."
```

---

## Task 2: test_build_with_edge_bb.py

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/integration/test_build_with_edge_bb.py`

- [ ] **Step 1: 테스트 파일 작성**

```python
from collections import Counter

import pytest

import pormake


@pytest.fixture(scope="module")
def hkust1_with_edge(database, builder):
    tbo = database.get_topo("tbo")
    node_bbs = {0: database.get_bb("N10"), 1: database.get_bb("N409")}
    edge_bbs = {(0, 1): database.get_bb("E41")}
    return builder.build_by_type(
        topology=tbo, node_bbs=node_bbs, edge_bbs=edge_bbs
    )


def test_edge_build_atom_count(hkust1_with_edge):
    assert len(hkust1_with_edge.atoms) == 1200


def test_edge_build_composition(hkust1_with_edge):
    composition = Counter(hkust1_with_edge.atoms.get_chemical_symbols())
    assert composition == Counter({"C": 864, "O": 192, "H": 96, "Cu": 48})


def test_edge_build_adds_atoms_versus_no_edge(hkust1_with_edge):
    # Inserting E41 on every (0,1) edge must increase the atom count
    # relative to the node-only HKUST-1 (624 atoms, a README constant).
    assert len(hkust1_with_edge.atoms) > 624
```

설계 메모:
- `test_edge_build_adds_atoms_versus_no_edge`는 node-only HKUST-1의 알려진 상수(624)와 비교해 "edge 삽입이 원자를 늘린다"는 정성적 불변량을 검증. node-only를 다시 빌드하지 않고 module fixture 결과를 README 상수와 직접 비교 (재빌드 회피)
- 3개 test 모두 단일 module fixture 빌드를 공유

- [ ] **Step 2: 실행**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
uv run pytest tests/integration/test_build_with_edge_bb.py -v
```
Expected: `3 passed`.

실패 시 Task 1과 동일 정책.

- [ ] **Step 3: 전체 회귀**

Run:
```bash
uv run pytest -v
```
Expected: `113 passed` (직전 110 + 새 3).

- [ ] **Step 4: 커밋**

```bash
git add tests/integration/test_build_with_edge_bb.py
git commit -m "test: add edge-BB insertion integration test

Build HKUST-1 with E41 edge building blocks on every (0,1) edge
and assert 1200 atoms, exact composition (C864 O192 H96 Cu48),
and the qualitative invariant that edge insertion increases the
atom count beyond the node-only 624."
```

---

## Task 3: test_build_low_symmetry.py

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/integration/test_build_low_symmetry.py`

- [ ] **Step 1: 테스트 파일 작성**

```python
from collections import Counter

import pytest

import pormake


@pytest.fixture(scope="module")
def low_symmetry_mof(database, builder):
    ith = database.get_topo("ith")
    node_bbs = {0: database.get_bb("N3"), 1: database.get_bb("N114")}
    edge_bbs = {(0, 1): database.get_bb("E41")}
    return builder.build_by_type(
        topology=ith, node_bbs=node_bbs, edge_bbs=edge_bbs
    )


def test_low_symmetry_atom_count(low_symmetry_mof):
    assert len(low_symmetry_mof.atoms) == 294


def test_low_symmetry_composition(low_symmetry_mof):
    composition = Counter(low_symmetry_mof.atoms.get_chemical_symbols())
    assert composition == Counter({"C": 216, "O": 48, "H": 24, "Ce": 6})
```

설계 메모:
- ith 위상 + N114(저대칭 란타나이드 클러스터, Ce 포함) 시나리오. README 예제 5
- 조성에 Ce가 6개 — N114가 노드 타입 1(CN 12)에 배치됨을 확인

- [ ] **Step 2: 실행**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
uv run pytest tests/integration/test_build_low_symmetry.py -v
```
Expected: `2 passed`.

- [ ] **Step 3: 전체 회귀**

Run:
```bash
uv run pytest -v
```
Expected: `115 passed` (직전 113 + 새 2).

- [ ] **Step 4: 커밋**

```bash
git add tests/integration/test_build_low_symmetry.py
git commit -m "test: add low-symmetry MOF build integration test

Build an ith-topology MOF with the low-symmetry N114 lanthanide
node and assert 294 atoms and exact composition (C216 O48 H24
Ce6), confirming the Ce cluster lands on node type 1."
```

---

## Task 4: test_build_chimera.py

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/integration/test_build_chimera.py`

- [ ] **Step 1: 테스트 파일 작성**

```python
from collections import Counter

import pytest

import pormake


@pytest.fixture(scope="module")
def chimera_mof(database, builder):
    tbo = database.get_topo("tbo")
    node_bbs = {0: database.get_bb("N10"), 1: database.get_bb("N409")}
    edge_bbs = {(0, 1): database.get_bb("E41")}
    bbs = builder.make_bbs_by_type(
        topology=tbo, node_bbs=node_bbs, edge_bbs=edge_bbs
    )
    n13 = database.get_bb("N13")
    for idx in [33, 38, 40, 49, 53, 55]:
        bbs[idx] = n13.copy()
    return builder.build(topology=tbo, bbs=bbs)


def test_chimera_atom_count(chimera_mof):
    assert len(chimera_mof.atoms) == 1320


def test_chimera_composition(chimera_mof):
    composition = Counter(chimera_mof.atoms.get_chemical_symbols())
    assert composition == Counter(
        {"C": 960, "H": 156, "O": 144, "Cu": 36, "N": 24}
    )


def test_chimera_contains_nitrogen_from_porphyrin(chimera_mof):
    # N13 (porphyrin) introduces nitrogen, which pure HKUST-1 lacks.
    symbols = set(chimera_mof.atoms.get_chemical_symbols())
    assert "N" in symbols
```

설계 메모:
- README 예제 3 — `make_bbs_by_type`로 slot 리스트를 만든 뒤 일부 N409 슬롯(33/38/40/49/53/55)을 N13(porphyrin)으로 교체하고 `build`
- 질소(N) 존재는 porphyrin 치환의 직접 증거 — 순수 HKUST-1엔 N이 없음. 정성적 불변량
- 조성에서 Cu가 48→36으로 감소(6개 paddle-wheel이 porphyrin으로 대체), N 24개 추가

- [ ] **Step 2: 실행**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
uv run pytest tests/integration/test_build_chimera.py -v
```
Expected: `3 passed`.

- [ ] **Step 3: 전체 회귀**

Run:
```bash
uv run pytest -v
```
Expected: `118 passed` (직전 115 + 새 3).

- [ ] **Step 4: 커밋**

```bash
git add tests/integration/test_build_chimera.py
git commit -m "test: add chimera MOF build integration test

Build a chimera MOF by replacing six N409 paddle-wheel slots with
N13 porphyrin via make_bbs_by_type + build, and assert 1320 atoms,
exact composition (C960 H156 O144 Cu36 N24), and that nitrogen
from the porphyrin appears (absent in pure HKUST-1)."
```

---

## Task 5: test_cif_roundtrip.py

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/integration/test_cif_roundtrip.py`

목적: `Framework.write_cif`로 쓴 CIF를 `ase.io.read`로 다시 읽어 원자 수·조성이 보존되는지 객체 단위로 검증 (텍스트 비교 안 함 — spec §3).

- [ ] **Step 1: 테스트 파일 작성**

```python
from collections import Counter

import ase.io
import pytest

import pormake


@pytest.fixture(scope="module")
def hkust1(database, builder):
    tbo = database.get_topo("tbo")
    node_bbs = {0: database.get_bb("N10"), 1: database.get_bb("N409")}
    return builder.build_by_type(topology=tbo, node_bbs=node_bbs)


def test_cif_roundtrip_preserves_atom_count(hkust1, tmp_path):
    cif = tmp_path / "hkust1.cif"
    hkust1.write_cif(str(cif))
    reloaded = ase.io.read(str(cif))
    assert len(reloaded) == len(hkust1.atoms)
    assert len(reloaded) == 624


def test_cif_roundtrip_preserves_composition(hkust1, tmp_path):
    cif = tmp_path / "hkust1.cif"
    hkust1.write_cif(str(cif))
    reloaded = ase.io.read(str(cif))
    assert Counter(reloaded.get_chemical_symbols()) == Counter(
        hkust1.atoms.get_chemical_symbols()
    )


def test_cif_file_is_created(hkust1, tmp_path):
    cif = tmp_path / "hkust1.cif"
    hkust1.write_cif(str(cif))
    assert cif.exists()
    assert cif.stat().st_size > 0
```

설계 메모:
- `tmp_path`로 CIF를 격리된 임시 경로에 작성 — 패키지/저장소 오염 없음
- 텍스트 비교가 아닌 `ase.io.read` 후 객체 단위 비교 (부동소수점 출력 포맷 차이 회피, spec §3)
- 이 파일이 spec §5의 `test_framework.py`(Stage 3에서 이관) 역할을 수행 — Framework는 빌드 결과물이라 여기(integration)가 올바른 위치

- [ ] **Step 2: 실행**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
uv run pytest tests/integration/test_cif_roundtrip.py -v
```
Expected: `3 passed`.

- [ ] **Step 3: 전체 회귀**

Run:
```bash
uv run pytest -v
```
Expected: `121 passed` (직전 118 + 새 3).

- [ ] **Step 4: 커밋**

```bash
git add tests/integration/test_cif_roundtrip.py
git commit -m "test: add CIF write/read roundtrip integration test

Write a built HKUST-1 to CIF and reload it with ase.io.read,
asserting atom count and composition are preserved (object-level
comparison, not text diff per spec section 3). Covers the
Framework.write_cif path deferred from Stage 3."
```

---

## Task 6: PR 생성 및 매트릭스 확인

(코드 변경 없는 운영 task)

- [ ] **Step 1: 브랜치 푸시**

```bash
git push -u origin feat/testing-stage4
```

- [ ] **Step 2: PR 생성**

```bash
gh pr create --base main --head feat/testing-stage4 --title "Add MOF build integration tests (Stage 4)" --body "$(cat <<'EOF'
## Summary

테스트 인프라 도입 spec의 Stage 4 (integration). README 핵심 빌드 시나리오를 end-to-end 검증.

- `test_build_hkust1.py` (3): HKUST-1 — 624 atoms, 조성, cubic cell sanity
- `test_build_with_edge_bb.py` (3): E41 삽입 — 1200 atoms, 조성, 정성적 증가
- `test_build_low_symmetry.py` (2): ith + N114 — 294 atoms, Ce 포함 조성
- `test_build_chimera.py` (3): 슬롯별 N13 치환 — 1320 atoms, N 도입
- `test_cif_roundtrip.py` (3): write_cif → ase.io.read 왕복 (Stage 3 test_framework 이관)

총 14개 통합 테스트. 로컬 전체 121 passed.

## 검증 정책 (spec §3)

빌드는 동일 플랫폼에서 결정적(2회 빌드 cellpar 차이 0.0)임을 측정 확인. 따라서:
- 원자 수·원소 조성: exact (플랫폼 무관 결정적)
- cell 길이: HKUST-1만 rel=5e-2 sanity (cross-platform jax 차이 흡수)
- cell 각도·원자 좌표: 비검증 (cross-platform 불안정 / 원자 순서 가변)

## Plan / Spec

- \`docs/superpowers/plans/2026-05-28-testing-foundation-stage4.md\`
- \`docs/superpowers/specs/2026-05-28-testing-foundation-design.md\`

## Test plan

- [ ] CI 매트릭스 6칸 전부 녹색 (특히 Linux에서 cell sanity 통과 확인)
- [ ] 머지 후 main에서 \`uv run pytest -v\` → 121 passed
EOF
)"
```

- [ ] **Step 3: 매트릭스 모니터링**

```bash
gh pr checks <N> --watch --interval 30
```
Expected: 6/6 success. **특히 주목:** Linux runner에서 HKUST-1 cell 길이가 `rel=5e-2` 안에 드는지. 만약 Linux에서 cell sanity가 깨지면 baseline(macOS 채취)과 Linux jax 결과 차이가 5%를 초과한 것이므로, tolerance를 넓히기 전에 실제 Linux 측정값을 확인하고 보고.

- [ ] **Step 4: 머지는 사용자 검토 후 수동**

매트릭스 녹색 확인 후 보고하고 사용자의 머지 결정을 기다린다.

---

## Stage 4 완료 기준

- PR이 origin/main에 머지됨
- 머지 후 main에서 `uv run pytest -v` → 121 passed
- 통합 테스트 전체 <90초 (spec §1 목표)

---

## 명시적 비범위

- `test_scaler.py` 단위 테스트 → 별도 follow-up (scaler는 이 통합 테스트들에서 간접 검증됨)
- Stage 2 권고 CN 커버리지 보강 → 별도 follow-up
- `experimental/decomposer`·`app/` 테스트 → 별도 spec
- spec 문서의 `상태: Draft → User Review` 갱신 → 전 stage 머지 후 housekeeping PR
