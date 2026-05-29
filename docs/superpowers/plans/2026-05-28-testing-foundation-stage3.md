# Testing Foundation — Stage 3 (Unit) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 협력자가 적고 복잡도 있는 핵심 로직(xyz 파서, cgd→Topology 빌드, RMSD 계산, Database 조회)에 단위 테스트를 추가해 리팩토링 안전망을 만든다.

**Architecture:** `tests/unit/`에 4개 테스트 파일. 각 파일은 한 모듈의 public 동작을 검증한다. 기존 동작을 고정하는 characterization 성격이라 테스트는 처음부터 통과한다(회귀 베이스라인). 무거운 객체는 Stage 1의 session-scoped `database` fixture를 재사용한다. `Topology`는 pickle 부작용을 피하기 위해 `database.get_topo` 대신 cgd 경로로 직접 생성한다.

**Tech Stack:** pytest 9.x, ase, pymatgen, scipy (locator), numpy

**Spec:** `docs/superpowers/specs/2026-05-28-testing-foundation-design.md`

**Prerequisite:** Stage 1(#38), Stage 2(#46) 머지 완료. 이 plan은 그 위 main에서 분기한 `feat/testing-stage3`에서 실행한다 (이미 생성됨).

---

## Spec deviation: `test_framework.py`는 Stage 4로 이동

spec §5는 Stage 3에 `test_framework.py`(write_cif → ase.io.read 왕복)를 두었다. 그러나 소스 확인 결과 `Framework.__init__(atoms, bonds, bond_types, info)`는 `builder.build()`가 만든 `info`(topology·located_bbs·relax_obj·max_rmsd·mean_rmsd 포함)에 의존하고, `write_cif`는 `info["topology"]`를 역참조한다. 즉 `Framework`는 빌드 결과물이라 단위 테스트로 격리하기 어렵다. CIF 왕복 검증은 Stage 4의 `test_cif_roundtrip.py`에서 실제 빌드 결과로 수행한다. Stage 3는 순수 단위 대상 4개 파일에 집중한다.

---

## File Structure

| 경로 | 작업 | 책임 |
|------|------|------|
| `tests/unit/test_building_block.py` | Create | BB 로드/속성, copy() 독립성, xyz 파서(bond block 유무) |
| `tests/unit/test_topology.py` | Create | cgd→Topology 메타데이터, 인덱스/타입 일관성 |
| `tests/unit/test_locator.py` | Create | calculate_rmsd 값(README 시나리오)과 정성적 순서 |
| `tests/unit/test_database.py` | Create | get_bb/get_topo, 없는 키 에러, 리스트 조회 |

각 파일은 task 단위로 분리해 별도 커밋으로 쌓는다.

---

## 채취된 베이스라인 값 (현재 0.2.2 코드, main `285b210` 기준)

테스트에 하드코드할 값. 이미 실제 실행으로 확인됨:

**Building blocks:**
| BB | n_atoms | n_connection_points | is_edge | has bond block |
|----|---------|---------------------|---------|----------------|
| N10 | 12 | 3 | False | yes |
| N409 | 18 | 4 | False | yes |
| E41 | 8 | 2 | True | yes |
| N198 | 28 | 6 | False | yes |

**Topologies:**
| topo | n_slots | n_nodes | n_edges | n_node_types | n_edge_types | spacegroup |
|------|---------|---------|---------|--------------|--------------|------------|
| tbo | 152 | 56 | 96 | 2 | 1 | Fm-3m |
| pcu | 4 | 1 | 3 | 1 | 1 | Pm-3m |
| acs | 8 | 2 | 6 | 1 | 1 | P63/mmc |
| ith | 32 | 8 | 24 | 2 | 1 | Pm-3n |

**RMSD (calculate_rmsd 기본 max_n_slices=6):**
- acs `unique_local_structures[0]` vs N198 → 0.0206 (README: ~0.02)
- pcu `unique_local_structures[0]` vs N198 → 0.4213 (README: ~0.42)

**Custom xyz 파서 동작 (확인됨):**
- bond block 있는 3-connection-point + 1 C (4 atoms): n_cp=3, is_edge=False, bonds=3개, bond_types=['S','S','S']
- 동일 좌표에서 bond block 없는 경우: 자동 계산으로 bonds=3개, pairs=`[[0,3],[1,3],[2,3]]`

---

## Task 1: test_building_block.py

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/unit/test_building_block.py`

- [ ] **Step 1: 브랜치 확인**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
git branch --show-current
```
Expected: `feat/testing-stage3`. 아니면 BLOCKED.

- [ ] **Step 2: 테스트 파일 작성**

```python
import numpy as np
import pytest

import pormake


def test_node_bb_basic_properties(database):
    bb = database.get_bb("N10")
    assert isinstance(bb, pormake.BuildingBlock)
    assert bb.n_atoms == 12
    assert bb.n_connection_points == 3
    assert bb.is_node is True
    assert bb.is_edge is False


def test_edge_bb_is_edge(database):
    bb = database.get_bb("E41")
    assert bb.n_atoms == 8
    assert bb.n_connection_points == 2
    assert bb.is_edge is True
    assert bb.is_node is False


def test_connection_point_indices_match_count(database):
    bb = database.get_bb("N409")
    assert bb.n_connection_points == 4
    assert len(bb.connection_point_indices) == 4
    # connection_points are the positions at those indices
    assert bb.connection_points.shape == (4, 3)


def test_copy_is_independent(database):
    bb = database.get_bb("N10")
    clone = bb.copy()
    # Same data
    assert clone.n_atoms == bb.n_atoms
    assert clone.n_connection_points == bb.n_connection_points
    np.testing.assert_allclose(
        clone.atoms.get_positions(), bb.atoms.get_positions()
    )
    # Mutating the clone must not touch the original (deepcopy)
    clone.atoms.set_positions(clone.atoms.get_positions() + 1.0)
    assert not np.allclose(
        clone.atoms.get_positions(), bb.atoms.get_positions()
    )


def test_xyz_parser_reads_bond_block(tmp_path):
    xyz = tmp_path / "cn_with_bonds.xyz"
    xyz.write_text(
        "4\n"
        "   cn_with_bonds\n"
        "X    1.0 0.0 0.0\n"
        "X    -0.5 0.866 0.0\n"
        "X    -0.5 -0.866 0.0\n"
        "C    0.0 0.0 0.0\n"
        "   0    3 S\n"
        "   1    3 S\n"
        "   2    3 S\n"
    )
    bb = pormake.BuildingBlock(str(xyz))
    assert bb.name == "cn_with_bonds"
    assert bb.n_atoms == 4
    assert bb.n_connection_points == 3
    assert bb.is_node is True
    assert len(bb.bonds) == 3
    assert list(bb.bond_types) == ["S", "S", "S"]


def test_xyz_parser_auto_generates_bonds_when_absent(tmp_path):
    xyz = tmp_path / "cn_no_bonds.xyz"
    xyz.write_text(
        "4\n"
        "   cn_no_bonds\n"
        "X    1.0 0.0 0.0\n"
        "X    -0.5 0.866 0.0\n"
        "X    -0.5 -0.866 0.0\n"
        "C    0.0 0.0 0.0\n"
    )
    bb = pormake.BuildingBlock(str(xyz))
    assert bb.n_connection_points == 3
    # Bonds are auto-generated by distance threshold (3 X-C bonds)
    pairs = {tuple(sorted(pair)) for pair in np.array(bb.bonds).tolist()}
    assert pairs == {(0, 3), (1, 3), (2, 3)}
```

설계 메모:
- `database` fixture는 Stage 1 conftest의 session-scoped
- `tmp_path`는 pytest 내장 fixture — custom xyz를 격리된 임시 경로에 작성
- copy() 테스트는 deepcopy 독립성을 행동으로 검증 (구현 세부 아닌 관찰 가능한 효과)
- 파서 테스트는 README §6 사양(X = connection point, bond block optional)을 고정

- [ ] **Step 3: 실행**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
uv run pytest tests/unit/test_building_block.py -v
```
Expected: `6 passed`.

만약 실패하면, 베이스라인 값이 코드와 어긋난 것이므로 BLOCKED로 보고 (값 임의 수정 금지 — 원인 분석 우선).

- [ ] **Step 4: 전체 회귀**

Run:
```bash
uv run pytest -v
```
Expected: `90 passed` (Stage 1+2 smoke 84 + 새 6).

- [ ] **Step 5: 커밋**

```bash
git add tests/unit/test_building_block.py
git commit -m "test: add building block unit tests

Cover BuildingBlock load properties (N10/E41/N409), node vs edge
classification, deepcopy independence, and the xyz parser's two
paths: explicit bond block vs distance-based auto-generation.

Baseline values captured against the 0.2.2 database."
```

---

## Task 2: test_topology.py

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/unit/test_topology.py`

- [ ] **Step 1: 테스트 파일 작성**

```python
import pytest

import pormake


# (name, n_slots, n_nodes, n_edges, n_node_types, n_edge_types, spacegroup)
TOPOLOGY_FACTS = [
    ("tbo", 152, 56, 96, 2, 1, "Fm-3m"),
    ("pcu", 4, 1, 3, 1, 1, "Pm-3m"),
    ("acs", 8, 2, 6, 1, 1, "P63/mmc"),
    ("ith", 32, 8, 24, 2, 1, "Pm-3n"),
]


def make_topology(database, name):
    # Build directly from the cgd path to avoid the .pickle side effect
    # that database.get_topo writes into the package directory.
    return pormake.Topology(database.topo_dir / f"{name}.cgd")


@pytest.mark.parametrize(
    "name, n_slots, n_nodes, n_edges, n_node_types, n_edge_types, sg",
    TOPOLOGY_FACTS,
)
def test_topology_metadata(
    database, name, n_slots, n_nodes, n_edges, n_node_types, n_edge_types, sg
):
    topo = make_topology(database, name)
    assert topo.n_slots == n_slots
    assert topo.n_nodes == n_nodes
    assert topo.n_edges == n_edges
    assert topo.n_node_types == n_node_types
    assert topo.n_edge_types == n_edge_types
    assert topo.spacegroup == sg


@pytest.mark.parametrize("name", [f[0] for f in TOPOLOGY_FACTS])
def test_topology_index_consistency(database, name):
    topo = make_topology(database, name)
    # nodes + edges partition all slots
    assert topo.n_nodes + topo.n_edges == topo.n_slots
    assert len(topo.node_indices) == topo.n_nodes
    assert len(topo.edge_indices) == topo.n_edges
    # unique type counts match the reported counts
    assert len(topo.unique_node_types) == topo.n_node_types
    assert len(topo.unique_edge_types) == topo.n_edge_types
    # one local structure per unique node type
    assert len(topo.unique_local_structures) == topo.n_node_types
```

설계 메모:
- `make_topology`는 `Topology(cgd_path)`로 직접 생성해 `database.get_topo`의 pickle 쓰기 부작용을 회피. `database.topo_dir`는 session fixture에서 접근 (database.py가 노출)
- 메타데이터 값은 모두 README와 일치하며 실제 실행으로 채취됨
- 일관성 테스트는 "노드+엣지=전체 슬롯", "고유 타입 수=보고된 타입 수" 같은 구조 불변량 — 구현이 바뀌어도 동작이 같으면 통과 (리팩토링 내성)

- [ ] **Step 2: 실행**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
uv run pytest tests/unit/test_topology.py -v
```
Expected: `8 passed` (4 metadata + 4 consistency).

실패 시 BLOCKED로 보고 (값 임의 수정 금지).

- [ ] **Step 3: 전체 회귀**

Run:
```bash
uv run pytest -v
```
Expected: `98 passed` (직전 90 + 새 8).

- [ ] **Step 4: 커밋**

```bash
git add tests/unit/test_topology.py
git commit -m "test: add topology unit tests

Pin metadata (slots, nodes, edges, type counts, spacegroup) for
tbo/pcu/acs/ith and assert structural invariants: node+edge slot
partition, unique-type counts, one local structure per node type.

Topology is built directly from the cgd path to avoid the pickle
side effect of database.get_topo. Baseline values match README."
```

---

## Task 3: test_locator.py

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/unit/test_locator.py`

- [ ] **Step 1: 테스트 파일 작성**

```python
import pytest

import pormake


def make_topology(database, name):
    return pormake.Topology(database.topo_dir / f"{name}.cgd")


def test_rmsd_low_for_matching_node(database, locator):
    # acs node type 0 is a triangular prism; N198 is a triangular-prism
    # metal cluster, so the RMSD should be near zero.
    acs = make_topology(database, "acs")
    n198 = database.get_bb("N198")
    rmsd = locator.calculate_rmsd(acs.unique_local_structures[0], n198)
    assert rmsd == pytest.approx(0.02, abs=0.05)


def test_rmsd_high_for_mismatching_node(database, locator):
    # pcu node type 0 is an octahedron; N198 (triangular prism) fits poorly.
    pcu = make_topology(database, "pcu")
    n198 = database.get_bb("N198")
    rmsd = locator.calculate_rmsd(pcu.unique_local_structures[0], n198)
    assert rmsd == pytest.approx(0.42, rel=0.1)


def test_matching_node_has_lower_rmsd_than_mismatching(database, locator):
    n198 = database.get_bb("N198")
    acs = make_topology(database, "acs")
    pcu = make_topology(database, "pcu")
    rmsd_acs = locator.calculate_rmsd(acs.unique_local_structures[0], n198)
    rmsd_pcu = locator.calculate_rmsd(pcu.unique_local_structures[0], n198)
    assert rmsd_acs < rmsd_pcu
```

설계 메모:
- `locator` fixture는 Stage 1 conftest의 session-scoped
- 허용 오차: spec §3 따름 — 대칭 매칭(acs)은 `abs=0.05`, 비대칭(pcu)은 `rel=0.1` (정렬 결과가 더 흔들리므로 약간 더 느슨)
- 정성적 순서 테스트(`acs < pcu`)는 절대값과 무관하게 항상 성립해야 하는 핵심 불변량 — jax/scipy 버전 차이에 가장 강건

- [ ] **Step 2: 실행**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
uv run pytest tests/unit/test_locator.py -v
```
Expected: `3 passed`.

실패 시: RMSD 값이 README/베이스라인과 크게 다르면 BLOCKED로 보고하고 실제 측정값을 함께 보고. 정성적 순서(`acs < pcu`)만 깨지면 이는 심각한 회귀이므로 반드시 BLOCKED.

- [ ] **Step 3: 전체 회귀**

Run:
```bash
uv run pytest -v
```
Expected: `101 passed` (직전 98 + 새 3).

- [ ] **Step 4: 커밋**

```bash
git add tests/unit/test_locator.py
git commit -m "test: add locator RMSD unit tests

Verify calculate_rmsd against README scenarios: low RMSD (~0.02)
for the matching acs triangular-prism node with N198, high RMSD
(~0.42) for the mismatching pcu octahedron, and the qualitative
invariant that the matching node always scores lower.

Tolerances per spec section 3 (abs=0.05 symmetric, rel=0.1 else)."
```

---

## Task 4: test_database.py

**Files:**
- Create: `/Users/lsw91/Workspace/PORMAKE/tests/unit/test_database.py`

- [ ] **Step 1: 테스트 파일 작성**

```python
import pytest

import pormake


def test_get_bb_returns_building_block(database):
    bb = database.get_bb("N10")
    assert isinstance(bb, pormake.BuildingBlock)
    assert bb.name == "N10"


def test_get_topo_returns_topology(database):
    topo = database.get_topo("pcu")
    assert isinstance(topo, pormake.Topology)
    assert topo.name == "pcu"


def test_get_bb_missing_name_raises(database):
    with pytest.raises(Exception):
        database.get_bb("definitely_not_a_real_bb_name_zzz")


def test_get_topo_missing_name_raises(database):
    with pytest.raises(Exception):
        database.get_topo("definitely_not_a_real_topo_zzz")


def test_bb_list_nonempty_and_contains_known(database):
    names = database.bb_list
    assert len(names) > 0
    assert "N10" in names
    assert "E41" in names


def test_topo_list_nonempty_and_contains_known(database):
    names = database.topo_list
    assert len(names) > 0
    assert "tbo" in names
    assert "pcu" in names
```

설계 메모:
- `get_bb`/`get_topo`의 정상 경로 + 에러 경로(없는 이름 → Exception) 양쪽 검증. database.py는 로드 실패 시 일반 `Exception`을 raise하므로 `pytest.raises(Exception)`으로 받음
- `bb_list`/`topo_list`는 디렉토리 스캔 결과 — 비어있지 않고 알려진 이름을 포함하는지 확인
- `test_get_topo_returns_topology`는 `get_topo`를 호출하므로 pcu.pickle이 패키지 디렉토리에 생성되는 부작용이 있음. 이는 `get_topo`의 의도된 동작(캐시)을 검증하는 것이므로 허용. `.gitignore`의 `*.pickle`로 git에는 잡히지 않음

- [ ] **Step 2: 실행**

Run:
```bash
cd /Users/lsw91/Workspace/PORMAKE
uv run pytest tests/unit/test_database.py -v
```
Expected: `6 passed`.

- [ ] **Step 3: 전체 회귀**

Run:
```bash
uv run pytest -v
```
Expected: `107 passed` (직전 101 + 새 6).

- [ ] **Step 4: 커밋**

```bash
git add tests/unit/test_database.py
git commit -m "test: add database unit tests

Cover get_bb/get_topo happy paths, missing-name error paths
(both raise Exception), and bb_list/topo_list directory scans
returning non-empty results containing known names."
```

---

## Task 5: PR 생성 및 매트릭스 확인

(코드 변경 없는 운영 task)

- [ ] **Step 1: 브랜치 푸시**

```bash
git push -u origin feat/testing-stage3
```

- [ ] **Step 2: PR 생성**

```bash
gh pr create --base main --head feat/testing-stage3 --title "Add unit tests for core modules (Stage 3)" --body "$(cat <<'EOF'
## Summary

테스트 인프라 도입 spec의 Stage 3 (unit tests).

- `test_building_block.py`: 로드/속성, node vs edge, deepcopy 독립성, xyz 파서(bond block 유무) — 6 tests
- `test_topology.py`: tbo/pcu/acs/ith 메타데이터 + 구조 불변량 — 8 tests
- `test_locator.py`: calculate_rmsd README 시나리오 + 정성적 순서 — 3 tests
- `test_database.py`: get_bb/get_topo, 없는 키 에러, 리스트 조회 — 6 tests

총 23개 단위 테스트 추가. 전체 107 passed.

### Spec deviation

`test_framework.py`(CIF 왕복)는 Framework가 빌드 결과물(`builder.build()` 의존)이라 단위 테스트로 부적합 → Stage 4 `test_cif_roundtrip.py`로 이동.

## Plan / Spec

- `docs/superpowers/plans/2026-05-28-testing-foundation-stage3.md`
- `docs/superpowers/specs/2026-05-28-testing-foundation-design.md`

## Test plan

- [ ] CI 매트릭스 6칸 전부 녹색
- [ ] 로컬 `uv run pytest -v` → 107 passed
EOF
)"
```

- [ ] **Step 3: 매트릭스 모니터링**

```bash
gh pr checks <N> --watch --interval 30
```
Expected: 6/6 success.

- [ ] **Step 4: 머지는 사용자 검토 후 수동**

매트릭스 녹색 확인 후 보고하고 사용자의 머지 결정을 기다린다.

---

## Stage 3 완료 기준

- PR이 origin/main에 머지됨
- 머지 후 main에서 `uv run pytest -v` → 107 passed
- 전체 단위 테스트 <5초 (spec §1 목표)

---

## 명시적 비범위

- `test_framework.py` / CIF 왕복 → Stage 4
- `test_scaler.py` → spec §5 Stage 3 task 목록에 없음. Scaler는 빌드 파이프라인 내부 협력자라 Stage 4 통합에서 간접 검증. 단위가 꼭 필요하면 별도 follow-up
- Stage 2 final review가 권고한 CN 커버리지 보강(CN 7/8/9/10/24 BB 샘플) → 별도 follow-up PR (smoke 영역이라 Stage 3 unit과 결이 다름)
- `experimental/decomposer`·`app/` 테스트 → 별도 spec
