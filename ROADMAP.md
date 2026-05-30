# PORMAKE Roadmap

This document tracks planned features and longer-term research directions for
PORMAKE. Items are ordered roughly by maturity: the earlier ones are closer to a
concrete implementation, the later ones are still exploratory.

Each entry is written as a short research note (motivation, current state,
proposed approach, open questions) so that a mature item can be promoted into a
full design spec when work on it begins. Shipped work is recorded in
[CHANGELOG.md](./CHANGELOG.md), not here.

**Status values:** In progress · Research · Exploratory · Planned

---

## 1. Edge linker rotation / orientation assignment

**Status:** In progress (partial) · **Difficulty:** medium

**Motivation / Problem.**
When an edge building block (a linker) is placed between two connection points,
there is a residual rotational degree of freedom about the axis joining the two
connection (`X`) points. PORMAKE currently leaves this rotation essentially
arbitrary, so a linker can be inserted at any angle around that axis. For linkers
that are not rotationally symmetric this yields twisted, unphysical geometries.

**Current state.**
This is the item previously listed in the README as *"an enhanced algorithm for
improved placement of edge building blocks (considering symmetry)."* The
[geonho42 fork](https://github.com/geonho42/PORMAKE) already tackles the random
axial rotation for a class of cases and is a valuable starting point. It is not
yet clear that a rule-based fix alone is complete across all topologies and
linker symmetries.

**Investigation.**
A detailed investigation — precise code diagnosis (the 2-point Kabsch twist
degeneracy), prior-art analysis (the geonho42 fork and external tools such as
ToBaCCo, AuToGraFS, MOFBuilder), an adversarial comparison of three candidate
algorithms, and a recommended per-edge numerical approach (a
relaxation-surviving node-frame seed plus a 1-D axial optimization that extends
`Locator.locate`) — is written up in
[docs/research/edge-linker-twist-investigation.md](./docs/research/edge-linker-twist-investigation.md).

**Proposed approach.**
A two-stage scheme:
1. **Rule-based initial assignment** — derive each edge's orientation from the
   given topology (neighboring node orientations and local symmetry).
2. **Post-optimization** — because the optimized structure differs from the
   idealized input topology, a follow-up optimization refines the orientations on
   the *relaxed* geometry rather than trusting the rule-based assignment alone.

**Open questions.**
- Which rules generalize across topologies versus need per-topology handling?
- What is the objective for the post-optimization step (steric overlap, symmetry
  agreement, an energy proxy)?
- How does this interact with per-slot linker assignment (different linkers on
  edges of the same type)?

---

## 2. Energetically-correct MOF-5 (mirror-symmetric nodes)

**Status:** Research · **Difficulty:** medium–high

**Motivation / Problem.**
The current recipe for MOF-5 is artificial. Built naively on the `pcu` net,
MOF-5 has a single node type, so PORMAKE places one identical Zn₄O node building
block into every node slot with the same orientation. In the real MOF-5 crystal,
the Zn₄O clusters facing each other across a linker must be **mirror images** of
one another; only then do the benzene-dicarboxylate linkers connect without
twisting, which is the energetically favorable geometry. A single identical
orientation forces the linkers into a strained, twisted arrangement.

**Proposed approach.**
- Expand the `pcu` topology to a **2×2×2 supercell** so the eight nodes become
  individually addressable slots.
- Assign node orientations so that neighboring clusters are mirror-related (the
  correct alternating pattern).
- Because encoding that alternating pattern as an explicit rule is awkward, try a
  **symmetry-breaking trick**: apply a small perturbation to each node's atomic
  positions to gently break the exact symmetry, then run the existing orientation
  optimization. The hope is that the optimizer relaxes into the correct
  alternating, mirror-symmetric arrangement on its own.

**Open questions.**
- What perturbation magnitude and scheme reliably break the symmetry without
  landing in a wrong local minimum?
- Does the orientation optimizer actually converge to the correct mirror pattern,
  and how do we verify it (e.g. against the experimental MOF-5 structure)?
- Should supercell expansion be exposed as a general utility, useful well beyond
  MOF-5?

---

## 3. Topology symmetry-subgroup descent (toward P1)

**Status:** Research · **Difficulty:** high

**Motivation / Problem.**
A topology's node and edge *types* are fixed by its symmetry: highly symmetric
nets expose few distinct types, which limits how much can be varied when
assigning building blocks. Many interesting structures — and the symmetry
breaking needed for item 2 — require finer control than the parent symmetry
allows.

**Proposed approach.**
Starting from a topology's space-group symmetry, descend through
**group–subgroup relations** one maximal subgroup at a time, all the way down to
**P1**. At each step, slots that were symmetry-equivalent split into inequivalent
ones, generating **new node types and edge types**. Exposing this descent yields
a systematic ladder of progressively lower-symmetry variants of a net, making it
easy to design chimera-style structures (different building blocks on slots that
were previously locked together).

**Open questions.**
- How are subgroup paths enumerated and selected (there can be many)?
- How does the recomputed type information integrate with the existing
  `Topology` data structures and `.describe()` output?
- Relationship to item 2: MOF-5's alternating pattern is one concrete instance of
  a symmetry-lowered net.

---

## 4. 1D / 2D linker-based construction

**Status:** Exploratory · **Difficulty:** high

**Motivation / Problem.**
PORMAKE is built around 0D (point-like) nodes and linkers. Construction with
**1D (rod / chain)** and **2D (sheet)** building blocks — e.g. rod-shaped
secondary building units — is currently only possible in an ad-hoc way, with no
systematic methodology yet.

**Current state.**
No explicit support. This is the least-developed direction.

**Open questions.**
- How should rod / sheet building blocks and their connection points be
  represented? The `X`-point model assumes discrete connection points.
- How far does the topology abstraction need to extend to describe periodic
  rod / layer connectivity?
- This entry intentionally records the direction only; it should get its own
  brainstorming and design pass before any implementation.

---

## 5. Simple web application

**Status:** Planned · **Difficulty:** medium

A lightweight web application for generating porous materials interactively
(carried over from the original roadmap). This is an engineering / tooling item
rather than a research direction: expose topology and building-block selection
and MOF construction through a browser front end.
