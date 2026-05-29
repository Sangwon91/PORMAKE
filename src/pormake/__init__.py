"""PORMAKE: a builder for porous crystalline materials.

PORMAKE ("porous material maker") assembles periodic reticular
structures -- primarily Metal-Organic Frameworks (MOFs) and related
porous crystals -- by placing molecular building blocks onto the
vertices and edges of an abstract topology (a "net").

The high-level workflow combines an abstract net, real molecular
fragments, a geometric alignment step, and a unit-cell rescaling step
to emit a chemically reasonable periodic framework:

1. A :class:`Topology` describes how vertices (nodes) and edge centers
   connect, read from a ``.cgd`` net file.
2. A :class:`BuildingBlock` is a molecular fragment read from an
   ``.xyz`` file, where dummy "X" atoms mark connection points.
3. A :class:`Database` loads the bundled topology and building-block
   libraries by name.
4. A :class:`Locator` aligns a building block onto a topology slot by
   minimizing the RMSD between connection-point directions.
5. A :class:`Scaler` rescales the topology unit cell so node-to-node
   distances match the real building-block sizes.
6. A :class:`Builder` orchestrates the assembly and produces the final
   periodic framework, writable to CIF.

Public API
----------
Builder
    Orchestrates assembly of building blocks onto a topology.
BuildingBlock
    Molecular fragment with marked connection points.
Database
    Loader for the bundled topology/building-block libraries.
Locator
    Aligns building blocks onto topology slots by RMSD minimization.
Scaler
    Rescales the topology cell to fit the chosen building blocks.
Topology
    Abstract net of nodes and edge centers with connectivity.
"""

from .builder import Builder
from .building_block import BuildingBlock
from .database import Database
from .locator import Locator
from .scaler import Scaler
from .topology import Topology
