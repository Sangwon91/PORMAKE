"""Access to the bundled topology and building-block libraries.

This module exposes :class:`Database`, the lookup layer that lets users
fetch reference nets and molecular fragments by name instead of dealing
with raw ``.cgd``/``.xyz`` files. It is usually the first object a caller
touches: the topologies and building blocks it returns are the inputs the
:class:`~pormake.builder.Builder` assembles into a framework.
"""
import pickle
from pathlib import Path

from .building_block import BuildingBlock
from .log import logger
from .topology import Topology


class Database:
    """Load the bundled topology and building-block libraries by name.

    PORMAKE ships a ``database/`` directory containing abstract nets as
    ``.cgd`` files (cached as ``.pickle`` for fast reloading) and
    molecular building blocks as ``.xyz`` files. ``Database`` is the
    central registry that resolves names to fully constructed
    :class:`~pormake.topology.Topology` and
    :class:`~pormake.building_block.BuildingBlock` objects, so the rest
    of the assembly pipeline can request components by short name
    instead of file path.

    By default it points at the packaged ``database/topologies`` and
    ``database/bbs`` folders, but custom directories may be supplied to
    work with user-provided libraries.

    Parameters
    ----------
    topo_dir : str or pathlib.Path, optional
        Directory holding topology ``.cgd``/``.pickle`` files. If
        ``None``, the bundled ``database/topologies`` directory is used.
    bb_dir : str or pathlib.Path, optional
        Directory holding building-block ``.xyz`` files. If ``None``,
        the bundled ``database/bbs`` directory is used.

    Attributes
    ----------
    topo_dir : pathlib.Path
        Resolved directory of topology files.
    bb_dir : pathlib.Path
        Resolved directory of building-block files.

    Raises
    ------
    Exception
        If either resolved directory does not exist.
    """

    def __init__(self, topo_dir=None, bb_dir=None):
        """Resolve and validate the topology/building-block directories."""
        db_path = Path(__file__).parent / "database"

        if topo_dir is None:
            self.topo_dir = db_path / "topologies"
            logger.debug("Default topology DB is loaded.")
        else:
            self.topo_dir = Path(topo_dir)
            logger.debug('topo_dir type is changed to %s' % type(self.topo_dir))

        if bb_dir is None:
            self.bb_dir = db_path / "bbs"
            logger.debug("Default building block DB is loaded.")
        else:
            self.bb_dir = Path(bb_dir)
            logger.debug('bb_dir type is changed to %s' % type(self.bb_dir))

        if not self.topo_dir.exists():
            message = "%s does not exist." % self.topo_dir
            logger.error(message)
            raise Exception(message)

        if not self.bb_dir.exists():
            message = "%s does not exist." % self.bb_dir
            logger.error(message)
            raise Exception(message)

    def _get_topology_list(self):
        """Return the names of all topologies in ``topo_dir``.

        Returns
        -------
        list of str
            Stems (names without extension) of every ``.cgd`` file in
            the topology directory.
        """
        return [p.stem for p in self.topo_dir.glob("*.cgd")]

    @property
    def topology_list(self):
        """list of str: Names of all available topologies."""
        return self._get_topology_list()

    @property
    def topo_list(self):
        """list of str: Convenience alias for :attr:`topology_list`."""
        return self._get_topology_list()

    def _get_bb_list(self):
        """Return the names of all building blocks in ``bb_dir``.

        Returns
        -------
        list of str
            Stems (names without extension) of every ``.xyz`` file in
            the building-block directory.
        """
        return [p.stem for p in self.bb_dir.glob("*.xyz")]

    @property
    def building_block_list(self):
        """list of str: Names of all available building blocks."""
        return self._get_bb_list()

    @property
    def bb_list(self):
        """list of str: Convenience alias for :attr:`building_block_list`."""
        return self._get_bb_list()

    def serialize(self):
        """Pre-pickle every bundled topology for fast later loading.

        Builds each topology from its ``.cgd`` file and writes a
        matching ``.pickle`` next to it. Because parsing a ``.cgd`` net
        is comparatively slow, pickling the libraries up front lets
        :meth:`get_topology` reload them quickly on subsequent runs.
        Topologies that fail to parse are skipped with a debug message.
        """
        print("Database serialization starts.")
        n_topos = len(self.topo_list)
        for i, name in enumerate(self.topo_list, start=1):
            cgd_path = self.topo_dir / (name + ".cgd")
            try:
                topo = Topology(cgd_path)
            except Exception as e:
                logger.debug(f'Invalid topology {e}')
                continue

            # Save pickle
            pickle_path = self.topo_dir / (name + ".pickle")
            with pickle_path.open("wb") as f:
                pickle.dump(topo, f)
            logger.debug("Pickle %s saved" % pickle_path)

            percent = i / n_topos * 100
            print("\rProgress: %.1f %% (%d/%d)" % (percent, i, n_topos), end="")

    def get_topology(self, name):
        """Load a topology by name, using the pickle cache when present.

        Attempts to load the pre-pickled topology first; if no pickle
        exists, the ``.cgd`` file is parsed into a
        :class:`~pormake.topology.Topology` and the result is cached as
        a ``.pickle`` for next time.

        Parameters
        ----------
        name : str
            Topology name without extension (e.g. ``"pcu"``).

        Returns
        -------
        pormake.topology.Topology
            The requested topology.

        Raises
        ------
        Exception
            If the topology cannot be loaded from pickle or ``.cgd``.
        """
        # Add .cgd to the topology name.
        pickle_path = self.topo_dir / (name + ".pickle")
        try:
            with pickle_path.open("rb") as f:
                topo = pickle.load(f)
            logger.debug("Topology is loaded from pickle.")
            return topo
        except Exception as e:
            # logger.exception(e)
            logger.debug("No %s.pickle in DB. Try cgd format.", name)

        cgd_path = self.topo_dir / (name + ".cgd")
        try:
            topo = Topology(cgd_path)
        except Exception as e:
            message = "Topology loading is failed: %s." % e
            logger.error(message)
            raise Exception(message)

        # Save pickle
        with pickle_path.open("wb") as f:
            pickle.dump(topo, f)
        logger.debug("Pickle %s saved" % pickle_path)

        return topo

    def get_topo(self, name):
        """Convenience alias for :meth:`get_topology`."""
        return self.get_topology(name)

    def get_building_block(self, name):
        """Load a building block by name from the ``.xyz`` library.

        Parameters
        ----------
        name : str
            Building-block name (with or without ``.xyz`` extension);
            only the stem is used to locate the file.

        Returns
        -------
        pormake.building_block.BuildingBlock
            The requested building block.

        Raises
        ------
        Exception
            If the building block cannot be loaded.
        """
        # Add .xyz to the building block name.
        name = Path(name).stem + ".xyz"

        path = self.bb_dir / name

        try:
            bb = BuildingBlock(path)
        except Exception as e:
            message = "BuildingBlock loading is failed: %s." % e
            logger.error(message)
            raise Exception(message)

        return bb

    def get_bb(self, name):
        """Convenience alias for :meth:`get_building_block`."""
        return self.get_building_block(name)
