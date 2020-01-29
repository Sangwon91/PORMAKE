from .log import logger
from .topology import Topology
from .building_block import BuildingBlock

from pathlib import Path

class Database:
    def __init__(self):
        db_dir = Path(__file__).parent / "database"
        self.topo_dir = db_dir / "topologies"
        self.bb_dir = db_dir / "bbs"

    def _get_topology_list(self):
        return [p.stem for p in self.topo_dir.glob("*.cgd")]

    @property
    def topology_list(self):
        return self._get_topology_list()

    def _get_bb_list(self):
        return [p.stem for p in self.bb_dir.glob("*.xyz")]

    @property
    def bb_list(self):
        return self._get_bb_list()

    def get_topology(self, name):
        # Add .cgd to the topology name.
        name = Path(name).stem + ".cgd"

        path = self.topo_dir / name

        try:
            topo = Topology(path)
        except Exception as e:
            message = "Topology loading is failed: %s." % e
            logger.error(message)
            raise Exception(message)

        return topo

    def get_topo(self, name):
        return self.get_topology(name)

    def get_building_block(self, name):
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
        return self.get_building_block(name)
