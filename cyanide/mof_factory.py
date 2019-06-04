from itertools import product

import numpy as np

from .log import logger
from .mof import MOF
from .scaler import Scaler
from .locator import Locator
from .builder import Builder
from .building_block import BuildingBlock


class MofFactory:
    def __init__(self,
            topologies, all_node_bbs, all_edge_bbs,
            max_rmsd=0.25, max_ratio=1.6):
        self.topologies = topologies
        self.all_node_bbs = all_node_bbs
        self.all_edge_bbs = all_edge_bbs

        self.max_rmsd = max_rmsd
        self.max_ratio = max_ratio

        self.scaler = Scaler()
        self.locator = Locator()
        self.builder = Builder()

    def set_inputs(topologies=None, all_node_bbs=None, all_edge_bbs=None):
        if topologies is not None:
            self.topologies = topologies
        if all_noe_bbs is not None:
            self.all_node_bbs = all_node_bbs
        if all_edge_bbs is not None:
            self.all_edge_bbs = all_edge_bbs

    def set_thresholds(max_rmsd=None, max_ratio=None):
        if max_rmsd is not None:
            self.max_rmsd = max_rmsd
        if max_ratio is not None:
            self.max_ratio = max_ratio

    def manufacture(self, savedir=".", stem=""):
        for i, (t, n, e) in enumerate(self.generate_valid_combinations()):
            logger.info(
                "{}'th MOF Construction, Topology: {}".format(i, t.name)
            )
            try:
                mof = self.builder.build(t, n, e)
                save_path = f"{savedir}/{stem}{i}.cif"
                logger.info(f"Writing cif. Save path: {save_path}")
                mof.write_cif(save_path)
            except Exception as e:
                logger.error("MOF Build fails: {}".format(e))

    def generate_valid_combinations(self):
        for topology in self.topologies:
            #logger.debug("topology: {}".format(topology.name))
            gen = self._generate_valid_node_bbs_and_edge_bbs(topology)
            for node_bbs, edge_bbs in gen:
                yield topology, node_bbs, edge_bbs

    def _generate_valid_node_bbs_and_edge_bbs(self, topology):
        local_structures = topology.unique_local_structures

        valid_node_bb_indices = [[] for _ in local_structures]
        for i, target in enumerate(local_structures):
            for j, bb in enumerate(self.all_node_bbs):
                if bb.n_connection_points != len(target.atoms):
                    continue

                _, rmsd_ = self.locator.locate(target, bb)

                logger.debug(
                    "topo: {}, local {}, RMSD: {}"
                    .format(topology.name, i, rmsd_)
                )

                if rmsd_ <= self.max_rmsd:
                    valid_node_bb_indices[i].append(j)
                    logger.debug(
                        "Appended {}".format(valid_node_bb_indices[i])
                    )

        logger.debug(
            "topo {}, All {}".format(topology.name, valid_node_bb_indices)
        )
        # Check empty candidates.
        for indices in valid_node_bb_indices:
            # There is no valid node builing block at that point type.
            if not indices:
                logger.debug("Empty indices exist.")
                # Stop generation.
                return

        for node_bb_indices in product(*valid_node_bb_indices):
            logger.debug(f"node indices: {node_bb_indices}")
            node_bbs = [self.all_node_bbs[i] for i in node_bb_indices]
            gen = self._generate_valid_edge_bbs(topology, node_bbs)
            for edge_bbs in gen:
                logger.debug(f"edge_bbs: {edge_bbs}")
                yield node_bbs, edge_bbs

    def _generate_valid_edge_bbs(self, topology, node_bbs):
        unique_edge_types = [tuple(e) for e in topology.unique_edge_types]
        n_edge_types = topology.n_edge_types
        logger.debug(f"n_edge_types: {n_edge_types}")
        for edge_bbs in product(self.all_edge_bbs, repeat=n_edge_types):
            edge_bbs = {k: v for k, v in zip(unique_edge_types, edge_bbs)}

            max_len, min_len = self.scaler.calculate_max_min_edge_lengths(
                                   topology, node_bbs, edge_bbs
                               )

            ratio = max_len / min_len
            logger.debug(f"ratio: {ratio}")
            if ratio > self.max_ratio:
                continue

            yield edge_bbs
