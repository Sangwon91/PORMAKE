from pathlib import Path
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
            try:
                logger.info(
                    "{}'th MOF Construction, Topology: {}".format(i, t.name)
                )

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

                _, _, rmsd_ = self.locator.locate(target, bb)

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
            # Only MOF and assume linker has organics.
            has_metal = False
            for i in node_bbs:
                if i.has_metal:
                    has_metal = True
                    logger.debug(f"{i} has metal.")
            if not has_metal:
                continue
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
            yield edge_bbs

class RandomMofFactory:
    def __init__(self, all_topologies, all_node_bbs, all_edge_bbs):
        self.all_topologies = np.array(all_topologies, dtype=object)
        self.all_node_bbs = np.array(all_node_bbs, dtype=object)
        self.all_edge_bbs = np.array(all_edge_bbs, dtype=object)

    def manufacture(self,
            n_samples, max_rmsd=0.25, savedir=".", key_file="key.txt"):
        locator = Locator()
        builder = Builder()

        n_all_topologies = self.all_topologies.shape[0]
        n_all_node_bbs = self.all_node_bbs.shape[0]
        n_all_edge_bbs = self.all_edge_bbs.shape[0]

        mof_book = set()
        if Path(key_file).exists():
            with open(key_file, "r") as f:
                for line in f:
                    mof_book.add(line.split()[1])

            key_file = open(key_file, "a")
        else:
            key_file = open(key_file, "w")

        while len(mof_book) < n_samples:
            # Pick random topology
            i = np.random.randint(0, n_all_topologies)
            topology = self.all_topologies[i]

            # Pick two edges.
            indices = np.random.randint(0, n_all_edge_bbs, size=2)
            two_edge_bbs = self.all_edge_bbs[indices]
            def rand_edge():
                if np.random.rand() < 0.5:
                    return two_edge_bbs[0]
                else:
                    return two_edge_bbs[1]

            edge_bbs = {
                (i, j): rand_edge() for i, j in topology.unique_edge_types
            }

            # Pick random nodes.
            local_structures = topology.unique_local_structures
            node_bbs = []
            for target in local_structures:
                node_bb_found = False
                perm = np.random.permutation(n_all_node_bbs)
                for bb in self.all_node_bbs[perm]:
                    if bb.n_connection_points != len(target.atoms):
                        continue

                    _, _, rmsd_ = locator.locate(target, bb)

                    if rmsd_ <= max_rmsd:
                        node_bbs.append(bb)
                        node_bb_found = True
                        break

                if not node_bb_found:
                    break

            if len(node_bbs) != len(local_structures):
                continue

            has_metal = False
            for bb in node_bbs+list(edge_bbs.values()):
                if bb.has_metal:
                    has_metal = True
                    break
            # Only MOFs.
            if not has_metal:
                continue

            key = topology.name + ":"
            name_list = [bb.name for bb in node_bbs+list(edge_bbs.values())]
            key += ":".join(name_list)

            print("Trying key: {}".format(key))
            # Only unique MOFs.
            if key in mof_book:
                continue

            mof_book.add(key)

            try:
                n_mofs = len(mof_book)
                mof = builder.build(topology, node_bbs, edge_bbs)
                abc = mof.atoms.get_cell_lengths_and_angles()[:3]
                if (abc < 4.5).any():
                    logger.warning("Too small cell. Skip.")
                    continue

                mof.write_cif("{}/{}.cif".format(savedir, n_mofs))

                key_file.write("{} {}\n".format(n_mofs, key))
                key_file.flush()
                print(n_mofs, key)
            except Exception as e:
                print(e)
