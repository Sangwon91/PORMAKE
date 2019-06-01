from .log import logger

import os
# Use CPUs only.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Turn off meaningless warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
logger.debug("GPUs are disabled for CPU calculation for tensorflow.")

from itertools import permutations
from collections import defaultdict

import numpy as np
import scipy as sp
import scipy.optimize

# For automatic differentiation.
import tensorflow as tf

class Scaler:
    """
    Scale topology using given nodes and edges building blocks information.
    """
    def scale(self, topology, node_bbs, edge_bbs=None, custom_edges=None):
        """
        Inputs:
            topology (Topology): topology object.
            node_bbs (List of BuildingBlocks): list of node building blocks.
            edge_bbs (Dict of BuildingBlocks): dict of edge building blocks.
                The key of the dict is (i, j) where i and j are node types.
            custom_edges: Custom edge at specific edge index e. It is a dict,
                keys are edge index and values are building block.
        """
        logger.debug("Scaler.scale starts.")

        if edge_bbs is None:
            edge_bbs = defaultdict(lambda: None)

        # make empty dictionary.
        if custom_edges is None:
            custom_edges = {}

        bond_length = 1.5

        # Get unique types of topology edges.
        edge_types = np.unique(
            topology.edge_types[topology.edge_indices],
            axis=0,
        )

        # Cast edge_bbs to defaultdict.
        edge_bbs = defaultdict(lambda: None, edge_bbs)

        # Calculate edge lengths of each type.
        edge_lengths = {}
        for i, j in edge_types:
            # BuildingBlock.length is the length between centroid and
            # a connection point.
            len_i = node_bbs[i].length
            len_j = node_bbs[j].length

            length = len_i + len_j + bond_length

            if edge_bbs[(i, j)] is not None:
                length += 2.0*edge_bbs[(i, j)].length + bond_length

            edge_lengths[(i, j)] = length

            logger.info(f"Length of edge type ({i}, {j}) = {length:.4f}")

        # Get pairs of bond indices and images (periodic boundary) and
        # target norms (lengths) of edges.
        pairs = []
        images = []
        target_norms = []
        c = topology.atoms.cell
        invc = np.linalg.inv(c)
        for e in topology.edge_indices:
            # ni: neigbor with index i.
            ni, nj = topology.neighbor_list[e]

            i = ni.index
            j = nj.index

            # Save index pair.
            pairs.append([i, j])

            # Calculate image.
            # d = d_{ij}
            d = nj.distance_vector - ni.distance_vector

            ri = topology.atoms.positions[i]
            rj = topology.atoms.positions[j]
            s = (d - (rj-ri)) @ invc

            images.append(s)

            # Calculate target norm of the edge.
            ti, tj = topology.get_edge_type(e)

            # Resizing for custom edges.
            edge_length = edge_lengths[(ti, tj)]
            edge_bb = edge_bbs[(ti, tj)]
            if e in custom_edges:
                if edge_bb is not None:
                    edge_length -= 2.0*edge_bb.length
                    edge_length += 2.0*custom_edges[e].length
                else:
                    edge_length += 2.0*custom_edges[e].length + bond_length

                logger.info(
                    f"Edge index {e} is custom, Length: {edge_length:.4f}"
                )

            target_norm = edge_length
            target_norms.append(target_norm)

        # Type casting to np.array.
        pairs = np.array(pairs)
        images = np.around(images)
        target_norms = np.array(target_norms)

        min_length = target_norms.min()
        max_length = target_norms.max()
        logger.info(f"Max edge length: {max_length:.4f}")
        logger.info(f"Min edge length: {min_length:.4f}")

        # Target norms are normalized to be the min length == 1.
        target_norms /= min_length

        max_min_ratio = max_length / min_length

        if max_min_ratio > 2.0:
            logger.warning(
                f"The max/min ratio of edge = {max_min_ratio:.2f} > 2, "
                "the optimized topology can be weird."
            )

        # Get angle triples.
        # New data view of pairs and images for the triples.
        data_view = defaultdict(list)
        for (i, j), image in zip(pairs, images):
            data_view[i].append((j, image))
            data_view[j].append((i, -image))

        # Triples for the calculatation of  angle between r_{ij} and r_{ik}.
        ij = []
        ik = []

        ij_image = []
        ik_image = []

        for i in topology.node_indices:
            neigbors = data_view[i]
            for (j, j_image), (k, k_image) in permutations(neigbors, 2):
                ij.append([i, j])
                ik.append([i, k])

                ij_image.append(j_image)
                ik_image.append(k_image)

        # Type cast.
        ij = np.array(ij)
        ik = np.array(ik)

        ij_image = np.array(ij_image)
        ik_image = np.array(ik_image)

        # Calculate original angles of nodes.
        c = topology.atoms.cell
        s = topology.atoms.get_scaled_positions()

        diff = s[np.newaxis, :, :] - s[:, np.newaxis, :]

        i = ij[:, 0]
        j = ij[:, 1]

        ij_vec = (diff[i, j] + ij_image) @ c

        i = ik[:, 0]
        k = ik[:, 1]

        ik_vec = (diff[i, k] + ik_image) @ c

        ij_norm = np.linalg.norm(ij_vec, axis=-1)
        ik_norm = np.linalg.norm(ik_vec, axis=-1)

        # Cos(angle) values.
        target_cos = np.sum(ij_vec * ik_vec, axis=-1) / ij_norm / ik_norm
        target_cos = np.around(target_cos, decimals=5)

        # Helper function for calculate of objective function.

        # Numerically safe norm for derivatives.
        def calc_norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1)+1e-3)

        def calc_norms_and_cos(s, c):
            """
            External variables:
                topology, pairs, image, ij, ik, ij_image, ik_image.
            """
            n = topology.n_all_points

            diff = s[tf.newaxis, :, :] - s[:, tf.newaxis, :]

            dist = (tf.gather_nd(diff, pairs) + images) @ c
            norms = calc_norm(dist)

            ij_vec = (tf.gather_nd(diff, ij) + ij_image) @ c
            ik_vec = (tf.gather_nd(diff, ik) + ik_image) @ c

            ij_norm = calc_norm(ij_vec)
            ik_norm = calc_norm(ik_vec)

            cos = tf.reduce_sum(ij_vec * ik_vec, axis=-1) / ij_norm / ik_norm

            return norms, cos

        def objective(s, c):
            norms, cos = calc_norms_and_cos(s, c)

            dist_error = tf.reduce_mean(tf.square(norms-target_norms))
            angle_error = tf.reduce_mean(tf.square(cos-target_cos))

            return dist_error + 0.1*angle_error

        # Functions for scipy interface.
        def fun(x):
            n = topology.n_all_points

            s = tf.reshape(x[:-9], [n, 3])
            c = tf.reshape(x[-9:], [3, 3])

            v = objective(s, c)

            return v.numpy()

        def jac(x):
            n = topology.n_all_points
            x = tf.constant(x)

            with tf.GradientTape() as tape:
                tape.watch(x)

                s = tf.reshape(x[:-9], [n, 3])
                c = tf.reshape(x[-9:], [3, 3])

                v = objective(s, c)

            dx = tape.gradient(v, x)

            return dx.numpy()

        # Prepare geometry optimization.
        c = topology.atoms.cell
        s = topology.atoms.get_scaled_positions()

        # Bounds.
        zeros = np.zeros(shape=s.size)
        ones = np.ones(shape=s.size)

        bounds = np.stack([zeros, ones], axis=1).tolist()
        for i in range(9):
            bounds.append([None, None])

        x0 = np.concatenate([s.reshape(-1), c.reshape(-1)])

        logger.info("Topology optimization starts.")
        # Perform optimization.
        result = sp.optimize.minimize(
            x0=x0,
            fun=fun,
            jac=jac,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "disp": False},
        )

        ##########################################################
        # Should print the result of optimization in the future! #
        ##########################################################

        n = topology.n_all_points
        # Get output x.
        x = result.x
        c = x[-9:].reshape(3, 3)
        s = x[:-9].reshape(n, 3)

        # Calculate adjust_ratio.
        # The adjust_ratio make all edge length > target length.
        #norms, cos = calc_norms_and_cos(s, c)
        #norms = norms.numpy()
        #cos = cos.numpy()
        #adjust_ratio = np.max(target_norms / norms)
        # Adjust cell to make sure all edges are longer than target length.
        #c *= adjust_ratio

        # Check all lengths and angles.
        norms, cos = calc_norms_and_cos(s, c)
        norms = norms.numpy()
        cos = cos.numpy()

        logger.info("Optimized edge lengths.")
        logger.info("Index Target Result")
        for e, tn, n in zip(topology.edge_indices, target_norms, norms):
            n *= min_length
            tn *= min_length
            logger.info(f"{e:5d} {tn:6.3f} {n:6.3f}")

        # Return to normalized scale to real scale.
        c *= min_length

        # Update neigbors list in topology.
        new_data = [[] for _ in range(topology.n_all_points)]
        # Transform to Cartesian coordinates.
        r = s @ c
        inv_old_c = np.linalg.inv(topology.atoms.cell)
        for e in topology.edge_indices:
            ni, nj = topology.neighbor_list[e]

            i = ni.index
            j = nj.index

            ri = topology.atoms.positions[i]
            rj = topology.atoms.positions[j]

            d = nj.distance_vector - ni.distance_vector

            image = (d - (rj-ri)) @ inv_old_c

            # New edge center.
            ri = r[i]
            rj = r[j]

            d = rj - ri + np.dot(image, c)

            rc = ri + 0.5*d

            r[e] = rc

            new_data[e] += [
                (i, -0.5*d),
                (j, 0.5*d)
            ]

            new_data[i].append((e, 0.5*d))
            new_data[j].append((e, -0.5*d))

        # Make scaled topology.
        scaled_topology = topology.copy()
        scaled_topology.atoms.set_positions(r)
        scaled_topology.atoms.set_cell(c)
        scaled_topology.neighbor_list.set_data(new_data)

        return scaled_topology
