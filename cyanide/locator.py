import itertools

import numpy as np

import sklearn
import sklearn.cluster

from .third_party.Bio import SVDSuperimposer

class Locator:
    def __init__(self):
        self.superimposer = SVDSuperimposer()

    def reduce_points(self, points):
        def centers(p, n):
            kmeans = sklearn.cluster.KMeans(n_clusters=n, random_state=0)
            return kmeans.fit(p).cluster_centers_

        if len(points) <= 8:
            return points
        elif len(points) == 10:
            return centers(points, 4)
        elif len(points) == 12:
            return centers(points, 8)
        elif len(points) == 24:
            return centers(points, 8)

    def locate(self, target, bb):
        """
        Locate building block (bb) to target_points
        using the connection points of the bb.

        Return:
            located building block and RMS.
        """
        target_points = target.positions
        points = bb.local_structure().positions

        # Reduce points for fast calculations.
        target_points = self.reduce_points(target_points)
        points = self.reduce_points(points)

        # Alias.
        imposer = self.superimposer
        # Variable to find minimum RMS superimposition.
        min_rms = 1e30
        min_rms_transform = None
        # Calculate all possible permutations.
        for p in itertools.permutations(points):
            # Cast to numpy array.
            p = np.array(p)

            imposer.set(target_points, p)
            imposer.run()

            rms = imposer.get_rms()
            if rms < min_rms:
                min_rms = rms
                min_rms_transform = imposer.get_rotran()

        rot, trans = min_rms_transform
        bb = bb.copy()
        bb.atoms.set_positions(bb.atoms.positions @ rot + trans)

        return bb, min_rms
