import numpy as np

import ase
import ase.neighborlist


class Neighbor:
    def __init__(self, index, distance_vector):
        self.index = index
        self.distance_vector = distance_vector

    def __repr__(self):
        return "index: {}, distance vector: {}".format(
                   self.index, self.distance_vector
               )


class NeighborList:
    """
    Make the connectivity between nodes and edge centers for the topologies.
    """
    def __init__(self, atoms):
        # Should be changed "0.5 threshold" to "nearest 2 node for edge".
        # Because there are edges length != 1.0 (e.g. ith-d).
        # It occurs because the cgd file "try to" make all edge length to 1.0.
        eps = 1e-3
        cutoffs = {
            ("C", "C"): 0.0,
            ("O", "O"): 0.0,
            ("C", "O"): 0.5+eps,
        }

        I, J, D = ase.neighborlist.neighbor_list("ijD", atoms, cutoff=cutoffs)

        self.max_index = np.max(I)
        self.neighbor_list = [[] for _ in range(self.max_index+1)]

        for i, j, d in zip(I, J, D):
            self.neighbor_list[i].append(Neighbor(j, d))

        """
        # Pick nearest 2 nodes.
        edge_indices = np.argwhere(atoms.symbols == "O").reshape(-1)

        for i, l in enumerate(self.neighbor_list):
            l.sort(key=lambda x: np.linalg.norm(x.distance_vector))
            #l = l[:2]
            print([np.linalg.norm(n.distance_vector) for n in l])
        """

    def __getitem__(self, i):
        return self.neighbor_list[i]

    def __iter__(self):
        return iter(self.neighbor_list)

    def __repr__(self):
        output = ""
        for i, l in enumerate(self):
            line = "{}: {}\n".format(i, len(l))
            for n in l:
                line += "{}\n".format(n)
            output += line
        return output
