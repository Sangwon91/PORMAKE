class Scaler:
    """
    Calculate transform matrix for cell rescaling.
    """
    def __init__(self):
        pass

    def calculate(self, topology, node_bbs, edge_bb=None):
        """
        Only work for single edge bb types
        and maximum two node types.
        Topology information is not used in this version.
        """
        bond_length = 1.5

        node_lengths = [n.length for n in node_bbs]

        if len(node_lengths) == 1:
            scaling_factor = 2*node_lengths[0] + bond_length
        elif len(node_lengths) == 2:
            scaling_factor = node_lengths[0] + node_lengths[1] + bond_length
        else:
            raise NotImplementedError

        if edge_bb is not None:
            edge_length = edge_bb.length
            scaling_factor += 2*edge_length + bond_length

        # Assume edge length of topology is one.
        # To be changed in the future.
        scaling_factor /= 1.0

        return scaling_factor
