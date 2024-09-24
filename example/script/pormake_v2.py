import copy
import json
import time

import numpy as np
from m3gnet.models import Relaxer
from pymatgen.core import Element, Structure

import pormake as pm

# original Pormake setting

name = "mof_name"
database = pm.Database()
edge_bb = database.get_bb("E14")
topo = database.get_topo("pcu")
node_bb = database.get_bb("N16")

builder = pm.Builder()
current_node = {}
current_edge = {}

current_node[0] = node_bb
current_edge[(0, 0)] = edge_bb


# edge에 None이 들어가는지 확인, 첫 edge가 None일 경우 처음으로 나오는 None이 아닌 edge를 first_valid_edge_index로 설정하고, edge_representer 설정

for i, edge in enumerate(topo.edge_types):
    if tuple(edge.tolist()) in current_edge:
        if current_edge[tuple(edge.tolist())] is not None:
            first_valid_edge_index = i - topo.n_nodes
            edge_representer = current_edge[
                tuple(edge.tolist())
            ].find_furthest_atom_index()
            break

none_edge_list = []
for i, edge in enumerate(topo.edge_types):
    if i >= topo.n_nodes:
        if current_edge[tuple(edge.tolist())] is None:
            none_edge_list.append(1)
        else:
            none_edge_list.append(0)

print(first_valid_edge_index, none_edge_list, edge_representer)

# Original Pormake

GUN1 = builder.build_by_type(
    topo,
    current_node,
    current_edge,
    first_valid_edge_index=first_valid_edge_index,
    edge_representer=edge_representer,
)
GUN1.write_cif(name + '_ori_Pormake.cif', spacegroup_vis=False)

# initialization
# edge_angle_interval = 1, 3, 5, 10
# first_edge_angle_interval = 15, 30, 45
# threshold < 0.5

edge_angle_interval = 5
first_edge_angle_interval = 30
threshold = 0.3
energy_per_atom_list = []
min_angle_list_list = []

# edge 회전해서 space group에 맞게 edge 들어가도록 조정
a = time.time()

with open('./log.out', 'a') as f:
    for i in range(0, 91, first_edge_angle_interval):
        rotating_angle_list = [0] * topo.n_edges
        rotating_angle_list[first_valid_edge_index] = i
        min_angle_list = [0] * topo.n_edges
        min_angle_list[first_valid_edge_index] = i
        min_error = [100] * topo.n_edges
        min_error[first_valid_edge_index] = 0
        for e in range(0, topo.n_edges):
            if none_edge_list[e] == 1:
                min_error[e] = 0
        for j in range(0, 180, edge_angle_interval):
            rotating_angle_list = [j] * topo.n_edges
            rotating_angle_list[first_valid_edge_index] = i
            GUN1 = builder.build_by_type(
                topo,
                current_node,
                current_edge,
                first_valid_edge_index=first_valid_edge_index,
                rotating_angle_list=rotating_angle_list,
                edge_representer=edge_representer,
            )
            for k in [
                x for x in range(0, topo.n_edges) if x != first_valid_edge_index
            ]:
                if GUN1.min_array[k] < min_error[k]:  # update
                    min_error[k] = GUN1.min_array[k]
                    min_angle_list[k] = j
            print(
                f'first edge angle: {i}, remain edge angle: {j}',
                file=f,
                flush=True,
            )
            # print(rotating_angle_list, file = f, flush = True)
            # print(GUN1.min_array, file = f, flush = True)
            print(
                f'angle: {min_angle_list}\nerror: {min_error}',
                file=f,
                flush=True,
            )
            if all(value < threshold for value in min_error):
                break
        GUN1 = builder.build_by_type(
            topo,
            current_node,
            current_edge,
            first_valid_edge_index=first_valid_edge_index,
            rotating_angle_list=min_angle_list,
            edge_representer=edge_representer,
        )

        ase_atoms = GUN1.atoms
        lattice = ase_atoms.cell.array
        filtered_positions = []
        filtered_species = []
        relaxer = Relaxer()
        for atom in ase_atoms:
            if atom.symbol != 'Ne':
                filtered_positions.append(atom.position)
                filtered_species.append(Element(atom.symbol))
        pmg_structure = Structure(lattice, filtered_species, filtered_positions)
        relax_results = relaxer.relax(pmg_structure, steps=1)
        energy_per_atom = float(
            relax_results['trajectory'].energies[-1] / len(pmg_structure)
        )
        energy_per_atom_list.append(energy_per_atom)
        min_angle_list_list.append(min_angle_list)

    min_angle_list = min_angle_list_list[np.argmin(energy_per_atom_list)]

    GUN1 = builder.build_by_type(
        topo,
        current_node,
        current_edge,
        first_valid_edge_index=first_valid_edge_index,
        rotating_angle_list=min_angle_list,
        edge_representer=edge_representer,
    )
    print(
        f'Final angle: {min_angle_list}\nFinal error: {GUN1.min_array}',
        file=f,
        flush=True,
    )

# 기준점 여러개일 경우 extra 기준점 추가하기

extra = []
with open('./log.out', 'a') as f:
    while max(GUN1.min_array) > 1.0:
        print("add extra edge", file=f, flush=True)
        extra.append(np.argmax(GUN1.min_array))
        rotating_angle_list = [0] * topo.n_edges
        rotating_angle_list[first_valid_edge_index] = min_angle_list[
            first_valid_edge_index
        ]
        for ex in extra:
            rotating_angle_list[ex] = min_angle_list[ex]
        min_angle_list = copy.deepcopy(rotating_angle_list)
        min_error = [100] * topo.n_edges
        min_error[first_valid_edge_index] = 0
        for ex in extra:
            min_error[ex] = 0
        for e in range(0, topo.n_edges):
            if none_edge_list[e] == 1:
                min_error[e] = 0
        for j in range(0, 180, edge_angle_interval):
            for i in range(0, topo.n_edges):
                if i not in extra and i != first_valid_edge_index:
                    rotating_angle_list[i] = j
            GUN1 = builder.build_by_type(
                topo,
                current_node,
                current_edge,
                first_valid_edge_index=first_valid_edge_index,
                rotating_angle_list=rotating_angle_list,
                edge_representer=edge_representer,
                extra=extra,
            )
            for k in range(0, topo.n_edges):
                if GUN1.min_array[k] < min_error[k]:  # update
                    min_error[k] = GUN1.min_array[k]
                    min_angle_list[k] = j
            print(
                f'angle: {min_angle_list}\nerror: {min_error}',
                file=f,
                flush=True,
            )
            if all(value < threshold for value in min_error):
                break

        GUN1 = builder.build_by_type(
            topo,
            current_node,
            current_edge,
            first_valid_edge_index=first_valid_edge_index,
            rotating_angle_list=min_angle_list,
            edge_representer=edge_representer,
            extra=extra,
        )
        print(
            f'Final angle(extra): {min_angle_list}\nFinal error(extra): {GUN1.min_array}',
            file=f,
            flush=True,
        )

    print('Done', file=f, flush=True)

# Generating Final MOF Structure
# spacegroup_vis=True로 할 경우 space group atom 보이게 cif 파일 생성 가능
GUN1.write_cif(name + '_new_Pormake.cif', spacegroup_vis=False)

b = time.time()

# relaxation
pmg_structure = Structure.from_file(name + '_new_Pormake.cif')
relaxer = Relaxer()
relax_results = relaxer.relax(pmg_structure, steps=100000, fmax=0.05)
energy_per_atom = float(
    relax_results['trajectory'].energies[-1] / len(pmg_structure)
)
final_structure = relax_results['final_structure']
final_structure.to(filename=name + '_new_Pormake_relax.cif')

c = time.time()

# Result file
filename = name + 'output.json'
data = {
    "angle_interval": edge_angle_interval,
    "threshold": threshold,
    "time": b - a,
    "relaxation time": c - b,
    "min_array": GUN1.min_array,
    "min_angle": min_angle_list,
    "energy_per_atom": energy_per_atom,
}

with open(filename, 'w') as file:
    json.dump(data, file, indent=4)
