# MOF Decomposer

A module for extracting building blocks from MOFs. It's refactored from the
legacy code used in the PORMAKE paper. It is an experimental feature and may not
 be stable.

## Minimal usage
For the demonstration, create a MOF CIF file using PORMAKE. We use HKUST-1 here.

```python
import pormake as pm

database = pm.Database()
tbo = database.get_topo("tbo")

# Copper paddle-wheel.
N409 = database.get_bb("N409")

# BTC linker.
N10 = database.get_bb("N10")

# Make HKUST-1.
builder = pm.Builder()
HKUST1 = builder.build_by_type(topology=tbo, node_bbs={0: N10, 1: N409})
HKUST1.write_cif('HKUST-1.cif')
```

In order to deconstruct the MOF into building blocks, import `MOFDecomposer`.
```python
from pormake.experimental.decomposer import MOFDecomposer
```

Then load CIF of HKUST-1 using the following code. Bond information in the CIF can't
be loaded in the current version. Therefore all bond connectivities are computed by
distance based method.
```python
decomposer = MOFDecomposer(cif='HKUST-1.cif')
```

You can check the loaded MOF using `view()` method.
```python
decomposer.view()
```

![](./asset/HKUST-1.png)

In addition, you can clean up MOF structure by using `cleanup()` method. This
method removes interpenetration and isolated molecules in MOFs. **We highly
recommend using `cleanup` method before extracting building blocks.**
```python
decomposer.cleanup()
```

It's almost done. Now you can access the building blocks via the `building_blocks`
property. For the first access, it can take a few seconds.

Input:
```python
decomposer.building_blocks
```

Output:
```python
[
    Atoms(symbols='C6H3X3', ...),
    Atoms(symbols='CCuO2CuCO2C2O4X4', ...),
    Atoms(symbols='CCuO2CuCO2C2O4X4', ...),
    ...
]
```

The output of building_blocks is a list of `ase.Atoms`. You can check the
extracted building blocks using the `ase.visualize.view` method.

```python
from ase.visualize import view

view(decomposer.building_blocks[0])
view(decomposer.building_blocks[1])
```

![](./asset/node1.png) ![](./asset/node2.png)

Furthermore, you can group duplicated building blocks by using the `hash_atoms`
function. `hash_atoms` is a simple hash function that converts atoms object into
an unique integer.

Input:
```python
from collections import defaultdict
from pormake.experimental.decomposer import hash_atoms


bb_dict = defaultdict(list)
for atoms in decomposer.building_blocks:
    hash_ = hash_atoms(atoms)
    bb_dict[hash_].append(atoms)

print(bb_dict.keys())
```

Output:
```
dict_keys([154263, 865992])
```

As expected, there are only two types of building blocks in HKUST-1
(Copper paddle-wheel and BTC-linker).

Finally, you can check the dimension of building block using
the `estimate_atoms_dimension` function. For the current version of PORMAKE, building
blocks with zero dimension are allowed only.

Input:
```python
from pormake.experimental.decomposer import estimate_atoms_dimension

estimate_atoms_dimension(decomposer.building_blocks[0])
```

Output:
```
0
```
