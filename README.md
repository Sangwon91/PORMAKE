# PORMAKE
**Por**ous materials **Make**r

> Python library for the construction of porous materials using topology and building blocks.

## Installation
* Dependencies

```
numpy
scipy>=1.4.1
pymatgen
ase>=3.18.0
tensorflow>=1.15|tensorflow-gpu>=1.15
```

1. Install all dependencies.

```bash
$ pip install -r requirements.txt
```

2. Install `pormake` using `setup.py`

```bash
$ python setup.py install
```

## Examples

**1. Construction of HKUST-1**

Import `pormake` .

```python
import pormake as pm
```

Load `tbo` topology from the default database.

```python
database = pm.Database()
tbo = database.get_topo("tbo")
```

You can check the information using `.describe()` method.

```python
tbo.describe()
```

In this case, there are two node types (`0` and `1`) and one edge type (`(0, 1)`). `CN` in the node information indicates coordination number (number of adjacent nodes). 

```
===============================================================================
Topology tbo
Spacegroup: Fm-3m
-------------------------------------------------------------------------------
# of slots: 152 (56 nodes, 96 edges)
# of node types: 2
# of edge types: 1

-------------------------------------------------------------------------------
Node type information
-------------------------------------------------------------------------------
Node type: 0, CN: 3
  slot indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29
                30, 31
Node type: 1, CN: 4
  slot indices: 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
                42, 43, 44, 45, 46, 47, 48, 49, 50, 51
                52, 53, 54, 55

-------------------------------------------------------------------------------
Edge type information (adjacent node types) 
-------------------------------------------------------------------------------
Edge type: (0, 1)
  slot indices: 56, 57, 58, 59, 60, 61, 62, 63, 64, 65
                66, 67, 68, 69, 70, 71, 72, 73, 74, 75
                76, 77, 78, 79, 80, 81, 82, 83, 84, 85
                86, 87, 88, 89, 90, 91, 92, 93, 94, 95
                96, 97, 98, 99, 100, 101, 102, 103, 104, 105
                106, 107, 108, 109, 110, 111, 112, 113, 114, 115
                116, 117, 118, 119, 120, 121, 122, 123, 124, 125
                126, 127, 128, 129, 130, 131, 132, 133, 134, 135
                136, 137, 138, 139, 140, 141, 142, 143, 144, 145
                146, 147, 148, 149, 150, 151
===============================================================================
```

You can also visualize the topology using `.view()` method.

```python
tbo.view()
```

<img src="doc/tbo.png" width=400>

In order to construct HKUST-1, copper paddle-wheel cluster and BTC linker are required.  You can load the building blocks from the database. All visual description of the building blocks can be found at [here](doc/building_blocks.pdf).

```python
# bb: budilding block.
# Copper paddle-wheel.
N409 = database.get_bb("N409")
# BTC linker.
N10 = database.get_bb("N10")
```

You can visualize building blocks using `.view()` method.

```python
N409.view()
N10.view()
```

<img src="doc/N409.png" width=350> <img src="doc/N10.png" width=350>

Next, make `Builder` instance.

```python
builder = pm.Builder()
```

Make node type to building block dictionary. This dictionary is used for the construction of the MOF. Building blocks have to be assigned to each node type (in this case, `0` and `1`).

```python
# N10 is assigned to node type 0 because the coordination number of node type 0 is 3.
# Likewise, N409 is assigned to node type 1.
node_bbs = {
    0: N10,
    1: N409
}
```

Construct HKUST-1 using `builder`.

```python
HKUST1 = builder.build_by_type(topology=tbo, node_bbs=node_bbs)
```

You can visualize constructed MOF using `.view()` method.

```python
HKUST1.view()
```

<img src="doc/HKUST-1.png" width=400>

And save the HKUST-1 in `cif` format.

```python
HKUST1.write_cif("HKUST-1.cif")
```



**2. Inserting edge building block to HKUST-1**

From the above example, we can insert edge building blocks between `N409` and `N10`.

Load long and thin edge building block from the database.

```python
E41 = database.get_bb("E41")
E41.view()
```

<img src="doc/E41.png" width=400>

Make edge type to building block dictionary. Edge type is a tuple of the types of adjacent nodes: (`0`, `1`).

```python
edge_bbs = {(0, 1): E41}
```

Make new MOF with  `edge_bbs`.

```python
MOF = builder.build_by_type(topology=tbo, node_bbs=node_bbs, edge_bbs=edge_bbs)
```

Check the constructed MOF.

```python
MOF.view()
```

`E41` is inserted properly between `N409` and `N10`.

<img src="doc/MOF.png" width=400>



**3. Construction of *Chimera* MOF**

`pormake` can assign different building block to each slot. In this example, we will replace some of `N409` to porphyrin.

Load porphyrin from the database.

```python
N13 = database.get_bb("N13")
N13.view()
```

<img src="doc/N13.png" width=400>

Before the next step, you should know the equivalence of the following two approaches for MOF construction. 

```python
# Approach 1.
MOF = builder.build_by_type(topology=tbo, node_bbs=node_bbs, edge_bbs=edge_bbs)

# Approach 2.
# Same operation with different code.
bbs = builder.make_bbs_by_type(topology=tbo, node_bbs=node_bbs, edge_bbs=edge_bbs)
MOF = builder.build(topology=tbo, bbs=bbs)
```

Here, `bbs` is the list of building blocks. `bbs[i]` is the building block at i'th slot.

Change some of `N409` to `N13`. You can get the indices from the `tbo.describe()`

```python
bbs = builder.make_bbs_by_type(topology=tbo, node_bbs=node_bbs, edge_bbs=edge_bbs)

bbs[33] = N13.copy()
bbs[38] = N13.copy()
bbs[40] = N13.copy()
bbs[49] = N13.copy()
bbs[53] = N13.copy()
bbs[55] = N13.copy()
```

Make chimera MOF with modified `bbs`.

```python
MOF = builder.build(topology=tbo, bbs=bbs)
MOF.view()
```

<img src="doc/chimera.png" width=400>
