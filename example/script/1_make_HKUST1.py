# %%
import pormake as pm

database = pm.Database()

tbo = database.get_topo("tbo")
tbo.view()

# %%
# Copper paddle-wheel.
N409 = database.get_bb("N409")
# BTC linker.
N10 = database.get_bb("N10")
N409.view()
N10.view()

# %%
builder = pm.Builder()

node_bbs = {0: N10, 1: N409}

HKUST1 = builder.build_by_type(topology=tbo, node_bbs=node_bbs)
HKUST1.view()
HKUST1.write_cif('HKUST1.cif')
# %%
