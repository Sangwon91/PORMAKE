import pormake as pm

database = pm.Database()
tbo = database.get_topo("tbo")

# Copper paddle-wheel.
N409 = pm.BuildingBlock("./bbs/N409_charged.xyz")

# BTC linker.
N10 = pm.BuildingBlock("./bbs/N10_charged.xyz")

# Make HKUST-1.
build = pm.Builder()
HKUST1 = build.build_by_type(topology=tbo, node_bbs={0: N10, 1: N409})
HKUST1.write_cif('HKUST_1.cif')
