# %%
import pormake as pm


database = pm.Database()

ith = database.get_topo("ith")
ith.view()

# %%
node_bbs = {
    0: database.get_bb("N3"),
    1: database.get_bb("N114"),
}

for bb in node_bbs.values():
    bb.view()

edge_bbs = {(0, 1): database.get_bb("E41")}
edge_bbs[(0, 1)].view()

# %%
builder = pm.Builder()
MOF = builder.build_by_type(topology=ith, node_bbs=node_bbs, edge_bbs=edge_bbs)
MOF.view()
# %%
