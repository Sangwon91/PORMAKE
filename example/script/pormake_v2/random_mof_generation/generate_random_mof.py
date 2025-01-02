from pathlib import Path

from random_mof_generator import generate_outputs, get_node_cns

import pormake as pm

new_database = pm.Database(bb_dir=Path("./pormake_v2_database/bbs"))
node_cns = get_node_cns(new_database)

# n results, 원하는 mof 개수
n = 10
generate_outputs(n, node_cns)
