import random
from pathlib import Path

import pormake as pm

database = pm.Database()
new_database = pm.Database(bb_dir=Path("./pormake_v2_database/bbs"))

topologies = database._get_topology_list()
topo = random.choice(topologies)
topo = database.get_topo(topo)


# 각 노드의 CN 저장
def get_node_cns(database):
    node_cns = {}
    for i in range(1, 89):  # M1 ~ M88
        bb_name = f"M{i}"
        try:
            node_bb = database.get_bb(bb_name)
            node_cns[bb_name] = node_bb.n_connection_points
        except Exception as e:
            print(f"Failed to get CN for {bb_name}: {e}")
    for i in range(1, 778):  # N1 ~ N777
        bb_name = f"N{i}"
        try:
            node_bb = database.get_bb(bb_name)
            node_cns[bb_name] = node_bb.n_connection_points
        except Exception as e:
            print(f"Failed to get CN for {bb_name}: {e}")
    return node_cns


# n개의 결과 생성 및 저장
def generate_outputs(n, node_cns):
    outputs = []
    while len(outputs) < n:
        try:
            # 랜덤으로 하나의 topology 선택
            topo = random.choice(topologies)
            try:
                topo = database.get_topo(topo)
            except Exception as e:
                print(f"Skipping invalid topology: {e}")
                continue
            unique_cns = topo.unique_cn

            # 고유 CN에 따라 필요한 노드 출력 및 랜덤 선택
            selected_nodes = []
            edges = []
            metal_exist = False

            for i, cn in enumerate(unique_cns):
                matching_nodes = [
                    name for name, node_cn in node_cns.items() if node_cn == cn
                ]
                if matching_nodes:
                    # M으로 시작하는 노드 필터링
                    m_nodes = [
                        node for node in matching_nodes if node.startswith("M")
                    ]
                    other_nodes = [
                        node
                        for node in matching_nodes
                        if not node.startswith("M")
                    ]

                    # 노드 중 랜덤 선택
                    if cn > 6:
                        selected_node = random.choice(other_nodes)
                    else:
                        selected_node = random.choice(m_nodes)
                        metal_exist = True
                    selected_nodes.append(selected_node)

                else:
                    print(f"No nodes found for CN {cn}.")

            if not metal_exist:
                raise ValueError("no metal")

            # Edge 결정 및 연결 검사
            unique_edges = topo.unique_edge_types
            edges = []

            for edge_type in unique_edges:
                node1_idx, node2_idx = edge_type
                node1 = selected_nodes[node1_idx]
                node2 = selected_nodes[node2_idx]

                if node1.startswith("M") and node2.startswith("M"):
                    # M-M 연결: E1~E230 중 하나 랜덤 선택
                    edge_name = f"E{random.randint(1, 230)}"
                    edges.append(edge_name)
                    print(f"Edge between {node1} and {node2}: {edge_name}")
                elif (node1.startswith("M") and node2.startswith("N")) or (
                    node1.startswith("N") and node2.startswith("M")
                ):
                    # M-N 연결: None 사용
                    edges.append('None')
                    print(f"Edge between {node1} and {node2}: None")
                elif node1.startswith("N") and node2.startswith("N"):
                    # N-N 연결: 오류 출력
                    print(f"Error: Invalid edge between {node1} and {node2}")
                    raise ValueError(
                        f"Invalid edge between {node1} and {node2}"
                    )

            # 결과 저장
            result = [topo.name] + selected_nodes + edges
            output_text = "+".join(result)
            outputs.append(output_text)
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

    # 결과를 텍스트 파일로 저장
    with open("random_mof_names.txt", "w") as f:
        for output in outputs:
            f.write(output + "\n")


# input 이름을 통해 current_mof 생성 함수
def create_mof_from_name(input_name):
    try:
        database = new_database
        parts = input_name.split("+")
        name = input_name
        topo_name = parts[0]
        # 노드와 엣지 구분
        node_names = []
        edge_names = []
        for part in parts[1:]:
            if part.startswith("E") or (part == "None"):
                edge_names.append(part)
            else:
                node_names.append(part)

        # Topology 가져오기
        topo = database.get_topo(topo_name)

        # Nodes와 Edges 설정
        current_nodes = {
            i: database.get_bb(node) for i, node in enumerate(node_names)
        }
        current_edges = {}

        for edge, edge_name in zip(topo.unique_edge_types, edge_names):
            if edge_name == "None":
                current_edges[tuple(edge)] = None

            else:
                current_edges[tuple(edge)] = database.get_bb(edge_name)

        return name, topo, current_nodes, current_edges

    except Exception as e:
        print(f"Error while creating MOF: {e}")
        return None
