"""
Test script for ClusterGraph (hardware cluster graph)
"""
from cluster_graph import build_local_cluster_graph, build_manual_cluster

# Test 1: Auto-detect local cluster (GPUs on this machine)
def test_local_cluster():
    print("\n[Auto-detect Local Cluster]")
    graph = build_local_cluster_graph()
    print(graph)
    for node in graph.nodes.values():
        print(node)
    for edge in graph.edges:
        print(edge)

def test_manual_cluster():
    print("\n[Manual Cluster Construction]")
    node_specs = [
        {'node_id': 'cuda:0', 'device_type': 'GPU', 'properties': {'memory_mb': 24576}},
        {'node_id': 'cuda:1', 'device_type': 'GPU', 'properties': {'memory_mb': 24576}},
        {'node_id': 'cpu:0', 'device_type': 'CPU', 'properties': {'memory_mb': 65536}},
    ]
    edge_specs = [
        {'src': 'cuda:0', 'dst': 'cuda:1', 'bandwidth_gbps': 50.0, 'latency_us': 5.0},
        {'src': 'cuda:1', 'dst': 'cuda:0', 'bandwidth_gbps': 50.0, 'latency_us': 5.0},
        {'src': 'cuda:0', 'dst': 'cpu:0', 'bandwidth_gbps': 12.0, 'latency_us': 20.0},
        {'src': 'cuda:1', 'dst': 'cpu:0', 'bandwidth_gbps': 12.0, 'latency_us': 20.0},
    ]
    graph = build_manual_cluster(node_specs, edge_specs)
    print(graph)
    for node in graph.nodes.values():
        print(node)
    for edge in graph.edges:
        print(edge)

if __name__ == "__main__":
    test_local_cluster()
    test_manual_cluster()
