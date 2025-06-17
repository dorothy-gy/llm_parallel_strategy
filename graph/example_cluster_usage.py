"""
Example usage of ClusterGraph for heterogeneous cluster representation
"""
from cluster_graph import build_local_cluster_graph, build_manual_cluster

# Example: Build and print local cluster graph
graph = build_local_cluster_graph()
print("[Local Cluster Graph]")
print(graph)
for node in graph.nodes.values():
    print(node)
for edge in graph.edges:
    print(edge)

# Example: Build a manual cluster graph (multi-node, multi-device)
manual_nodes = [
    {'node_id': 'node1:cuda:0', 'device_type': 'GPU', 'properties': {'memory_mb': 24576}},
    {'node_id': 'node1:cuda:1', 'device_type': 'GPU', 'properties': {'memory_mb': 24576}},
    {'node_id': 'node2:cuda:0', 'device_type': 'GPU', 'properties': {'memory_mb': 40960}},
    {'node_id': 'node2:cpu:0', 'device_type': 'CPU', 'properties': {'memory_mb': 128000}},
]
manual_edges = [
    {'src': 'node1:cuda:0', 'dst': 'node1:cuda:1', 'bandwidth_gbps': 50.0, 'latency_us': 5.0},
    {'src': 'node1:cuda:1', 'dst': 'node1:cuda:0', 'bandwidth_gbps': 50.0, 'latency_us': 5.0},
    {'src': 'node1:cuda:0', 'dst': 'node2:cuda:0', 'bandwidth_gbps': 10.0, 'latency_us': 100.0},
    {'src': 'node2:cuda:0', 'dst': 'node1:cuda:0', 'bandwidth_gbps': 10.0, 'latency_us': 100.0},
    {'src': 'node2:cuda:0', 'dst': 'node2:cpu:0', 'bandwidth_gbps': 12.0, 'latency_us': 20.0},
]
man_graph = build_manual_cluster(manual_nodes, manual_edges)
print("\n[Manual Multi-node Cluster Graph]")
print(man_graph)
for node in man_graph.nodes.values():
    print(node)
for edge in man_graph.edges:
    print(edge)
