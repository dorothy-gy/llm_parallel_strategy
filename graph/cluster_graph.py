"""
Cluster Graph Representation for Heterogeneous Device Clusters

This module defines the data structures and utilities to represent a hardware cluster as a graph.
Each node is a compute device (GPU, CPU, etc.), and each edge represents the connection (bandwidth, latency, etc.) between devices.

Features:
- Auto-detect local GPU devices (using torch/nvidia-smi if available)
- Allow manual specification of cluster topology
- Support for bandwidth/latency/other edge attributes
- Export and visualization utilities
"""

import os
from typing import List, Dict, Optional, Any

class DeviceType:
    GPU = "GPU"
    CPU = "CPU"
    OTHER = "OTHER"

class ClusterNode:
    def __init__(self, node_id: str, device_type: str, properties: Optional[Dict[str, Any]] = None):
        self.node_id = node_id  # e.g. 'cuda:0', 'cpu:0', 'node1:cuda:0'
        self.device_type = device_type  # GPU/CPU/OTHER
        self.properties = properties or {}  # e.g. {'memory': 24*1024, 'compute_capability': '8.0'}

    def __repr__(self):
        return f"<ClusterNode {self.node_id} ({self.device_type}) {self.properties}>"

class ClusterEdge:
    def __init__(self, src: str, dst: str, bandwidth_gbps: float, latency_us: float, properties: Optional[Dict[str, Any]] = None):
        self.src = src
        self.dst = dst
        self.bandwidth_gbps = bandwidth_gbps
        self.latency_us = latency_us
        self.properties = properties or {}

    def __repr__(self):
        return f"<Edge {self.src}->{self.dst} {self.bandwidth_gbps}Gbps {self.latency_us}us>"

class ClusterGraph:
    def __init__(self):
        self.nodes: Dict[str, ClusterNode] = {}
        self.edges: List[ClusterEdge] = []

    def add_node(self, node: ClusterNode):
        self.nodes[node.node_id] = node

    def add_edge(self, edge: ClusterEdge):
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[ClusterNode]:
        return self.nodes.get(node_id)

    def neighbors(self, node_id: str) -> List[ClusterNode]:
        return [self.nodes[e.dst] for e in self.edges if e.src == node_id]

    def as_dict(self):
        return {
            'nodes': {nid: vars(n) for nid, n in self.nodes.items()},
            'edges': [vars(e) for e in self.edges]
        }

    def __repr__(self):
        return f"<ClusterGraph nodes={len(self.nodes)} edges={len(self.edges)}>"

# --- Utilities for auto-detecting local GPUs ---
def detect_local_gpus() -> List[ClusterNode]:
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        nodes = []
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            node = ClusterNode(
                node_id=f"cuda:{i}",
                device_type=DeviceType.GPU,
                properties={
                    'name': props.name,
                    'memory_mb': props.total_memory // (1024*1024),
                    'compute_capability': f"{props.major}.{props.minor}",
                }
            )
            nodes.append(node)
        return nodes
    except ImportError:
        return []
    except Exception:
        return []

def detect_interconnect_bandwidth(num_gpus: int) -> List[ClusterEdge]:
    # For local GPUs, assume full mesh NVLink/PCIe (bandwidth/latency can be refined)
    edges = []
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                # Example: NVLink ~50GBps, PCIe ~12GBps, latency ~5us
                edges.append(ClusterEdge(
                    src=f"cuda:{i}", dst=f"cuda:{j}",
                    bandwidth_gbps=50.0,  # default, can be refined
                    latency_us=5.0
                ))
    return edges

# --- Manual cluster construction ---
def build_manual_cluster(node_specs: List[Dict], edge_specs: List[Dict]) -> ClusterGraph:
    graph = ClusterGraph()
    for n in node_specs:
        graph.add_node(ClusterNode(**n))
    for e in edge_specs:
        graph.add_edge(ClusterEdge(**e))
    return graph

# --- Main entry for auto-detecting local cluster ---
def build_local_cluster_graph() -> ClusterGraph:
    graph = ClusterGraph()
    nodes = detect_local_gpus()
    for node in nodes:
        graph.add_node(node)
    edges = detect_interconnect_bandwidth(len(nodes))
    for edge in edges:
        graph.add_edge(edge)
    return graph

# --- Visualization (optional, for later extension) ---
# def visualize_cluster_graph(graph: ClusterGraph, save_path: Optional[str] = None):
#     ...
