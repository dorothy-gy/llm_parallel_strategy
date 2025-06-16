"""
GPT Model Graph Representation

This module provides functionality to represent GPT models as directed graphs
where each layer is a node. This representation is designed for pipeline parallel
optimization using graph partitioning algorithms.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import networkx as nx
import numpy as np


class LayerType(Enum):
    """Enumeration of different layer types in GPT model"""
    EMBEDDING = "embedding"
    TRANSFORMER_BLOCK = "transformer_block"
    LAYER_NORM = "layer_norm"
    OUTPUT_HEAD = "output_head"
    ATTENTION = "attention"
    MLP = "mlp"


@dataclass
class LayerNode:
    """Represents a layer node in the GPT model graph"""
    layer_id: int
    layer_type: LayerType
    name: str
    hidden_size: int
    seq_length: int
    num_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    vocab_size: Optional[int] = None
    
    # Performance characteristics
    parameter_count: int = 0
    memory_usage: float = 0.0  # MB
    compute_flops: float = 0.0  # GFLOPS
    
    # Parallel configuration hints
    tensor_parallel_dim: Optional[int] = None
    supports_pipeline_parallel: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation"""
        return {
            'layer_id': self.layer_id,
            'layer_type': self.layer_type.value,
            'name': self.name,
            'hidden_size': self.hidden_size,
            'seq_length': self.seq_length,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'vocab_size': self.vocab_size,
            'parameter_count': self.parameter_count,
            'memory_usage': self.memory_usage,
            'compute_flops': self.compute_flops,
            'tensor_parallel_dim': self.tensor_parallel_dim,
            'supports_pipeline_parallel': self.supports_pipeline_parallel
        }


@dataclass
class EdgeInfo:
    """Represents an edge between layers in the model graph"""
    source_id: int
    target_id: int
    data_shape: Tuple[int, ...]
    data_type: str = "float32"
    communication_cost: float = 0.0  # MB/s
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation"""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'data_shape': self.data_shape,
            'data_type': self.data_type,
            'communication_cost': self.communication_cost
        }


class GPTModelGraph:
    """
    Graph representation of GPT model with layers as nodes
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GPT model graph
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.nodes: Dict[int, LayerNode] = {}
        self.edges: List[EdgeInfo] = []
        self.graph = nx.DiGraph()
        
        # Extract model parameters
        self.num_layers = config.get("n_layer", 48)
        self.hidden_size = config.get("n_embd", 1600)
        self.num_heads = config.get("n_head", 32)
        self.head_dim = config.get("head_dim", 50)
        self.vocab_size = config.get("vocab_size", 50257)
        self.seq_length = config.get("n_positions", 1024)
        self.intermediate_size = self.hidden_size * 4  # Standard GPT MLP ratio
        
        self._build_graph()
    
    def _build_graph(self):
        """Build the complete GPT model graph"""
        current_id = 0
        
        # 1. Word Embedding Layer
        embedding_node = self._create_embedding_node(current_id)
        self._add_node(embedding_node)
        current_id += 1
        
        # 2. Transformer Blocks
        transformer_nodes = []
        for layer_idx in range(self.num_layers):
            # Each transformer block as a single node
            transformer_node = self._create_transformer_block_node(current_id, layer_idx)
            self._add_node(transformer_node)
            transformer_nodes.append(transformer_node)
            
            # Add edge from previous layer
            if layer_idx == 0:
                self._add_edge(embedding_node.layer_id, transformer_node.layer_id)
            else:
                self._add_edge(transformer_nodes[layer_idx-1].layer_id, transformer_node.layer_id)
            
            current_id += 1
        
        # 3. Final Layer Norm
        layer_norm_node = self._create_layer_norm_node(current_id)
        self._add_node(layer_norm_node)
        self._add_edge(transformer_nodes[-1].layer_id, layer_norm_node.layer_id)
        current_id += 1
        
        # 4. Output Head (LM Head)
        output_node = self._create_output_head_node(current_id)
        self._add_node(output_node)
        self._add_edge(layer_norm_node.layer_id, output_node.layer_id)
    
    def _create_embedding_node(self, layer_id: int) -> LayerNode:
        """Create word embedding layer node"""
        param_count = self.vocab_size * self.hidden_size
        memory_usage = param_count * 4 / (1024 * 1024)  # Float32 to MB
        compute_flops = self.seq_length * self.hidden_size * 2  # Lookup + forward
        
        return LayerNode(
            layer_id=layer_id,
            layer_type=LayerType.EMBEDDING,
            name="word_embeddings",
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
            parameter_count=param_count,
            memory_usage=memory_usage,
            compute_flops=compute_flops,
            tensor_parallel_dim=self.vocab_size,
            supports_pipeline_parallel=True
        )
    
    def _create_transformer_block_node(self, layer_id: int, block_idx: int) -> LayerNode:
        """Create transformer block node"""
        # Parameters: attention + MLP + layer norms
        attention_params = 4 * self.hidden_size * self.hidden_size  # Q, K, V, O projections
        mlp_params = 2 * self.hidden_size * self.intermediate_size  # up and down projections
        layernorm_params = 2 * self.hidden_size  # 2 layer norms
        total_params = attention_params + mlp_params + layernorm_params
        
        memory_usage = total_params * 4 / (1024 * 1024)  # Float32 to MB
        
        # Compute FLOPs (simplified estimation)
        attention_flops = 4 * self.seq_length * self.seq_length * self.hidden_size  # Attention
        mlp_flops = 8 * self.seq_length * self.hidden_size * self.intermediate_size  # MLP
        compute_flops = (attention_flops + mlp_flops) / 1e9  # Convert to GFLOPs
        
        return LayerNode(
            layer_id=layer_id,
            layer_type=LayerType.TRANSFORMER_BLOCK,
            name=f"transformer_block_{block_idx}",
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            parameter_count=total_params,
            memory_usage=memory_usage,
            compute_flops=compute_flops,
            tensor_parallel_dim=self.num_heads,
            supports_pipeline_parallel=True
        )
    
    def _create_layer_norm_node(self, layer_id: int) -> LayerNode:
        """Create final layer norm node"""
        param_count = self.hidden_size
        memory_usage = param_count * 4 / (1024 * 1024)
        compute_flops = self.seq_length * self.hidden_size * 5 / 1e9  # Norm operations
        
        return LayerNode(
            layer_id=layer_id,
            layer_type=LayerType.LAYER_NORM,
            name="final_layer_norm",
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
            parameter_count=param_count,
            memory_usage=memory_usage,
            compute_flops=compute_flops,
            tensor_parallel_dim=None,
            supports_pipeline_parallel=True
        )
    
    def _create_output_head_node(self, layer_id: int) -> LayerNode:
        """Create output head (LM head) node"""
        param_count = self.hidden_size * self.vocab_size
        memory_usage = param_count * 4 / (1024 * 1024)
        compute_flops = self.seq_length * self.hidden_size * self.vocab_size / 1e9
        
        return LayerNode(
            layer_id=layer_id,
            layer_type=LayerType.OUTPUT_HEAD,
            name="lm_head",
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
            parameter_count=param_count,
            memory_usage=memory_usage,
            compute_flops=compute_flops,
            tensor_parallel_dim=self.vocab_size,
            supports_pipeline_parallel=True
        )
    
    def _add_node(self, node: LayerNode):
        """Add a node to the graph"""
        self.nodes[node.layer_id] = node
        self.graph.add_node(
            node.layer_id,
            **node.to_dict()
        )
    
    def _add_edge(self, source_id: int, target_id: int):
        """Add an edge between two nodes"""
        # Calculate data transfer size
        data_shape = (1, self.seq_length, self.hidden_size)  # Batch size assumed to be 1
        data_size_mb = np.prod(data_shape) * 4 / (1024 * 1024)  # Float32 to MB
        
        edge = EdgeInfo(
            source_id=source_id,
            target_id=target_id,
            data_shape=data_shape,
            communication_cost=data_size_mb
        )
        
        self.edges.append(edge)
        self.graph.add_edge(
            source_id,
            target_id,
            **edge.to_dict()
        )
    
    def get_transformer_nodes(self) -> List[LayerNode]:
        """Get all transformer block nodes"""
        return [node for node in self.nodes.values() 
                if node.layer_type == LayerType.TRANSFORMER_BLOCK]
    
    def get_execution_order(self) -> List[int]:
        """Get the execution order of layers"""
        return list(nx.topological_sort(self.graph))
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters in the model"""
        return sum(node.parameter_count for node in self.nodes.values())
    
    def get_total_memory_usage(self) -> float:
        """Get total memory usage in MB"""
        return sum(node.memory_usage for node in self.nodes.values())
    
    def get_total_compute_flops(self) -> float:
        """Get total compute FLOPs in GFLOPs"""
        return sum(node.compute_flops for node in self.nodes.values())
    
    def get_layer_by_id(self, layer_id: int) -> Optional[LayerNode]:
        """Get layer node by ID"""
        return self.nodes.get(layer_id)
    
    def get_layers_by_type(self, layer_type: LayerType) -> List[LayerNode]:
        """Get all layers of a specific type"""
        return [node for node in self.nodes.values() if node.layer_type == layer_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire graph to dictionary"""
        return {
            'config': self.config,
            'nodes': {str(k): v.to_dict() for k, v in self.nodes.items()},
            'edges': [edge.to_dict() for edge in self.edges],
            'stats': {
                'total_layers': len(self.nodes),
                'total_parameters': self.get_total_parameters(),
                'total_memory_mb': self.get_total_memory_usage(),
                'total_compute_gflops': self.get_total_compute_flops()
            }
        }
    
    def save_to_json(self, filepath: str):
        """Save graph to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def visualize_graph(self) -> str:
        """Generate a text representation of the graph"""
        output = []
        output.append("=== GPT Model Graph ===")
        output.append(f"Model: GPT-{self.num_layers} layers")
        output.append(f"Hidden size: {self.hidden_size}")
        output.append(f"Attention heads: {self.num_heads}")
        output.append(f"Sequence length: {self.seq_length}")
        output.append(f"Vocabulary size: {self.vocab_size}")
        output.append("")
        
        output.append("=== Layer Structure ===")
        execution_order = self.get_execution_order()
        
        for layer_id in execution_order:
            node = self.nodes[layer_id]
            output.append(f"Layer {layer_id}: {node.name}")
            output.append(f"  Type: {node.layer_type.value}")
            output.append(f"  Parameters: {node.parameter_count:,}")
            output.append(f"  Memory: {node.memory_usage:.2f} MB")
            output.append(f"  Compute: {node.compute_flops:.2f} GFLOPs")
            if node.tensor_parallel_dim:
                output.append(f"  Tensor Parallel Dim: {node.tensor_parallel_dim}")
            output.append("")
        
        output.append("=== Model Statistics ===")
        output.append(f"Total Parameters: {self.get_total_parameters():,}")
        output.append(f"Total Memory: {self.get_total_memory_usage():.2f} MB")
        output.append(f"Total Compute: {self.get_total_compute_flops():.2f} GFLOPs")
        
        return "\n".join(output)


def load_gpt_config(model_size: str = "gpt-1.5b") -> Dict[str, Any]:
    """
    Load GPT configuration from galvatron meta configs
    Args:
        model_size: Model size identifier (e.g., "gpt-1.5b")
    Returns:
        Configuration dictionary
    """
    # Get the current script directory and find galvatron path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    galvatron_path = os.path.dirname(current_dir)
    
    # Try to load from galvatron meta configs
    config_path = os.path.join(galvatron_path, "galvatron", "models", "gpt_hf", "meta_configs", f"{model_size}.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Fallback configuration path
        config_path = os.path.join(galvatron_path, "galvatron", "models", "gpt_fa", "meta_configs", f"{model_size}.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default GPT-1.5B configuration
            config = {
                "n_layer": 48,
                "n_embd": 1600,
                "n_head": 32,
                "head_dim": 50,
                "vocab_size": 50257,
                "n_positions": 1024
            }
    
    return config


def create_gpt_model_graph(model_size: str = "gpt-1.5b") -> GPTModelGraph:
    """
    Create GPT model graph from configuration
    Args:
        model_size: Model size identifier
    Returns:
        GPTModelGraph instance
    """
    config = load_gpt_config(model_size)
    return GPTModelGraph(config)


if __name__ == "__main__":
    # Example usage
    graph = create_gpt_model_graph("gpt-1.5b")
    print(graph.visualize_graph())
    
    # Save graph to JSON
    output_path = os.path.join(os.path.dirname(__file__), "gpt_1.5b_graph.json")
    graph.save_to_json(output_path)
    print(f"\nGraph saved to: {output_path}")