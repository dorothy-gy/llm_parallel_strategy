"""
Graph Analyzer for GPT Model Graphs

This module provides analysis capabilities for GPT model graphs including:
- Graph partitioning for pipeline parallelism
- Performance analysis and optimization
- Memory and computation distribution analysis
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
import networkx as nx

from gpt_model_graph import GPTModelGraph, LayerNode, LayerType


@dataclass
class PartitionInfo:
    """Information about a graph partition (pipeline stage)"""
    partition_id: int
    layer_ids: List[int]
    start_layer: int
    end_layer: int
    total_parameters: int
    total_memory_mb: float
    total_compute_gflops: float
    communication_cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert partition info to dictionary"""
        return {
            'partition_id': self.partition_id,
            'layer_ids': self.layer_ids,
            'start_layer': self.start_layer,
            'end_layer': self.end_layer,
            'total_parameters': self.total_parameters,
            'total_memory_mb': self.total_memory_mb,
            'total_compute_gflops': self.total_compute_gflops,
            'communication_cost': self.communication_cost
        }


@dataclass
class GraphPartitionResult:
    """Result of graph partitioning for pipeline parallelism"""
    partitions: List[PartitionInfo]
    total_stages: int
    balance_score: float
    max_memory_usage: float
    total_communication_cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert partition result to dictionary"""
        return {
            'partitions': [p.to_dict() for p in self.partitions],
            'total_stages': self.total_stages,
            'balance_score': self.balance_score,
            'max_memory_usage': self.max_memory_usage,
            'total_communication_cost': self.total_communication_cost
        }


class GraphAnalyzer:
    """
    Analyzer for GPT model graphs with partitioning capabilities
    """
    
    def __init__(self, model_graph: GPTModelGraph):
        """
        Initialize graph analyzer
        Args:
            model_graph: GPT model graph to analyze
        """
        self.graph = model_graph
        self.partitions: List[PartitionInfo] = []
    
    def analyze_graph_properties(self) -> Dict[str, Any]:
        """
        Analyze basic graph properties
        Returns:
            Dictionary containing graph analysis results
        """
        transformer_nodes = self.graph.get_transformer_nodes()
        execution_order = self.graph.get_execution_order()
        
        # Calculate statistics
        total_params = self.graph.get_total_parameters()
        total_memory = self.graph.get_total_memory_usage()
        total_compute = self.graph.get_total_compute_flops()
        
        # Parameter distribution
        param_by_type = {}
        memory_by_type = {}
        compute_by_type = {}
        
        for node in self.graph.nodes.values():
            node_type = node.layer_type.value
            if node_type not in param_by_type:
                param_by_type[node_type] = 0
                memory_by_type[node_type] = 0
                compute_by_type[node_type] = 0
            
            param_by_type[node_type] += node.parameter_count
            memory_by_type[node_type] += node.memory_usage
            compute_by_type[node_type] += node.compute_flops
        
        # Bottleneck analysis
        max_memory_layer = max(self.graph.nodes.values(), key=lambda x: x.memory_usage)
        max_compute_layer = max(self.graph.nodes.values(), key=lambda x: x.compute_flops)
        max_param_layer = max(self.graph.nodes.values(), key=lambda x: x.parameter_count)
        
        return {
            'basic_stats': {
                'total_layers': len(self.graph.nodes),
                'transformer_layers': len(transformer_nodes),
                'total_parameters': total_params,
                'total_memory_mb': total_memory,
                'total_compute_gflops': total_compute
            },
            'distribution': {
                'parameters_by_type': param_by_type,
                'memory_by_type': memory_by_type,
                'compute_by_type': compute_by_type
            },
            'bottlenecks': {
                'max_memory_layer': {
                    'layer_id': max_memory_layer.layer_id,
                    'name': max_memory_layer.name,
                    'memory_mb': max_memory_layer.memory_usage
                },
                'max_compute_layer': {
                    'layer_id': max_compute_layer.layer_id,
                    'name': max_compute_layer.name,
                    'compute_gflops': max_compute_layer.compute_flops
                },
                'max_param_layer': {
                    'layer_id': max_param_layer.layer_id,
                    'name': max_param_layer.name,
                    'parameters': max_param_layer.parameter_count
                }
            },
            'execution_order': execution_order
        }
    
    def partition_graph_uniform(self, num_stages: int) -> GraphPartitionResult:
        """
        Partition graph uniformly across pipeline stages
        Args:
            num_stages: Number of pipeline stages
        Returns:
            Graph partitioning result
        """
        transformer_nodes = self.graph.get_transformer_nodes()
        
        if num_stages > len(transformer_nodes):
            raise ValueError(f"Number of stages ({num_stages}) cannot exceed transformer layers ({len(transformer_nodes)})")
        
        # Get execution order and filter transformer blocks
        execution_order = self.graph.get_execution_order()
        transformer_layer_ids = [node.layer_id for node in transformer_nodes]
        transformer_layer_ids.sort()
        
        # Calculate layers per stage
        layers_per_stage = len(transformer_layer_ids) // num_stages
        remainder = len(transformer_layer_ids) % num_stages
        
        partitions = []
        current_idx = 0
        
        for stage_id in range(num_stages):
            # Calculate number of layers for this stage
            stage_layers = layers_per_stage + (1 if stage_id < remainder else 0)
            
            # Get layer IDs for this stage
            stage_layer_ids = transformer_layer_ids[current_idx:current_idx + stage_layers]
            
            # Add non-transformer layers to appropriate stages
            if stage_id == 0:
                # Add embedding layer to first stage
                embedding_nodes = self.graph.get_layers_by_type(LayerType.EMBEDDING)
                if embedding_nodes:
                    stage_layer_ids = [embedding_nodes[0].layer_id] + stage_layer_ids
            
            if stage_id == num_stages - 1:
                # Add final layer norm and output head to last stage
                layer_norm_nodes = self.graph.get_layers_by_type(LayerType.LAYER_NORM)
                output_nodes = self.graph.get_layers_by_type(LayerType.OUTPUT_HEAD)
                if layer_norm_nodes:
                    stage_layer_ids.append(layer_norm_nodes[0].layer_id)
                if output_nodes:
                    stage_layer_ids.append(output_nodes[0].layer_id)
            
            # Calculate partition statistics
            partition_info = self._calculate_partition_stats(stage_id, stage_layer_ids)
            partitions.append(partition_info)
            
            current_idx += stage_layers
        
        # Calculate overall statistics
        balance_score = self._calculate_balance_score(partitions)
        max_memory = max(p.total_memory_mb for p in partitions)
        total_comm_cost = sum(p.communication_cost for p in partitions)
        
        return GraphPartitionResult(
            partitions=partitions,
            total_stages=num_stages,
            balance_score=balance_score,
            max_memory_usage=max_memory,
            total_communication_cost=total_comm_cost
        )
    
    def partition_graph_balanced(self, num_stages: int, balance_metric: str = "memory") -> GraphPartitionResult:
        """
        Partition graph using balanced approach based on specified metric
        Args:
            num_stages: Number of pipeline stages
            balance_metric: Metric to balance ("memory", "compute", "parameters")
        Returns:
            Graph partitioning result
        """
        transformer_nodes = self.graph.get_transformer_nodes()
        
        if num_stages > len(transformer_nodes):
            raise ValueError(f"Number of stages ({num_stages}) cannot exceed transformer layers ({len(transformer_nodes)})")
        
        # Get metric values for each transformer layer
        metric_values = []
        for node in transformer_nodes:
            if balance_metric == "memory":
                metric_values.append(node.memory_usage)
            elif balance_metric == "compute":
                metric_values.append(node.compute_flops)
            elif balance_metric == "parameters":
                metric_values.append(node.parameter_count)
            else:
                raise ValueError(f"Unknown balance metric: {balance_metric}")
        
        # Use dynamic programming for balanced partitioning
        partitions_indices = self._balance_partitions(metric_values, num_stages)
        
        # Convert indices to actual partitions
        partitions = []
        transformer_layer_ids = [node.layer_id for node in transformer_nodes]
        transformer_layer_ids.sort()
        
        for stage_id, (start_idx, end_idx) in enumerate(partitions_indices):
            stage_layer_ids = transformer_layer_ids[start_idx:end_idx + 1]
            
            # Add non-transformer layers to appropriate stages
            if stage_id == 0:
                embedding_nodes = self.graph.get_layers_by_type(LayerType.EMBEDDING)
                if embedding_nodes:
                    stage_layer_ids = [embedding_nodes[0].layer_id] + stage_layer_ids
            
            if stage_id == num_stages - 1:
                layer_norm_nodes = self.graph.get_layers_by_type(LayerType.LAYER_NORM)
                output_nodes = self.graph.get_layers_by_type(LayerType.OUTPUT_HEAD)
                if layer_norm_nodes:
                    stage_layer_ids.append(layer_norm_nodes[0].layer_id)
                if output_nodes:
                    stage_layer_ids.append(output_nodes[0].layer_id)
            
            partition_info = self._calculate_partition_stats(stage_id, stage_layer_ids)
            partitions.append(partition_info)
        
        # Calculate overall statistics
        balance_score = self._calculate_balance_score(partitions)
        max_memory = max(p.total_memory_mb for p in partitions)
        total_comm_cost = sum(p.communication_cost for p in partitions)
        
        return GraphPartitionResult(
            partitions=partitions,
            total_stages=num_stages,
            balance_score=balance_score,
            max_memory_usage=max_memory,
            total_communication_cost=total_comm_cost
        )
    
    def _balance_partitions(self, values: List[float], num_partitions: int) -> List[Tuple[int, int]]:
        """
        Use dynamic programming to create balanced partitions
        Args:
            values: Values to balance across partitions
            num_partitions: Number of partitions to create
        Returns:
            List of (start_index, end_index) tuples for each partition
        """
        n = len(values)
        if num_partitions >= n:
            return [(i, i) for i in range(n)]
        
        # Prefix sums for efficient range sum calculation
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + values[i]
        
        # DP table: dp[i][j] = minimum maximum sum using j partitions for first i elements
        dp = [[float('inf')] * (num_partitions + 1) for _ in range(n + 1)]
        
        # Base case: 0 elements, 0 partitions
        dp[0][0] = 0
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, min(i, num_partitions) + 1):
                for k in range(j - 1, i):
                    partition_sum = prefix_sum[i] - prefix_sum[k]
                    dp[i][j] = min(dp[i][j], max(dp[k][j - 1], partition_sum))
        
        # Backtrack to find actual partitions
        partitions = []
        i, j = n, num_partitions
        
        while j > 0:
            # Find the k that gave us the optimal solution
            for k in range(j - 1, i):
                partition_sum = prefix_sum[i] - prefix_sum[k]
                if dp[i][j] == max(dp[k][j - 1], partition_sum):
                    partitions.append((k, i - 1))
                    i, j = k, j - 1
                    break
        
        partitions.reverse()
        return partitions
    
    def _calculate_partition_stats(self, partition_id: int, layer_ids: List[int]) -> PartitionInfo:
        """
        Calculate statistics for a partition
        Args:
            partition_id: ID of the partition
            layer_ids: List of layer IDs in the partition
        Returns:
            PartitionInfo object
        """
        total_params = 0
        total_memory = 0.0
        total_compute = 0.0
        communication_cost = 0.0
        
        for layer_id in layer_ids:
            node = self.graph.get_layer_by_id(layer_id)
            if node:
                total_params += node.parameter_count
                total_memory += node.memory_usage
                total_compute += node.compute_flops
        
        # Calculate communication cost (data transfer between stages)
        for edge in self.graph.edges:
            if edge.source_id in layer_ids and edge.target_id not in layer_ids:
                communication_cost += edge.communication_cost
        
        return PartitionInfo(
            partition_id=partition_id,
            layer_ids=layer_ids,
            start_layer=min(layer_ids) if layer_ids else -1,
            end_layer=max(layer_ids) if layer_ids else -1,
            total_parameters=total_params,
            total_memory_mb=total_memory,
            total_compute_gflops=total_compute,
            communication_cost=communication_cost
        )
    
    def _calculate_balance_score(self, partitions: List[PartitionInfo]) -> float:
        """
        Calculate balance score for partitions (lower is better)
        Args:
            partitions: List of partition information
        Returns:
            Balance score (coefficient of variation)
        """
        if not partitions:
            return float('inf')
        
        # Use memory usage as the primary balance metric
        memory_values = [p.total_memory_mb for p in partitions]
        
        mean_memory = np.mean(memory_values)
        std_memory = np.std(memory_values)
        
        # Coefficient of variation (lower is better balanced)
        cv = std_memory / mean_memory if mean_memory > 0 else float('inf')
        return cv
    
    def compare_partitioning_strategies(self, num_stages: int) -> Dict[str, Any]:
        """
        Compare different partitioning strategies
        Args:
            num_stages: Number of pipeline stages
        Returns:
            Comparison results
        """
        strategies = {}
        
        # Uniform partitioning
        try:
            uniform_result = self.partition_graph_uniform(num_stages)
            strategies['uniform'] = uniform_result.to_dict()
        except Exception as e:
            strategies['uniform'] = {'error': str(e)}
        
        # Balanced partitioning by different metrics
        for metric in ['memory', 'compute', 'parameters']:
            try:
                balanced_result = self.partition_graph_balanced(num_stages, metric)
                strategies[f'balanced_{metric}'] = balanced_result.to_dict()
            except Exception as e:
                strategies[f'balanced_{metric}'] = {'error': str(e)}
        
        return strategies
    
    def optimize_pipeline_stages(self, min_stages: int = 2, max_stages: int = 16) -> Dict[str, Any]:
        """
        Find optimal number of pipeline stages
        Args:
            min_stages: Minimum number of stages to consider
            max_stages: Maximum number of stages to consider
        Returns:
            Optimization results
        """
        transformer_nodes = self.graph.get_transformer_nodes()
        max_possible_stages = min(max_stages, len(transformer_nodes))
        
        results = {}
        best_score = float('inf')
        best_stages = min_stages
        
        for num_stages in range(min_stages, max_possible_stages + 1):
            try:
                # Use memory-balanced partitioning as default
                result = self.partition_graph_balanced(num_stages, "memory")
                
                # Calculate overall score (balance + communication cost)
                score = result.balance_score + 0.1 * result.total_communication_cost
                
                results[num_stages] = {
                    'result': result.to_dict(),
                    'score': score
                }
                
                if score < best_score:
                    best_score = score
                    best_stages = num_stages
                    
            except Exception as e:
                results[num_stages] = {'error': str(e)}
        
        return {
            'optimization_results': results,
            'recommended_stages': best_stages,
            'best_score': best_score
        }
    
    def generate_partitioning_report(self, num_stages: int) -> str:
        """
        Generate a detailed report of graph partitioning
        Args:
            num_stages: Number of pipeline stages
        Returns:
            Formatted report string
        """
        analysis = self.analyze_graph_properties()
        strategies = self.compare_partitioning_strategies(num_stages)
        
        report = []
        report.append("=== Graph Partitioning Analysis Report ===")
        report.append("")
        
        # Basic statistics
        report.append("=== Model Statistics ===")
        stats = analysis['basic_stats']
        report.append(f"Total Layers: {stats['total_layers']}")
        report.append(f"Transformer Layers: {stats['transformer_layers']}")
        report.append(f"Total Parameters: {stats['total_parameters']:,}")
        report.append(f"Total Memory: {stats['total_memory_mb']:.2f} MB")
        report.append(f"Total Compute: {stats['total_compute_gflops']:.2f} GFLOPs")
        report.append("")
        
        # Partitioning strategies comparison
        report.append(f"=== Partitioning Strategies ({num_stages} stages) ===")
        for strategy_name, strategy_result in strategies.items():
            if 'error' in strategy_result:
                report.append(f"{strategy_name}: ERROR - {strategy_result['error']}")
                continue
                
            report.append(f"\n{strategy_name.upper()}:")
            report.append(f"  Balance Score: {strategy_result['balance_score']:.4f}")
            report.append(f"  Max Memory Usage: {strategy_result['max_memory_usage']:.2f} MB")
            report.append(f"  Total Communication Cost: {strategy_result['total_communication_cost']:.2f} MB")
            
            # Show partition details
            for partition in strategy_result['partitions']:
                report.append(f"    Stage {partition['partition_id']}: "
                           f"Layers {partition['start_layer']}-{partition['end_layer']}, "
                           f"Memory: {partition['total_memory_mb']:.2f} MB, "
                           f"Params: {partition['total_parameters']:,}")
        
        report.append("")
        report.append("=== Recommendations ===")
        
        # Find best strategy
        best_strategy = None
        best_score = float('inf')
        
        for strategy_name, strategy_result in strategies.items():
            if 'error' not in strategy_result:
                score = strategy_result['balance_score'] + 0.1 * strategy_result['total_communication_cost']
                if score < best_score:
                    best_score = score
                    best_strategy = strategy_name
        
        if best_strategy:
            report.append(f"Recommended Strategy: {best_strategy}")
            report.append(f"Reason: Best balance between memory distribution and communication cost")
        else:
            report.append("No viable partitioning strategy found.")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    from gpt_model_graph import create_gpt_model_graph
    
    # Create GPT model graph
    graph = create_gpt_model_graph("gpt-1.5b")
    
    # Analyze the graph
    analyzer = GraphAnalyzer(graph)
    
    # Generate analysis report
    print(analyzer.generate_partitioning_report(num_stages=4))
    
    # Optimize number of stages
    optimization = analyzer.optimize_pipeline_stages(min_stages=2, max_stages=8)
    print(f"\nOptimal number of stages: {optimization['recommended_stages']}")