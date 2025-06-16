"""
Visualization utilities for GPT Model Graph

This module provides visualization capabilities for the GPT model graph
to help understand the model structure and partitioning strategies.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Optional
import os

from gpt_model_graph import create_gpt_model_graph, LayerType
from graph_analyzer import GraphAnalyzer


def visualize_model_structure(graph, save_path: Optional[str] = None):
    """
    Create a visualization of the model structure
    Args:
        graph: GPTModelGraph instance
        save_path: Optional path to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("Matplotlib and NetworkX are required for visualization")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Model layer structure
    layers = list(graph.nodes.values())
    layer_types = [node.layer_type.value for node in layers]
    memory_usage = [node.memory_usage for node in layers]
    
    # Color coding for layer types
    colors = {
        'embedding': 'lightblue',
        'transformer_block': 'lightgreen', 
        'layer_norm': 'yellow',
        'output_head': 'lightcoral'
    }
    
    bar_colors = [colors.get(lt, 'gray') for lt in layer_types]
    
    ax1.bar(range(len(layers)), memory_usage, color=bar_colors)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Memory Usage by Layer')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, color=color, label=layer_type) 
                      for layer_type, color in colors.items()]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Parameter distribution
    param_by_type = {}
    for node in layers:
        node_type = node.layer_type.value
        if node_type not in param_by_type:
            param_by_type[node_type] = 0
        param_by_type[node_type] += node.parameter_count
    
    ax2.pie(param_by_type.values(), labels=param_by_type.keys(), autopct='%1.1f%%')
    ax2.set_title('Parameter Distribution by Layer Type')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def visualize_partitioning_strategies(analyzer, num_stages: int, save_path: Optional[str] = None):
    """
    Visualize different partitioning strategies
    Args:
        analyzer: GraphAnalyzer instance
        num_stages: Number of pipeline stages
        save_path: Optional path to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required for visualization")
        return
    
    strategies = analyzer.compare_partitioning_strategies(num_stages)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    strategy_names = ['uniform', 'balanced_memory', 'balanced_compute', 'balanced_parameters']
    
    for i, strategy_name in enumerate(strategy_names):
        if strategy_name not in strategies or 'error' in strategies[strategy_name]:
            axes[i].text(0.5, 0.5, f'{strategy_name}\nERROR', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{strategy_name.replace("_", " ").title()}')
            continue
        
        strategy_data = strategies[strategy_name]
        partitions = strategy_data['partitions']
        
        # Extract data
        stage_ids = [p['partition_id'] for p in partitions]
        memory_usage = [p['total_memory_mb'] for p in partitions]
        param_counts = [p['total_parameters'] / 1e6 for p in partitions]  # Convert to millions
        
        # Create bar chart
        x = np.arange(len(stage_ids))
        width = 0.35
        
        ax = axes[i]
        bars1 = ax.bar(x - width/2, memory_usage, width, label='Memory (MB)', alpha=0.8)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, param_counts, width, label='Parameters (M)', alpha=0.8, color='orange')
        
        ax.set_xlabel('Pipeline Stage')
        ax.set_ylabel('Memory Usage (MB)')
        ax2.set_ylabel('Parameters (Millions)')
        ax.set_title(f'{strategy_name.replace("_", " ").title()}\nBalance Score: {strategy_data["balance_score"]:.4f}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Stage {sid}' for sid in stage_ids])
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}M', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'Partitioning Strategies Comparison ({num_stages} stages)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Partitioning visualization saved to: {save_path}")
    else:
        plt.show()


def visualize_stage_optimization(analyzer, min_stages: int = 2, max_stages: int = 12, save_path: Optional[str] = None):
    """
    Visualize the optimization of pipeline stage count
    Args:
        analyzer: GraphAnalyzer instance
        min_stages: Minimum number of stages
        max_stages: Maximum number of stages
        save_path: Optional path to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required for visualization")
        return
    
    # Collect data for different stage counts
    stage_counts = []
    balance_scores = []
    max_memories = []
    comm_costs = []
    
    transformer_nodes = analyzer.graph.get_transformer_nodes()
    max_possible_stages = min(max_stages, len(transformer_nodes))
    
    for num_stages in range(min_stages, max_possible_stages + 1):
        try:
            result = analyzer.partition_graph_balanced(num_stages, "memory")
            stage_counts.append(num_stages)
            balance_scores.append(result.balance_score)
            max_memories.append(result.max_memory_usage)
            comm_costs.append(result.total_communication_cost)
        except Exception as e:
            print(f"Error for {num_stages} stages: {e}")
            continue
    
    if not stage_counts:
        print("No valid stage configurations found")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Balance Score vs Stage Count
    axes[0, 0].plot(stage_counts, balance_scores, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Number of Pipeline Stages')
    axes[0, 0].set_ylabel('Balance Score (lower is better)')
    axes[0, 0].set_title('Load Balance vs Pipeline Stages')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Max Memory vs Stage Count
    axes[0, 1].plot(stage_counts, max_memories, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Number of Pipeline Stages')
    axes[0, 1].set_ylabel('Max Memory per Stage (MB)')
    axes[0, 1].set_title('Memory Usage vs Pipeline Stages')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Communication Cost vs Stage Count
    axes[1, 0].plot(stage_counts, comm_costs, 'go-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Number of Pipeline Stages')
    axes[1, 0].set_ylabel('Communication Cost (MB)')
    axes[1, 0].set_title('Communication Cost vs Pipeline Stages')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Combined Score (Balance + 0.1 * Comm Cost)
    combined_scores = [bs + 0.1 * cc for bs, cc in zip(balance_scores, comm_costs)]
    axes[1, 1].plot(stage_counts, combined_scores, 'mo-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Number of Pipeline Stages')
    axes[1, 1].set_ylabel('Combined Score (lower is better)')
    axes[1, 1].set_title('Overall Optimization Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Mark optimal point
    if combined_scores:
        min_idx = np.argmin(combined_scores)
        optimal_stages = stage_counts[min_idx]
        optimal_score = combined_scores[min_idx]
        
        axes[1, 1].scatter([optimal_stages], [optimal_score], 
                          color='red', s=100, zorder=5, marker='*')
        axes[1, 1].annotate(f'Optimal: {optimal_stages} stages', 
                           (optimal_stages, optimal_score),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.suptitle('Pipeline Stage Optimization Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimization visualization saved to: {save_path}")
    else:
        plt.show()


def generate_all_visualizations():
    """Generate all visualizations for the GPT model graph"""
    
    print("Generating visualizations for GPT model graph...")
    
    # Create model graph
    graph = create_gpt_model_graph("gpt-1.5b")
    analyzer = GraphAnalyzer(graph)
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        matplotlib_available = True
    except ImportError:
        print("Matplotlib not available. Generating text-based visualizations instead.")
        matplotlib_available = False
    
    if matplotlib_available:
        # Generate visualizations
        print("\n1. Generating model structure visualization...")
        visualize_model_structure(graph, "model_structure.png")
        
        print("\n2. Generating partitioning strategies visualization...")
        visualize_partitioning_strategies(analyzer, 4, "partitioning_strategies.png")
        
        print("\n3. Generating stage optimization visualization...")
        visualize_stage_optimization(analyzer, 2, 12, "stage_optimization.png")
        
        print("\nAll visualizations generated successfully!")
        
    else:
        # Generate text-based alternatives
        print("\n1. Model Structure Summary:")
        print(graph.visualize_graph())
        
        print("\n2. Partitioning Analysis:")
        report = analyzer.generate_partitioning_report(4)
        print(report)
        
        print("\n3. Stage Optimization:")
        optimization = analyzer.optimize_pipeline_stages(2, 12)
        print(f"Recommended stages: {optimization['recommended_stages']}")
        print(f"Best score: {optimization['best_score']:.4f}")


if __name__ == "__main__":
    generate_all_visualizations()
