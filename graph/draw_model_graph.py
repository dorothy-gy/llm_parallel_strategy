"""
GPT Model Graph Visualization

This script creates detailed visualizations of the GPT model graph structure,
showing the layer-wise representation and connections.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from typing import Dict, List
import json

from gpt_model_graph import create_gpt_model_graph, LayerType
from graph_analyzer import GraphAnalyzer


def create_detailed_graph_visualization():
    """Create a comprehensive visualization of the GPT model graph"""
    
    # Create model graph
    graph = create_gpt_model_graph("gpt-1.5b")
    analyzer = GraphAnalyzer(graph)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Subplot 1: Network Graph Structure
    ax1 = plt.subplot(2, 3, 1)
    draw_network_graph(graph, ax1)
    
    # Subplot 2: Layer Memory Distribution
    ax2 = plt.subplot(2, 3, 2)
    draw_memory_distribution(graph, ax2)
    
    # Subplot 3: Layer Type Distribution
    ax3 = plt.subplot(2, 3, 3)
    draw_layer_type_distribution(graph, ax3)
    
    # Subplot 4: Sequential Layer View
    ax4 = plt.subplot(2, 3, 4)
    draw_sequential_view(graph, ax4)
    
    # Subplot 5: Pipeline Partitioning
    ax5 = plt.subplot(2, 3, 5)
    draw_pipeline_partitioning(analyzer, ax5)
    
    # Subplot 6: Model Statistics
    ax6 = plt.subplot(2, 3, 6)
    draw_model_statistics(graph, ax6)
    
    plt.suptitle('GPT-1.5B Model Graph Visualization', fontsize=20, y=0.98)
    plt.tight_layout()
    plt.savefig('comprehensive_model_graph.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return graph


def draw_network_graph(graph, ax):
    """Draw the network graph structure"""
    # Create NetworkX graph for visualization
    G = nx.DiGraph()
    
    # Add nodes with attributes
    node_colors = []
    node_sizes = []
    labels = {}
    
    color_map = {
        LayerType.EMBEDDING: '#FF6B6B',
        LayerType.TRANSFORMER_BLOCK: '#4ECDC4', 
        LayerType.LAYER_NORM: '#FFE66D',
        LayerType.OUTPUT_HEAD: '#FF8E53'
    }
    
    # Sample nodes for visualization (show every 4th transformer block)
    nodes_to_show = []
    for node_id, node in graph.nodes.items():
        if (node.layer_type != LayerType.TRANSFORMER_BLOCK or 
            node_id % 4 == 1 or 
            node_id in [1, len(graph.nodes)-2]):  # Show first and last transformer
            nodes_to_show.append(node_id)
    
    for node_id in nodes_to_show:
        node = graph.nodes[node_id]
        G.add_node(node_id)
        node_colors.append(color_map.get(node.layer_type, '#CCCCCC'))
        node_sizes.append(max(100, min(1000, node.memory_usage * 2)))
        
        # Create labels
        if node.layer_type == LayerType.TRANSFORMER_BLOCK:
            layer_num = node_id - 1  # Adjust for embedding layer
            labels[node_id] = f'T{layer_num}'
        else:
            labels[node_id] = node.name.split('_')[0][:3].upper()
    
    # Add edges
    for i in range(len(nodes_to_show) - 1):
        G.add_edge(nodes_to_show[i], nodes_to_show[i + 1])
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title('Model Graph Structure\n(Node size ∝ Memory Usage)', fontsize=12)
    ax.axis('off')
    
    # Legend
    legend_elements = [mpatches.Patch(color=color, label=layer_type.value.replace('_', ' ').title()) 
                      for layer_type, color in color_map.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def draw_memory_distribution(graph, ax):
    """Draw memory usage distribution across layers"""
    layers = list(graph.nodes.values())
    layer_ids = [node.layer_id for node in layers]
    memory_usage = [node.memory_usage for node in layers]
    
    # Color by layer type
    colors = []
    color_map = {
        LayerType.EMBEDDING: '#FF6B6B',
        LayerType.TRANSFORMER_BLOCK: '#4ECDC4',
        LayerType.LAYER_NORM: '#FFE66D', 
        LayerType.OUTPUT_HEAD: '#FF8E53'
    }
    
    for node in layers:
        colors.append(color_map.get(node.layer_type, '#CCCCCC'))
    
    bars = ax.bar(layer_ids, memory_usage, color=colors, alpha=0.7)
    
    # Highlight embedding and output layers
    for i, node in enumerate(layers):
        if node.layer_type in [LayerType.EMBEDDING, LayerType.OUTPUT_HEAD]:
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2)
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Distribution Across Layers')
    ax.grid(True, alpha=0.3)
    
    # Annotate special layers
    for i, node in enumerate(layers):
        if node.layer_type == LayerType.EMBEDDING:
            ax.annotate('Embedding', (node.layer_id, node.memory_usage),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        elif node.layer_type == LayerType.OUTPUT_HEAD:
            ax.annotate('Output', (node.layer_id, node.memory_usage),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)


def draw_layer_type_distribution(graph, ax):
    """Draw pie chart of layer type distribution"""
    layer_counts = {}
    memory_by_type = {}
    
    for node in graph.nodes.values():
        layer_type = node.layer_type.value
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        memory_by_type[layer_type] = memory_by_type.get(layer_type, 0) + node.memory_usage
    
    # Create pie chart for memory distribution
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#FF8E53']
    wedges, texts, autotexts = ax.pie(memory_by_type.values(), 
                                     labels=[f"{k.replace('_', ' ').title()}\n({layer_counts[k]} layers)" 
                                            for k in memory_by_type.keys()],
                                     autopct='%1.1f%%',
                                     colors=colors,
                                     startangle=90)
    
    ax.set_title('Memory Distribution by Layer Type')
    
    # Add total memory in center
    total_memory = sum(memory_by_type.values())
    ax.text(0, 0, f'Total\n{total_memory:.1f} MB', 
           ha='center', va='center', fontsize=10, fontweight='bold')


def draw_sequential_view(graph, ax):
    """Draw sequential view of the model"""
    layers = list(graph.nodes.values())
    
    # Create a simplified sequential representation
    y_positions = []
    colors = []
    labels = []
    widths = []
    
    color_map = {
        LayerType.EMBEDDING: '#FF6B6B',
        LayerType.TRANSFORMER_BLOCK: '#4ECDC4',
        LayerType.LAYER_NORM: '#FFE66D',
        LayerType.OUTPUT_HEAD: '#FF8E53'
    }
    
    current_y = 0
    for node in layers:
        if node.layer_type == LayerType.TRANSFORMER_BLOCK:
            # Group transformer blocks
            if not y_positions or colors[-1] != color_map[LayerType.TRANSFORMER_BLOCK]:
                y_positions.append(current_y)
                colors.append(color_map[LayerType.TRANSFORMER_BLOCK])
                labels.append(f'Transformer Blocks\n(48 layers)')
                widths.append(len(graph.get_transformer_nodes()))
                current_y += 1
        else:
            y_positions.append(current_y)
            colors.append(color_map[node.layer_type])
            labels.append(node.name.replace('_', ' ').title())
            widths.append(1)
            current_y += 1
    
    # Draw horizontal bars
    bars = ax.barh(range(len(y_positions)), widths, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Layer Count')
    ax.set_title('Sequential Model Structure')
    ax.invert_yaxis()
    
    # Add parameter counts
    param_info = {
        'Embedding': graph.get_layers_by_type(LayerType.EMBEDDING)[0].parameter_count,
        'Transformer': sum(n.parameter_count for n in graph.get_transformer_nodes()),
        'LayerNorm': graph.get_layers_by_type(LayerType.LAYER_NORM)[0].parameter_count,
        'Output': graph.get_layers_by_type(LayerType.OUTPUT_HEAD)[0].parameter_count
    }
    
    for i, (bar, label) in enumerate(zip(bars, labels)):
        if 'Transformer' in label:
            params = param_info['Transformer'] / 1e6
        elif 'Embedding' in label:
            params = param_info['Embedding'] / 1e6  
        elif 'Norm' in label:
            params = param_info['LayerNorm'] / 1e6
        else:
            params = param_info['Output'] / 1e6
        
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
               f'{params:.1f}M params', va='center', fontsize=8)


def draw_pipeline_partitioning(analyzer, ax):
    """Draw pipeline partitioning visualization"""
    # Get optimal partitioning
    result = analyzer.partition_graph_balanced(4, "memory")
    
    # Create visualization
    stage_ids = [p.partition_id for p in result.partitions]
    memory_usage = [p.total_memory_mb for p in result.partitions]
    layer_counts = [p.end_layer - p.start_layer + 1 for p in result.partitions]
    
    x = np.arange(len(stage_ids))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, memory_usage, width, label='Memory (MB)', alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, layer_counts, width, label='Layer Count', 
                   alpha=0.8, color='orange')
    
    ax.set_xlabel('Pipeline Stage')
    ax.set_ylabel('Memory Usage (MB)', color='blue')
    ax2.set_ylabel('Number of Layers', color='orange')
    ax.set_title(f'Pipeline Partitioning (4 stages)\nBalance Score: {result.balance_score:.4f}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Stage {sid}' for sid in stage_ids])
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')


def draw_model_statistics(graph, ax):
    """Draw model statistics summary"""
    ax.axis('off')
    
    # Calculate statistics
    total_params = graph.get_total_parameters()
    total_memory = graph.get_total_memory_usage()
    total_compute = graph.get_total_compute_flops()
    transformer_nodes = graph.get_transformer_nodes()
    
    # Create text summary
    stats_text = f"""
GPT-1.5B Model Statistics

Architecture:
• Total Layers: {len(graph.nodes)}
• Transformer Blocks: {len(transformer_nodes)}
• Hidden Size: {graph.hidden_size}
• Attention Heads: {graph.num_heads}
• Sequence Length: {graph.seq_length}
• Vocabulary Size: {graph.vocab_size:,}

Resources:
• Total Parameters: {total_params:,}
  ({total_params/1e9:.2f} Billion)
• Total Memory: {total_memory:.2f} MB
  ({total_memory/1024:.2f} GB)
• Total Compute: {total_compute:.2f} GFLOPs

Layer Distribution:
• Embedding: {len(graph.get_layers_by_type(LayerType.EMBEDDING))} layer
• Transformer: {len(transformer_nodes)} layers
• Layer Norm: {len(graph.get_layers_by_type(LayerType.LAYER_NORM))} layer
• Output Head: {len(graph.get_layers_by_type(LayerType.OUTPUT_HEAD))} layer

Performance Characteristics:
• Avg Memory/Layer: {total_memory/len(graph.nodes):.2f} MB
• Avg Params/Transformer: {sum(n.parameter_count for n in transformer_nodes)/len(transformer_nodes)/1e6:.1f}M
• Memory Efficiency: {total_params/(total_memory*1024*1024/4):.2f}
"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
    
    ax.set_title('Model Overview', fontsize=12, pad=20)


def create_simplified_graph_diagram():
    """Create a simplified, easy-to-understand graph diagram"""
    
    graph = create_gpt_model_graph("gpt-1.5b")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Create a flow diagram
    layers = list(graph.nodes.values())
    
    # Define positions
    positions = {}
    labels = {}
    colors = {}
    sizes = {}
    
    color_map = {
        LayerType.EMBEDDING: '#FF6B6B',
        LayerType.TRANSFORMER_BLOCK: '#4ECDC4',
        LayerType.LAYER_NORM: '#FFE66D',
        LayerType.OUTPUT_HEAD: '#FF8E53'
    }
    
    # Position nodes in a flow
    x_pos = 0
    y_pos = 0
    
    for i, node in enumerate(layers):
        if node.layer_type == LayerType.EMBEDDING:
            positions[node.layer_id] = (x_pos, 0)
            labels[node.layer_id] = 'Word\nEmbedding'
            x_pos += 2
            
        elif node.layer_type == LayerType.TRANSFORMER_BLOCK:
            # Group transformer blocks
            if i <= 12:  # First quarter
                positions[node.layer_id] = (x_pos, 1)
                if i == 1:
                    labels[node.layer_id] = 'Transformer\nBlocks 1-12'
                    x_pos += 3
            elif i <= 24:  # Second quarter
                positions[node.layer_id] = (x_pos, 1)
                if i == 13:
                    labels[node.layer_id] = 'Transformer\nBlocks 13-24'
                    x_pos += 3
            elif i <= 36:  # Third quarter
                positions[node.layer_id] = (x_pos, 1)
                if i == 25:
                    labels[node.layer_id] = 'Transformer\nBlocks 25-36'
                    x_pos += 3
            else:  # Fourth quarter
                positions[node.layer_id] = (x_pos, 1)
                if i == 37:
                    labels[node.layer_id] = 'Transformer\nBlocks 37-48'
                    x_pos += 3
                    
        elif node.layer_type == LayerType.LAYER_NORM:
            positions[node.layer_id] = (x_pos, 0)
            labels[node.layer_id] = 'Final\nLayer Norm'
            x_pos += 2
            
        elif node.layer_type == LayerType.OUTPUT_HEAD:
            positions[node.layer_id] = (x_pos, 0)
            labels[node.layer_id] = 'Output\nHead'
        
        colors[node.layer_id] = color_map[node.layer_type]
        sizes[node.layer_id] = max(1000, min(3000, node.memory_usage * 3))
    
    # Create NetworkX graph for visualization
    G = nx.DiGraph()
    
    # Only show representative nodes
    representative_nodes = [0, 1, 13, 25, 37, 49, 50]  # Key layers
    for node_id in representative_nodes:
        G.add_node(node_id)
    
    # Add edges between representative nodes
    for i in range(len(representative_nodes) - 1):
        G.add_edge(representative_nodes[i], representative_nodes[i + 1])
    
    # Filter positions and other attributes
    filtered_pos = {k: v for k, v in positions.items() if k in representative_nodes}
    filtered_colors = [colors[k] for k in representative_nodes]
    filtered_sizes = [sizes[k] for k in representative_nodes]
    filtered_labels = {k: v for k, v in labels.items() if k in representative_nodes}
    
    # Draw the graph
    nx.draw_networkx_nodes(G, filtered_pos, node_color=filtered_colors,
                          node_size=filtered_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, filtered_pos, edge_color='gray',
                          arrows=True, arrowsize=30, arrowstyle='->', 
                          width=2, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, filtered_pos, filtered_labels, 
                           font_size=10, font_weight='bold', ax=ax)
    
    # Add memory usage annotations
    for node_id in representative_nodes:
        node = graph.nodes[node_id]
        x, y = filtered_pos[node_id]
        memory_text = f'{node.memory_usage:.0f} MB'
        ax.text(x, y-0.3, memory_text, ha='center', va='top', 
               fontsize=8, style='italic',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax.set_title('GPT-1.5B Model Graph Structure\n(Simplified View)', fontsize=16, pad=20)
    ax.axis('off')
    
    # Add legend
    legend_elements = [mpatches.Patch(color=color, label=layer_type.value.replace('_', ' ').title()) 
                      for layer_type, color in color_map.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    # Add flow arrows
    ax.annotate('Data Flow', xy=(6, -0.8), xytext=(2, -0.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'),
               fontsize=12, color='red', weight='bold')
    
    plt.tight_layout()
    plt.savefig('simplified_model_graph.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Creating comprehensive GPT model graph visualization...")
    
    # Create detailed visualization
    graph = create_detailed_graph_visualization()
    
    print("Creating simplified graph diagram...")
    
    # Create simplified diagram  
    create_simplified_graph_diagram()
    
    print("Visualizations saved as:")
    print("- comprehensive_model_graph.png")
    print("- simplified_model_graph.png")
