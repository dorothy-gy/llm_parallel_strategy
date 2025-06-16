"""
Test script for GPT Model Graph functionality

This script demonstrates the usage of the graph module for GPT model
representation and analysis.
"""

import os
import sys
import json

# Add the parent directory to path to import graph module
sys.path.append(os.path.dirname(__file__))

from gpt_model_graph import create_gpt_model_graph
from graph_analyzer import GraphAnalyzer


def test_gpt_model_graph():
    """Test GPT model graph creation and basic functionality"""
    print("=== Testing GPT Model Graph Creation ===")
    
    # Create GPT-1.5B model graph
    try:
        graph = create_gpt_model_graph("gpt-1.5b")
        print("✓ Successfully created GPT-1.5B model graph")
        
        # Print basic statistics
        print(f"✓ Total layers: {len(graph.nodes)}")
        print(f"✓ Total parameters: {graph.get_total_parameters():,}")
        print(f"✓ Total memory: {graph.get_total_memory_usage():.2f} MB")
        print(f"✓ Total compute: {graph.get_total_compute_flops():.2f} GFLOPs")
        
        # Test transformer nodes
        transformer_nodes = graph.get_transformer_nodes()
        print(f"✓ Transformer layers: {len(transformer_nodes)}")
        
        # Test execution order
        execution_order = graph.get_execution_order()
        print(f"✓ Execution order length: {len(execution_order)}")
        
        return graph
        
    except Exception as e:
        print(f"✗ Error creating graph: {e}")
        return None


def test_graph_analysis(graph):
    """Test graph analysis functionality"""
    print("\n=== Testing Graph Analysis ===")
    
    try:
        analyzer = GraphAnalyzer(graph)
        print("✓ Successfully created graph analyzer")
        
        # Test basic analysis
        analysis = analyzer.analyze_graph_properties()
        print("✓ Graph properties analysis completed")
        print(f"  - Basic stats keys: {list(analysis['basic_stats'].keys())}")
        print(f"  - Distribution keys: {list(analysis['distribution'].keys())}")
        print(f"  - Bottlenecks keys: {list(analysis['bottlenecks'].keys())}")
        
        return analyzer
        
    except Exception as e:
        print(f"✗ Error in graph analysis: {e}")
        return None


def test_graph_partitioning(analyzer):
    """Test graph partitioning functionality"""
    print("\n=== Testing Graph Partitioning ===")
    
    try:
        # Test uniform partitioning
        num_stages = 4
        uniform_result = analyzer.partition_graph_uniform(num_stages)
        print(f"✓ Uniform partitioning with {num_stages} stages completed")
        print(f"  - Balance score: {uniform_result.balance_score:.4f}")
        print(f"  - Max memory usage: {uniform_result.max_memory_usage:.2f} MB")
        
        # Test balanced partitioning
        balanced_result = analyzer.partition_graph_balanced(num_stages, "memory")
        print(f"✓ Memory-balanced partitioning with {num_stages} stages completed")
        print(f"  - Balance score: {balanced_result.balance_score:.4f}")
        print(f"  - Max memory usage: {balanced_result.max_memory_usage:.2f} MB")
        
        # Test strategy comparison
        strategies = analyzer.compare_partitioning_strategies(num_stages)
        print(f"✓ Strategy comparison completed")
        print(f"  - Available strategies: {list(strategies.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in graph partitioning: {e}")
        return False


def test_optimization(analyzer):
    """Test optimization functionality"""
    print("\n=== Testing Optimization ===")
    
    try:
        # Test stage optimization
        optimization = analyzer.optimize_pipeline_stages(min_stages=2, max_stages=8)
        print("✓ Pipeline stage optimization completed")
        print(f"  - Recommended stages: {optimization['recommended_stages']}")
        print(f"  - Best score: {optimization['best_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in optimization: {e}")
        return False


def test_report_generation(analyzer):
    """Test report generation"""
    print("\n=== Testing Report Generation ===")
    
    try:
        report = analyzer.generate_partitioning_report(num_stages=4)
        print("✓ Partitioning report generated successfully")
        print(f"  - Report length: {len(report)} characters")
        
        # Save report to file
        report_path = os.path.join(os.path.dirname(__file__), "test_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"  - Report saved to: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in report generation: {e}")
        return False


def test_json_export(graph):
    """Test JSON export functionality"""
    print("\n=== Testing JSON Export ===")
    
    try:
        # Export graph to JSON
        json_path = os.path.join(os.path.dirname(__file__), "test_graph.json")
        graph.save_to_json(json_path)
        print(f"✓ Graph exported to JSON: {json_path}")
        
        # Verify JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"  - JSON keys: {list(data.keys())}")
        print(f"  - Number of nodes: {len(data['nodes'])}")
        print(f"  - Number of edges: {len(data['edges'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in JSON export: {e}")
        return False


def main():
    """Main test function"""
    print("Starting GPT Model Graph Tests...")
    print("=" * 50)
    
    # Test 1: Graph creation
    graph = test_gpt_model_graph()
    if not graph:
        print("✗ Graph creation failed, stopping tests")
        return
    
    # Test 2: Graph analysis
    analyzer = test_graph_analysis(graph)
    if not analyzer:
        print("✗ Graph analysis failed, stopping remaining tests")
        return
    
    # Test 3: Graph partitioning
    if not test_graph_partitioning(analyzer):
        print("✗ Graph partitioning failed")
    
    # Test 4: Optimization
    if not test_optimization(analyzer):
        print("✗ Optimization failed")
    
    # Test 5: Report generation
    if not test_report_generation(analyzer):
        print("✗ Report generation failed")
    
    # Test 6: JSON export
    if not test_json_export(graph):
        print("✗ JSON export failed")
    
    print("\n" + "=" * 50)
    print("Tests completed!")
    
    # Print visualization
    print("\n=== Model Visualization ===")
    print(graph.visualize_graph())


if __name__ == "__main__":
    main()
