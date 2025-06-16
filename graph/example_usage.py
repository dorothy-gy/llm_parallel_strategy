"""
Example usage of GPT Model Graph for Pipeline Parallel Strategy Generation

This script demonstrates how to use the graph module to:
1. Create a GPT model graph representation
2. Analyze the model structure 
3. Generate pipeline parallel partitioning strategies
4. Export results for further analysis
"""

import os
import json
from gpt_model_graph import create_gpt_model_graph
from graph_analyzer import GraphAnalyzer


def demonstrate_gpt_graph_usage():
    """Demonstrate the complete workflow of GPT model graph analysis"""
    
    print("=== GPT Model Graph for Pipeline Parallel Strategy ===\n")
    
    # Step 1: Create GPT model graph
    print("1. Creating GPT-1.5B model graph...")
    model_graph = create_gpt_model_graph("gpt-1.5b")
    
    print(f"   ✓ Model created with {len(model_graph.nodes)} layers")
    print(f"   ✓ Total parameters: {model_graph.get_total_parameters():,}")
    print(f"   ✓ Total memory usage: {model_graph.get_total_memory_usage():.2f} MB")
    print(f"   ✓ Transformer layers: {len(model_graph.get_transformer_nodes())}")
    
    # Step 2: Initialize analyzer
    print("\n2. Initializing graph analyzer...")
    analyzer = GraphAnalyzer(model_graph)
    print("   ✓ Analyzer initialized")
    
    # Step 3: Analyze model properties
    print("\n3. Analyzing model properties...")
    properties = analyzer.analyze_graph_properties()
    
    print("   Model Distribution:")
    for layer_type, memory in properties['distribution']['memory_by_type'].items():
        print(f"     - {layer_type}: {memory:.2f} MB")
    
    # Step 4: Find optimal pipeline stages
    print("\n4. Optimizing pipeline stages...")
    optimization = analyzer.optimize_pipeline_stages(min_stages=2, max_stages=12)
    recommended_stages = optimization['recommended_stages']
    
    print(f"   ✓ Recommended pipeline stages: {recommended_stages}")
    print(f"   ✓ Optimization score: {optimization['best_score']:.4f}")
    
    # Step 5: Generate partitioning strategies
    print(f"\n5. Generating partitioning strategies for {recommended_stages} stages...")
    strategies = analyzer.compare_partitioning_strategies(recommended_stages)
    
    best_strategy = None
    best_score = float('inf')
    
    for strategy_name, strategy_data in strategies.items():
        if 'error' not in strategy_data:
            score = strategy_data['balance_score']
            print(f"   - {strategy_name}: Balance Score = {score:.4f}")
            if score < best_score:
                best_score = score
                best_strategy = strategy_name
    
    print(f"   ✓ Best strategy: {best_strategy}")
    
    # Step 6: Generate detailed report
    print(f"\n6. Generating detailed analysis report...")
    report = analyzer.generate_partitioning_report(recommended_stages)
    
    # Save report
    report_path = "pipeline_strategy_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"   ✓ Report saved to: {report_path}")
    
    # Step 7: Export graph data
    print("\n7. Exporting graph data...")
    
    # Export full graph
    model_graph.save_to_json("gpt_1.5b_model_graph.json")
    print("   ✓ Model graph exported to: gpt_1.5b_model_graph.json")
    
    # Export partitioning result
    if best_strategy and best_strategy in strategies:
        partition_data = strategies[best_strategy]
        with open("best_partition_strategy.json", 'w') as f:
            json.dump(partition_data, f, indent=2)
        print("   ✓ Best partition strategy exported to: best_partition_strategy.json")
    
    # Step 8: Generate pipeline parallel configuration
    print("\n8. Generating pipeline parallel configuration...")
    
    if best_strategy and best_strategy in strategies:
        partition_result = strategies[best_strategy]
        
        pipeline_config = {
            "model_name": "gpt-1.5b",
            "total_stages": partition_result['total_stages'],
            "strategy": best_strategy,
            "balance_score": partition_result['balance_score'],
            "max_memory_per_stage": partition_result['max_memory_usage'],
            "stages": []
        }
        
        for partition in partition_result['partitions']:
            stage_config = {
                "stage_id": partition['partition_id'],
                "layer_range": [partition['start_layer'], partition['end_layer']],
                "memory_mb": partition['total_memory_mb'],
                "parameters": partition['total_parameters'],
                "compute_gflops": partition['total_compute_gflops']
            }
            pipeline_config['stages'].append(stage_config)
        
        with open("pipeline_parallel_config.json", 'w') as f:
            json.dump(pipeline_config, f, indent=2)
        print("   ✓ Pipeline parallel config exported to: pipeline_parallel_config.json")
    
    print("\n=== Analysis Complete ===")
    print(f"Generated pipeline parallel strategy for GPT-1.5B:")
    print(f"  - Recommended stages: {recommended_stages}")
    print(f"  - Best strategy: {best_strategy}")
    print(f"  - Balance score: {best_score:.4f}")
    
    return model_graph, analyzer, pipeline_config


def analyze_different_stage_counts():
    """Analyze performance with different numbers of pipeline stages"""
    
    print("\n=== Pipeline Stage Count Analysis ===\n")
    
    model_graph = create_gpt_model_graph("gpt-1.5b")
    analyzer = GraphAnalyzer(model_graph)
    
    # Test different stage counts
    stage_counts = [2, 4, 6, 8, 12, 16]
    results = []
    
    print("Stage Count | Balance Score | Max Memory (MB) | Comm Cost (MB)")
    print("-" * 65)
    
    for stages in stage_counts:
        try:
            result = analyzer.partition_graph_balanced(stages, "memory")
            results.append({
                'stages': stages,
                'balance_score': result.balance_score,
                'max_memory': result.max_memory_usage,
                'comm_cost': result.total_communication_cost
            })
            
            print(f"{stages:11d} | {result.balance_score:13.4f} | {result.max_memory_usage:14.2f} | {result.total_communication_cost:13.2f}")
            
        except Exception as e:
            print(f"{stages:11d} | ERROR: {str(e)}")
    
    # Find optimal stage count
    if results:
        # Score = balance_score + 0.1 * communication_cost
        scored_results = [(r['balance_score'] + 0.1 * r['comm_cost'], r) for r in results]
        best_score, best_result = min(scored_results)
        
        print(f"\nOptimal configuration:")
        print(f"  - Stages: {best_result['stages']}")
        print(f"  - Balance Score: {best_result['balance_score']:.4f}")
        print(f"  - Max Memory: {best_result['max_memory']:.2f} MB")
        print(f"  - Communication Cost: {best_result['comm_cost']:.2f} MB")


if __name__ == "__main__":
    # Main demonstration
    model_graph, analyzer, config = demonstrate_gpt_graph_usage()
    
    # Additional analysis
    analyze_different_stage_counts()
    
    print(f"\n=== Files Generated ===")
    files = [
        "pipeline_strategy_report.txt",
        "gpt_1.5b_model_graph.json", 
        "best_partition_strategy.json",
        "pipeline_parallel_config.json"
    ]
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  ✓ {file} ({size:.1f} KB)")
        else:
            print(f"  ✗ {file} (not found)")
