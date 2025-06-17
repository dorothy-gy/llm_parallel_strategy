#!/usr/bin/env python3

from gpt_model_graph import create_gpt_model_graph, LayerType

def main():
    print("=" * 80)
    print("GPT-1.5B 模型图结构详细描述".center(80))
    print("=" * 80)
    
    graph = create_gpt_model_graph("gpt-1.5b")
    
    print(f"\n🏗️  模型图整体结构:")
    print(f"   📊 总节点数: {len(graph.nodes)} 个")
    print(f"   🔗 总边数: {len(graph.edges)} 条")
    print(f"   📈 图类型: 有向无环图 (DAG)")
    
    print(f"\n🧩 节点类型分布:")
    layer_types = {}
    for node in graph.nodes.values():
        layer_type = node.layer_type.value
        if layer_type not in layer_types:
            layer_types[layer_type] = []
        layer_types[layer_type].append(node)
    
    for layer_type, nodes in layer_types.items():
        print(f"   • {layer_type.replace('_', ' ').title()}: {len(nodes)} 个节点")
    
    print(f"\n📈 具体层结构:")
    print("   Layer ID | Layer Name           | Parameters   | Memory(MB) | Type")
    print("   " + "-" * 70)
    
    # 显示重要层
    important_layers = [0, 1, 2, 47, 48, 49, 50]
    for layer_id in important_layers:
        if layer_id in graph.nodes:
            node = graph.nodes[layer_id]
            print(f"   {layer_id:8d} | {node.name:<20} | {node.parameter_count:>12,} | {node.memory_usage:>8.2f} | {node.layer_type.value}")
        if layer_id == 2:
            print("   " + " " * 8 + "... (中间45个Transformer Blocks) ...")
    
    total_params = graph.get_total_parameters()
    total_memory = graph.get_total_memory_usage()
    
    print(f"\n💾 资源统计:")
    print(f"   • 总参数: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   • 总内存: {total_memory:.2f} MB ({total_memory/1024:.2f} GB)")
    
    print("\n🎨 生成的可视化文件:")
    print("   • comprehensive_model_graph.png - 6子图全面分析")
    print("   • model_structure.png - 基础结构图")
    print("   • partitioning_strategies.png - 分区策略")

if __name__ == "__main__":
    main()
