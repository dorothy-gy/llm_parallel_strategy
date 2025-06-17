"""
GPT模型图结构描述

基于我们构建的GPT-1.5B模型图，以下是图的详细结构说明：
"""

from gpt_model_graph import create_gpt_model_graph, LayerType
from graph_analyzer import GraphAnalyzer

def describe_model_graph():
    """详细描述GPT模型图的结构"""
    
    print("=" * 80)
    print("GPT-1.5B 模型图结构详细描述".center(80))
    print("=" * 80)
    
    # 创建模型图
    graph = create_gpt_model_graph("gpt-1.5b")
    analyzer = GraphAnalyzer(graph)
    
    print("\n🏗️  模型图整体结构:")
    print(f"   📊 总节点数: {len(graph.nodes)} 个")
    print(f"   🔗 总边数: {len(graph.edges)} 条")
    print(f"   📈 图类型: 有向无环图 (DAG)")
    print(f"   🔄 执行顺序: 严格线性 (Layer 0 → Layer 1 → ... → Layer 50)")
    
    print("\n🧩 节点类型分布:")
    layer_types = {}
    for node in graph.nodes.values():
        layer_type = node.layer_type.value
        if layer_type not in layer_types:
            layer_types[layer_type] = []
        layer_types[layer_type].append(node)
    
    for layer_type, nodes in layer_types.items():
        print(f"   • {layer_type.replace('_', ' ').title()}: {len(nodes)} 个节点")
        if layer_type == 'transformer_block':
            print(f"     └─ 编号: Layer 1 - Layer 48")
        elif layer_type == 'embedding':
            print(f"     └─ 编号: Layer 0 (Word Embeddings)")
        elif layer_type == 'layer_norm':
            print(f"     └─ 编号: Layer 49 (Final Layer Norm)")
        elif layer_type == 'output_head':
            print(f"     └─ 编号: Layer 50 (LM Head)")
    
    print("\n📊 节点属性详解:")
    print("   每个节点包含以下关键信息:")
    print("   • layer_id: 唯一标识符 (0-50)")
    print("   • layer_type: 层类型 (embedding/transformer_block/layer_norm/output_head)")
    print("   • parameter_count: 参数数量")
    print("   • memory_usage: 内存使用量 (MB)")
    print("   • compute_flops: 计算量 (GFLOPs)")
    print("   • tensor_parallel_dim: 张量并行维度提示")
    
    print("\n🔗 边连接信息:")
    print("   边表示数据流和层间依赖:")
    print("   • 每条边连接相邻的两层")
    print("   • 边权重表示数据传输成本")
    print("   • 数据流向: Layer i → Layer i+1")
    
    print("\n🎯 核心设计理念:")
    print("   1. 层级抽象: 每个Transformer层作为独立节点")
    print("   2. 性能建模: 节点包含详细的性能特征")
    print("   3. 并行友好: 支持多种并行策略的优化")
    print("   4. 可扩展性: 易于扩展到其他模型架构")
    
    print("\n📈 具体层结构展示:")
    print("   " + "─" * 70)
    
    # 展示前几层和后几层的详细信息
    important_layers = [0, 1, 2, 47, 48, 49, 50]
    
    for layer_id in important_layers:
        if layer_id in graph.nodes:
            node = graph.nodes[layer_id]
            print(f"   Layer {layer_id:2d}: {node.name:<20} | "
                  f"{node.parameter_count:>12,} params | "
                  f"{node.memory_usage:>8.2f} MB | "
                  f"{node.layer_type.value}")
        
        if layer_id == 2:
            print("   " + " " * 8 + "... (中间省略 45 个 Transformer Blocks) ...")
    
    print("   " + "─" * 70)
    
    print(f"\n💾 资源统计:")
    total_params = graph.get_total_parameters()
    total_memory = graph.get_total_memory_usage()
    total_compute = graph.get_total_compute_flops()
    
    print(f"   • 总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   • 总内存: {total_memory:.2f} MB ({total_memory/1024:.2f} GB)")
    print(f"   • 总计算量: {total_compute:.2f} GFLOPs")
    
    print(f"\n🔄 数据流分析:")
    print(f"   • 输入维度: (batch_size, {graph.seq_length}, {graph.hidden_size})")
    print(f"   • 中间表示: 每层输出 (batch_size, {graph.seq_length}, {graph.hidden_size})")
    print(f"   • 最终输出: (batch_size, {graph.seq_length}, {graph.vocab_size})")
    
    # 流水线分区示例
    print(f"\n⚡ 流水线分区示例 (4阶段):")
    result = analyzer.partition_graph_balanced(4, "memory")
    
    for i, partition in enumerate(result.partitions):
        layer_range = f"Layer {partition.start_layer}-{partition.end_layer}"
        memory = f"{partition.total_memory_mb:.1f} MB"
        params = f"{partition.total_parameters/1e6:.1f}M"
        print(f"   Stage {i}: {layer_range:<15} | {memory:<10} | {params:<8} params")
    
    print(f"   平衡分数: {result.balance_score:.4f} (越小越均衡)")
    
    print(f"\n🎨 可视化文件:")
    print(f"   • comprehensive_model_graph.png - 全面的6子图分析")
    print(f"   • simplified_model_graph.png - 简化的流程图")
    print(f"   • model_structure.png - 基础结构图")
    print(f"   • partitioning_strategies.png - 分区策略对比")
    print(f"   • stage_optimization.png - 阶段优化分析")
    
    print("\n" + "=" * 80)
    print("模型图为后续的图匹配算法和异构集群优化提供了完整的基础！")
    print("=" * 80)


def show_graph_json_structure():
    """展示JSON格式的图数据结构"""
    
    print("\n" + "=" * 60)
    print("JSON 数据结构示例".center(60))
    print("=" * 60)
    
    # 读取生成的JSON文件
    import json
    
    try:
        with open('gpt_1.5b_model_graph.json', 'r') as f:
            graph_data = json.load(f)
        
        print("\n📄 JSON文件结构:")
        print("├── config: 模型配置信息")
        print("├── nodes: 所有层节点的详细信息")
        print("├── edges: 层间连接信息")
        print("└── stats: 整体统计信息")
        
        print(f"\n📊 数据规模:")
        print(f"   • 节点数量: {len(graph_data['nodes'])}")
        print(f"   • 边数量: {len(graph_data['edges'])}")
        print(f"   • 总参数: {graph_data['stats']['total_parameters']:,}")
        print(f"   • 总内存: {graph_data['stats']['total_memory_mb']:.2f} MB")
        
        print(f"\n📝 节点示例 (Layer 1 - Transformer Block):")
        node_1 = graph_data['nodes']['1']
        for key, value in node_1.items():
            if isinstance(value, float):
                print(f"   • {key}: {value:.2f}")
            else:
                print(f"   • {key}: {value}")
        
        print(f"\n🔗 边示例 (Layer 0 → Layer 1):")
        edge_0 = graph_data['edges'][0]
        for key, value in edge_0.items():
            if isinstance(value, float):
                print(f"   • {key}: {value:.2f}")
            else:
                print(f"   • {key}: {value}")
                
    except FileNotFoundError:
        print("JSON文件未找到，请先运行模型图生成脚本")


if __name__ == "__main__":
    describe_model_graph()
    show_graph_json_structure()
