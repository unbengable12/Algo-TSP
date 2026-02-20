from evaluation import *
from loader import *
from metric import *
from solver import *
from visualization import generate_analysis

def spliter(content: str):
    print("\n" + "="*60)
    print(content)
    print("="*60)

def test():
    factory = TSPInstanceFactory("tsp_instances_solved.csv")
    # 加载特定类别的实例进行测试
    spliter("步骤1: 加载已求解的TSP实例")
    instances: List[TSPInstance] = factory.get_by_categories(["Small", "Clustered", "Grid"])  # 只加载Medium、Clustered和Grid类别的实例
    print(f"共加载 {len(instances)} 个实例")
    # 显示规模分布
    sizes = {}
    for inst in instances:
        n = inst.num_cities
        sizes[n] = sizes.get(n, 0) + 1
    print("规模分布:")
    for n in sorted(sizes.keys()):
        print(f"  {n} cities: {sizes[n]}个")

    # 准备求解器
    spliter("步骤2: 准备求解器")
    solvers: List[TSPSolver] = [
        NearestNeighborSolver(start=0),
        TwoOptSolver(NearestNeighborSolver(0))
    ]
    for s in solvers:
        print(f"\t- {s.name}")
    
    # 创建评估器并添加指标
    spliter("步骤3: 配置评估指标")
    evaluator = TSPEvaluator()
    evaluator.add_metric(TourLengthMetric()) \
             .add_metric(OptimalityGapMetric()) \
             .add_metric(EdgeMatchRateMetric()) \
             .add_metric(RuntimeMetric()) \
             .add_metric(MemoryMetric())
    print("已添加指标:")
    for m in evaluator.metrics:
        print(f"\t- {m.name}")
    
    # 单实例测试 (带稳定性分析,runs=5)
    spliter("步骤4: 单实例测试 (带稳定性分析, run=5)")
    test_instance = instances[0]
    print(f"测试实例: {test_instance}")
    print(f"已知最优解: {test_instance.get_optimal_distance():.2f}")
    evaluator.compare(solvers, test_instance, runs=5)
    
    # 批量基准测试
    spliter("步骤5: 批量基准测试")
    print(f"共评估 {len(instances)} 个实例")
    for solver in solvers:
        print(f"\n--- {solver.name} ---")
        for inst in instances:
            evaluator.evaluate(solver, inst, runs=5)

    # 导出结果
    spliter("步骤6: 导出结果")
    evaluator.export_results("evaluation_results_small.csv")

    # 可视化与分析输出
    spliter("步骤6.1: 可视化与统计分析输出")
    generate_analysis(evaluator.results, output_dir="outputs")
    
    # 简单统计分析
    spliter("步骤7: 统计分析")
    df = pd.DataFrame([
        {
            'solver': r.solver,
            'instance': r.instance,
            'num_cities': r.num_cities,
            **r.metrics
        } for r in evaluator.results
    ])
    print("\n按求解器分组统计:")
    for solver_name in df['solver'].unique():
        solver_df = df[df['solver'] == solver_name]
        print(f"\n{solver_name}:")
        print(f"\tCount: {len(solver_df)}")
        print(f"\tAvg OptimalityGap: {solver_df['OptimalityGap(%)'].mean():.2f}%")
        print(f"\tAvg Runtime: {solver_df['Runtime(s)'].mean():.4f}s")
        print(f"\tAvg Memory: {solver_df['Memory(B)'].mean():.2f}B")
        
if __name__ == "__main__":
    test()