import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from loader import TSPInstance, TSPInstanceFactory
from visualization import save_tour_gif

class TSPSolver(ABC):
    """TSP求解器抽象基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def solve(self, instance: TSPInstance, time_limit: float = 60.0) -> Tuple[List[int], float]:
        """
        求解TSP
        返回: (路径, 运行时间)
        """
        pass

class Metric(ABC):
    """评估指标抽象基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, tour: List[int], runtime: float, 
                  instance: TSPInstance) -> float:
        """计算指标值"""
        pass

@dataclass
class EvaluationResult:
    """单次评估结果"""
    solver: str
    instance: str
    category: str
    num_cities: int
    metrics: Dict[str, float]
    tour: List[int]
    
class TSPEvaluator:
    def __init__(self):
        self.metrics: List[Metric] = []
        self.results: List[EvaluationResult] = []
        self._runs_data: Dict[str, List[Dict]] = defaultdict(list)  # 用于稳定性统计

    def add_metric(self, metric: Metric) -> 'TSPEvaluator':
        self.metrics.append(metric)
        return self
    
    def evaluate(self, solver: TSPSolver, instance: TSPInstance,
                 time_limit: float = 36000.0, runs: int = 1,
                 save_gif: bool = True, gif_dir: str = "outputs/gifs") -> List[EvaluationResult]:
        """评估求解器，支持多次运行计算稳定性"""
        print(f"\n评估: {solver.name} on {instance.instance_id} (runs={runs})")
        
        run_results = []
        best_tour = None
        best_tour_len = float("inf")
        
        for run in range(runs):
            tour, runtime, memory = solver.solve(instance, time_limit)
            tour_len = instance.calculate_path_distance(tour)
            
            # 计算所有指标
            metric_values = {}
            for metric in self.metrics:
                value = metric.calculate(tour, runtime, memory, instance)
                metric_values[metric.name] = value
            
            # 存储单次结果
            self._runs_data[f"{solver.name}_{instance.instance_id}"].append(metric_values)

            if tour_len < best_tour_len:
                best_tour_len = tour_len
                best_tour = tour
            
            result = EvaluationResult(
                solver=solver.name,
                instance=instance.instance_id,
                category=instance.category,
                num_cities=instance.num_cities,
                metrics=metric_values,
                tour=tour
            )
            run_results.append(result)
        
        if save_gif and best_tour is not None:
            safe_solver = solver.name.replace(" ", "_")
            safe_instance = instance.instance_id.replace(" ", "_")
            out_path = os.path.join(
                gif_dir,
                safe_solver,
                f"{safe_instance}_best_len{best_tour_len:.2f}.gif"
            )
            save_tour_gif(instance, best_tour, out_path)

        # 打印结果
        self._print_result(run_results, runs)
        self.results.extend(run_results)
        
        return run_results
    
    def _print_result(self, run_results: List[EvaluationResult], runs: int):
        """打印评估结果"""
        if runs == 1:
            r = run_results[0]
            print(f"  结果: ", end="")
            for name, value in r.metrics.items():
                if "Rate" in name or "Gap" in name:
                    print(f"{name}={value:.2f}% ", end="")
                elif "Memory" in name:
                    print(f"{name}={value:.2f}B ", end="")
                elif "Runtime" in name:
                    print(f"{name}={value:.4f}s ", end="")
                else:
                    print(f"{name}={value:.2f} ", end="")
            print()
        else:
            # 多次运行：计算统计值
            print(f"  多次运行统计 (n={runs}):")
            metric_names = list(run_results[0].metrics.keys())
            for name in metric_names:
                values = [r.metrics[name] for r in run_results]
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # 4. 稳定性指标：显示平均值±标准差
                unit = "%" if "Rate" in name or "Gap" in name else "B" if "Memory" in name else "s" if "Runtime" in name else ""
                print(f"    {name}: {mean_val:.4f}±{std_val:.4f}{unit}")
    
    def compare(self, solvers: List[TSPSolver], instance: TSPInstance,
                runs: int = 1) -> Dict[str, List[EvaluationResult]]:
        """对比多个求解器"""
        print(f"\n{'='*60}")
        print(f"对比求解器 on {instance.instance_id} ({instance.category}, {instance.num_cities} cities)")
        print(f"{'='*60}")
        
        comparison = {}
        for solver in solvers:
            results = self.evaluate(solver, instance, runs=runs)
            comparison[solver.name] = results
        
        return comparison
    
    def benchmark(self, solver: TSPSolver, factory: TSPInstanceFactory,
                  categories: Optional[List[str]] = None,
                  runs: int = 1) -> pd.DataFrame:
        """批量基准测试，支持5.不同规模 和 6.不同结构的影响分析"""
        print(f"\n{'='*60}")
        print(f"批量基准测试: {solver.name}")
        print(f"{'='*60}")
        
        instances = factory.get_all()
        if categories:
            instances = [inst for inst in instances if inst.category in categories]
        
        records = []
        for inst in instances:
            try:
                results = self.evaluate(solver, inst, runs=runs)
                # 取平均（如果多次运行）
                avg_metrics = {}
                for name in results[0].metrics.keys():
                    values = [r.metrics[name] for r in results]
                    avg_metrics[name] = np.mean(values)
                    avg_metrics[f"{name}_std"] = np.std(values)
                
                records.append({
                    'instance_id': inst.instance_id,
                    'category': inst.category,
                    'num_cities': inst.num_cities,
                    **avg_metrics
                })
            except Exception as e:
                print(f"  错误: {e}")
        
        df = pd.DataFrame(records)
        
        # 分析不同规模和结构的影响
        self._analyze_impact(df)
        
        return df
    
    def _analyze_impact(self, df: pd.DataFrame):
        """分析规模和结构影响"""
        print(f"\n{'='*60}")
        print("影响分析")
        print(f"{'='*60}")
        
        # 5. 不同输入规模的影响
        print("\n【规模影响】按城市数量分组:")
        size_groups = df.groupby(pd.cut(df['num_cities'], bins=[0, 30, 100, 1000, 10000], 
                                       labels=['Small(<30)', 'Medium(30-100)', 
                                              'Large(100-1000)', 'XLarge(>1000)']))
        for name, group in size_groups:
            if not group.empty:
                gap_col = [c for c in group.columns if 'OptimalityGap' in c and '_std' not in c][0]
                time_col = [c for c in group.columns if 'Runtime' in c and '_std' not in c][0]
                print(f"  {name}: n={len(group)}, "
                      f"AvgGap={group[gap_col].mean():.2f}%, "
                      f"AvgTime={group[time_col].mean():.4f}s")
        
        # 6. 不同实例结构的影响
        print("\n【结构影响】按类别分组:")
        cat_groups = df.groupby('category')
        for name, group in cat_groups:
            gap_col = [c for c in group.columns if 'OptimalityGap' in c and '_std' not in c][0]
            time_col = [c for c in group.columns if 'Runtime' in c and '_std' not in c][0]
            print(f"  {name}: n={len(group)}, "
                  f"AvgGap={group[gap_col].mean():.2f}%, "
                  f"AvgTime={group[time_col].mean():.4f}s")
    
    def export_results(self, filename: str = "evaluation_results.csv"):
        """导出结果到CSV"""
        if not self.results:
            print("无结果可导出")
            return
        
        data = []
        for r in self.results:
            row = {
                'solver': r.solver,
                'instance': r.instance,
                'category': r.category,
                'num_cities': r.num_cities,
                **r.metrics
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"\n结果已导出: {filename}")