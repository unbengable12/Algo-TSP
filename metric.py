from evaluation import Metric

class OptimalityGapMetric(Metric):
    """最优差距（%）"""
    def __init__(self):
        super().__init__("OptimalityGap(%)")
    
    def calculate(self, tour, runtime, memory, instance):
        opt = instance.get_optimal_distance()
        if opt is None or opt == 0:
            return 0.0
        tour_len = instance.calculate_path_distance(tour)
        return ((tour_len - opt) / opt) * 100


class EdgeMatchRateMetric(Metric):
    """精确率/边匹配率（%）"""
    def __init__(self):
        super().__init__("EdgeMatchRate(%)")
    
    def calculate(self, tour, runtime, memory, instance):
        if not instance.optimal_path:
            return 0.0
        
        # 将路径转换为边集合
        def get_edges(path):
            edges = set()
            n = len(path)
            for i in range(n):
                a, b = path[i], path[(i+1) % n]
                # 无向边：用排序后的元组
                edges.add(tuple(sorted([a, b])))
            return edges
        
        tour_edges = get_edges(tour)
        optimal_edges = get_edges(instance.optimal_path)
        
        if not optimal_edges:
            return 0.0
        
        common = tour_edges & optimal_edges
        return (len(common) / len(optimal_edges)) * 100


class RuntimeMetric(Metric):
    """运行时间（秒）"""
    def __init__(self):
        super().__init__("Runtime(s)")
    
    def calculate(self, tour, runtime, memory, instance):
        return runtime


class MemoryMetric(Metric):
    """内存消耗（B）"""
    def __init__(self):
        super().__init__("Memory(B)")
    
    def calculate(self, tour, runtime, memory, instance):
        return memory


class TourLengthMetric(Metric):
    """路径长度"""
    def __init__(self):
        super().__init__("TourLength")
    
    def calculate(self, tour, runtime, memory, instance):
        return instance.calculate_path_distance(tour)