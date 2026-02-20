from evaluation import TSPSolver
from loader import TSPInstance
import tracemalloc
import time
from typing import Tuple, List, Optional

class NearestNeighborSolver(TSPSolver):
    def __init__(self, start: int = 0):
        super().__init__("NearestNeighbor")
        self.start = start
    
    def solve(self, instance: TSPInstance, time_limit: float = 36000.0) -> Tuple[List[int], float, float]:
        tracemalloc.start()
        start_time = time.perf_counter()
        
        unvisited = set(range(instance.num_cities))
        unvisited.remove(self.start)
        tour = [self.start]
        current = self.start
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: instance.get_distance(current, x))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        elapsed = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return tour, elapsed, peak


class TwoOptSolver(TSPSolver):
    def __init__(self, base_solver: Optional[TSPSolver] = None):
        super().__init__("TwoOpt")
        self.base_solver = base_solver or NearestNeighborSolver()
    
    def solve(self, instance: TSPInstance, time_limit: float = 36000.0) -> Tuple[List[int], float, float]:
        # 先检查tracemalloc状态，确保是全新的跟踪
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        # 开始跟踪TwoOpt自身的内存使用
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # 在调用base_solver时，暂时停止跟踪
        tracemalloc.stop()
        tour, _, _ = self.base_solver.solve(instance, time_limit * 0.3)
        tracemalloc.start()  # 重新开始跟踪TwoOpt的内存
        
        improved = True
        while improved:
            if time.perf_counter() - start_time > time_limit:
                break
            
            improved = False
            n = len(tour)
            for i in range(1, n - 2):
                for j in range(i + 2, n):
                    a, b = tour[i-1], tour[i]
                    c, d = tour[j-1], tour[j % n]
                    
                    old_dist = instance.get_distance(a, b) + instance.get_distance(c, d)
                    new_dist = instance.get_distance(a, c) + instance.get_distance(b, d)
                    
                    if new_dist < old_dist:
                        tour[i:j] = reversed(tour[i:j])
                        improved = True
                        break
                if improved:
                    break
        
        # 先获取内存数据，再停止跟踪
        current, peak = tracemalloc.get_traced_memory()
        elapsed = time.perf_counter() - start_time
        
        # 停止跟踪
        tracemalloc.stop()
        
        return tour, elapsed, peak