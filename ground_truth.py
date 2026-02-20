import pandas as pd
import numpy as np
import random

def generate_tsp_data():
    data = []
    instance_counter = 1

    # 配置参数：类型名称, 城市数量列表, 每种生成的样本数
    configs = [
        ("Small", range(10, 21), 2),  # 每个规模生成2个，总计20+
        ("Medium", [50, 75, 100], 7), # 每个规模生成7个，总计21
        ("Large", [200, 500], 10),    # 每个规模生成10个，总计20
    ]

    # 1-3. 生成基础随机实例
    for label, sizes, count_per_size in configs:
        for n in sizes:
            for _ in range(count_per_size):
                coords = np.random.rand(n, 2) * 1000  # 1000x1000 范围
                data.append(create_entry(instance_counter, label, n, coords))
                instance_counter += 1

    # 4. 聚类结构实例 (Clustered)
    for _ in range(20):
        n = 100
        k = 5  # 5个簇中心
        centers = np.random.rand(k, 2) * 1000
        coords = []
        for _ in range(n):
            center = centers[np.random.randint(0, k)]
            point = center + np.random.normal(0, 50, size=2) # 标准差50
            coords.append(point)
        data.append(create_entry(instance_counter, "Clustered", n, np.array(coords)))
        instance_counter += 1

    # 5. 网格结构实例 (Grid)
    grid_sizes = [4, 5, 6, 8, 10] # 例如 10x10=100
    for s in grid_sizes:
        for _ in range(4): # 5种规格各4个 = 20个
            n = s * s
            coords = [[i * 100, j * 100] for i in range(s) for j in range(s)]
            data.append(create_entry(instance_counter, "Grid", n, np.array(coords)))
            instance_counter += 1

    df = pd.DataFrame(data)
    df.to_csv("tsp_instances.csv", index=False)
    print(f"成功生成 {len(df)} 条 TSP 实例数据，并保存至 tsp_instances.csv")

def create_entry(idx, category, n, coords):
    # 计算欧几里得距离矩阵
    dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1))
    
    return {
        "instance_id": f"{category}_{idx}",
        "category": category,
        "num_cities": n,
        "city_coordinates": coords.tolist(),
        "distance_matrix": dist_matrix.tolist(),
        "total_distance": None  # 待求解
    }


generate_tsp_data()

import pandas as pd
import numpy as np
import ast
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- 1. 基础类定义 ---
class TSPInstance:
    def __init__(self, instance_id, category, num_cities, coords, dist_matrix, total_distance=None):
        self.instance_id = instance_id
        self.category = category
        self.num_cities = int(num_cities)
        self.coords = np.array(coords)
        self.distance_matrix = np.array(dist_matrix)
        self.total_distance = total_distance

    def get_distance(self, city_a, city_b):
        return self.distance_matrix[city_a][city_b]

class TSPLoader:
    @staticmethod
    def load_from_csv(file_path):
        df = pd.read_csv(file_path)
        instances = []
        for _, row in df.iterrows():
            coords = ast.literal_eval(row['city_coordinates'])
            dist_matrix = ast.literal_eval(row['distance_matrix'])
            instances.append(TSPInstance(
                row['instance_id'], row['category'], row['num_cities'], 
                coords, dist_matrix, row['total_distance']
            ))
        return instances

class TSPInstanceFactory:
    def __init__(self, csv_path):
        self._instances = TSPLoader.load_from_csv(csv_path)
    def get_all(self):
        return self._instances

# --- 2. 求解器定义 ---
class ORToolsSolver:
    def __init__(self, instance: TSPInstance, time_limit_sec=30):
        self.instance = instance
        self.n = instance.num_cities
        self.time_limit = time_limit_sec

    def solve(self):
        manager = pywrapcp.RoutingIndexManager(self.n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.instance.distance_matrix[from_node][to_node] * 1000)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = self.time_limit

        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            path = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                path.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            
            # 计算距离
            d = 0
            for i in range(len(path)):
                d += self.instance.distance_matrix[path[i]][path[(i + 1) % len(path)]]
            return d, path + [path[0]]
        return None, None

# --- 3. 并行任务封装 ---
def solve_single_instance(instance):
    # 注意：函数内要能访问到 ORToolsSolver
    try:
        t_limit = 5
        if instance.num_cities > 100: t_limit = 20
        if instance.num_cities >= 500: t_limit = 40
        
        solver = ORToolsSolver(instance, time_limit_sec=t_limit)
        dist, path = solver.solve()
        return {"instance_id": instance.instance_id, "total_distance": dist, "optimal_path": str(path), "status": "Success"}
    except Exception as e:
        return {"instance_id": instance.instance_id, "total_distance": None, "optimal_path": None, "status": f"Error"}

# --- 4. 主运行函数 ---
def run_parallel_tsp(csv_input, csv_output):
    factory = TSPInstanceFactory(csv_input)
    instances = factory.get_all()
    
    num_procs = max(1, cpu_count() - 1)
    results = []
    
    # 在 Notebook 环境中，imap 会按需序列化函数和对象
    with Pool(processes=num_procs) as pool:
        for result in tqdm(pool.imap_unordered(solve_single_instance, instances), total=len(instances), desc="求解进度"):
            results.append(result)
    
    df_results = pd.DataFrame(results)
    df_original = pd.read_csv(csv_input)
    if 'total_distance' in df_original.columns:
        df_original.drop(columns=['total_distance'], inplace=True)
    
    final_df = pd.merge(df_original, df_results, on="instance_id", how="left")
    final_df.to_csv(csv_output, index=False)
    print(f"完成！保存至 {csv_output}")

# 启动
run_parallel_tsp("tsp_instances.csv", "tsp_instances_solved.csv")