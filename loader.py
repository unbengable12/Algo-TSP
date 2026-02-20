import numpy as np
import pandas as pd
import ast
from typing import Optional, List

class TSPInstance:
    """表示单个 TSP 问题实例的数据模型"""
    def __init__(self, instance_id, category, num_cities, coords, dist_matrix, total_distance=None, optimal_path=None):
        self.instance_id = instance_id
        self.category = category
        self.num_cities = int(num_cities)
        
        # 数据转换
        self.coords = np.array(coords)
        self.distance_matrix = np.array(dist_matrix)
        
        # 结果与参考值
        self.total_distance = total_distance
        self.optimal_path = optimal_path  # 新增：存储最优路径序列

    def __repr__(self):
        status = "Solved" if self.optimal_path else "Unsolved"
        return f"<TSPInstance {self.instance_id} | Cities: {self.num_cities} | Category: {self.category} | Status: {status}>"

    def get_distance(self, city_a, city_b):
        """获取城市 a 和 b 之间的距离"""
        return self.distance_matrix[city_a][city_b]

    def get_optimal_distance(self) -> Optional[float]:
        """获取已知最优解的距离"""
        if self.optimal_path is not None:
            return self.calculate_path_distance(self.optimal_path)
        return self.total_distance

    def calculate_path_distance(self, path):
        """计算给定路径的总长度"""
        distance = 0
        for i in range(len(path)):
            distance += self.get_distance(path[i], path[(i + 1) % len(path)])
        return distance

class TSPLoader:
    """数据加载组件"""
    
    @staticmethod
    def load_from_csv(file_path):
        """解析 CSV 并返回 TSPInstance 对象列表"""
        df = pd.read_csv(file_path)
        instances = []
        
        for _, row in df.iterrows():
            # 安全解析字符串列表
            coords = ast.literal_eval(row['city_coordinates'])
            dist_matrix = ast.literal_eval(row['distance_matrix'])
            
            # 处理可能缺失的 optimal_path
            optimal_path = None
            if 'optimal_path' in row and pd.notna(row['optimal_path']):
                optimal_path = ast.literal_eval(row['optimal_path'])
            
            instance = TSPInstance(
                instance_id=row['instance_id'],
                category=row['category'],
                num_cities=row['num_cities'],
                coords=coords,
                dist_matrix=dist_matrix,
                total_distance=row['total_distance'],
                optimal_path=optimal_path # 传入新属性
            )
            instances.append(instance)
            
        return instances
class TSPInstanceFactory:
    """工厂类：根据需求过滤或获取特定类型的实例"""
    
    def __init__(self, csv_path):
        self._instances = TSPLoader.load_from_csv(csv_path)

    def get_all(self) -> List[TSPInstance]:
        return self._instances

    def get_by_category(self, category_name) -> List[TSPInstance]:
        """策略：按类型获取（'Small', 'Medium', 'Big', 'Clustered', 'Grid'）"""
        return [inst for inst in self._instances if inst.category == category_name]

    def get_by_categories(self, category_names: List[str]) -> List[TSPInstance]:
        """策略：按多个类型获取（['Small', 'Medium', 'Big','Clustered', 'Grid']）"""
        category_set = set(category_names)
        return [inst for inst in self._instances if inst.category in category_set]

    def get_by_id(self, instance_id) -> Optional[TSPInstance]:
        return next((inst for inst in self._instances if inst.instance_id == instance_id), None)