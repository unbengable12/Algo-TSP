## 项目说明

本项目用于`TSP`算法评测、对比与可视化分析，包含数据加载、求解器、评估指标、统计分析与图表输出。

## 快速开始

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirement.txt
python merge.py # 因为 github 文件大小限制, 需要分片上传, 使用 merge.py 合成一个csv文件
python test.py
```

运行后会：
- 输出控制台评估日志
- 生成结果`CSV`
- 生成可视化图表与`GIF`/`PNG`到`outputs`目录

## 目录结构

- `loader.py`：数据加载与实例工厂
- `solver.py`：算法实现
- `metric.py`：评估指标
- `evaluation.py`：评测逻辑
- `visualization.py`：图表与GIF/PNG输出
- `test.py`：测试入口
- `outputs/`：输出目录
- `ground_truth.py`: 测试数据生成

## 使用测试数据

使用工厂实例获取对应的数据：

```python
from loader import TSPInstanceFactory

# 初始化工厂
factory = TSPInstanceFactory("tsp_instances_solved.csv")

# 获取所有实例
all_instances = factory.get_all()

# 按单个类别获取
small_instances = factory.get_by_category("Small")
medium_instances = factory.get_by_category("Medium")
large_instances = factory.get_by_category("Large")
clustered_instances = factory.get_by_category("Clustered")
grid_instances = factory.get_by_category("Grid")

# 按多个类别获取
instances = factory.get_by_categories(["Medium", "Clustered", "Grid"])

# 按实例ID获取
instance = factory.get_by_id("instance_001")
```

### 评测流程

见`test.py`

## 拓展

### 1. 新增求解算法

在 `solver.py` 中继承 `TSPSolver` 并实现 `solve()`:
- 输入：`TSPInstance`, `time_limit`
- 输出：`tour`, `runtime`, `memory`

`solve()`中用`tracemalloc`计算内存


然后在`test.py`的`solvers`列表中注册。

### 2. 新增评估指标

在`metric.py`中继承`Metric`并实现`calculate()`。
再在`test.py`中为`evaluator.add_metric(...)`注册即可。

### 3. 扩展可视化

在`visualization.py`中新增绘图函数, 并在`generate_analysis()`中调用。
输出会自动写入`outputs/`。

### 4. Ground Truth
使用`random`生成输入数据，再使用`Google Ortools`求解.
```bash
pip install ortools
```

## 评估指标

- [x] **运行时间** - `outputs/evaluation_results_all.csv`
- [x] **最优差距** - `outputs/evaluation_results_all.csv` + `outputs/gap_bar.png` + `outputs/gap_heatmap_category.png`
- [x] **边匹配率** - `outputs/evaluation_results_all.csv`
- [x] **路径长度** - `outputs/evaluation_results_all.csv`
- [x] **稳定性** (均值、标准差) - `outputs/gap_boxplot.png`
- [ ] **方差**
- [x] **内存消耗** - `outputs/evaluation_results_all.csv`
- [x] **可扩展性分析** (按城市数, 按不同实例结构) - `outputs/runtime_heatmap_size.png` + `outputs/gap_heatmap_category.png`
- [x] **理论与实验一致性** (log-log 拟合分析时间增长趋势, 计算经验指数, 对比理论复杂度) - `outputs/theory_empirical_loglog.png` + `outputs/theory_empirical_summary.csv`
- [x] **显著性统计分析** - `outputs/significance_tests.csv` + `outputs/gap_boxplot.png`
- [x] **算法性能预测模型** - `outputs/runtime_prediction.png` + `outputs/runtime_prediction_model.csv`
- [x] **优势区间分析** - `outputs/advantage_region.png` + `outputs/advantage_region_table.csv`
- [x] **路径可视化** - `outputs/gifs/{solver}/{instance}_best_len{X}.gif` + `.png` 
- [ ] **收敛速度**