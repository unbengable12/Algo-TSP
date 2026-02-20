import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

try:
    import imageio.v2 as imageio
    _HAS_IMAGEIO = True
except Exception:
    try:
        import imageio
        _HAS_IMAGEIO = True
    except Exception:
        _HAS_IMAGEIO = False

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _ensure_dir(path: str) -> None:
    """确保目录存在，不存在则创建。"""
    os.makedirs(path, exist_ok=True)


def build_dataframe(results) -> pd.DataFrame:
    """将评测结果列表转换为DataFrame。"""
    rows = []
    for r in results:
        rows.append({
            "solver": r.solver,
            "instance": r.instance,
            "category": r.category,
            "num_cities": r.num_cities,
            **r.metrics
        })
    return pd.DataFrame(rows)


def _size_bins(df: pd.DataFrame) -> pd.DataFrame:
    """为规模分析添加城市数量分箱列。"""
    df = df.copy()
    df["size_bin"] = pd.cut(
        df["num_cities"],
        bins=[0, 30, 100, 300, 1000, 10000],
        labels=["<30", "30-100", "100-300", "300-1000", ">1000"],
        include_lowest=True
    )
    return df


def _plot_line_runtime(df: pd.DataFrame, out_dir: str) -> None:
    """绘制不同规模下的运行时间折线图。"""
    metric = "Runtime(s)"
    grp = df.groupby(["solver", "num_cities"])[metric].mean().reset_index()

    plt.figure(figsize=(8, 5))
    for solver, g in grp.groupby("solver"):
        plt.plot(g["num_cities"], g[metric], marker="o", label=solver)
    plt.xlabel("num_cities")
    plt.ylabel(metric)
    plt.title("Runtime vs num_cities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_line.png"), dpi=200)
    plt.close()


def _plot_bar_gap(df: pd.DataFrame, out_dir: str) -> None:
    """绘制平均最优差距柱状图。"""
    metric = "OptimalityGap(%)"
    agg = df.groupby("solver")[metric].mean().reset_index()
    plt.figure(figsize=(6, 4))
    plt.bar(agg["solver"], agg[metric])
    plt.xlabel("solver")
    plt.ylabel(metric)
    plt.title("Average Optimality Gap by Solver")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gap_bar.png"), dpi=200)
    plt.close()


def _plot_heatmap_category(df: pd.DataFrame, out_dir: str) -> None:
    """绘制类别-求解器的最优差距热图。"""
    metric = "OptimalityGap(%)"
    pivot = df.pivot_table(index="category", columns="solver", values=metric, aggfunc="mean")
    plt.figure(figsize=(7, 4))
    if _HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    else:
        plt.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.colorbar(label=metric)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Avg Gap by Category vs Solver")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gap_heatmap_category.png"), dpi=200)
    plt.close()


def _plot_heatmap_size(df: pd.DataFrame, out_dir: str) -> None:
    """绘制规模分箱-求解器的运行时间热图。"""
    metric = "Runtime(s)"
    dfb = _size_bins(df)
    pivot = dfb.pivot_table(index="size_bin", columns="solver", values=metric, aggfunc="mean", observed=False)
    plt.figure(figsize=(7, 4))
    if _HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="magma")
    else:
        plt.imshow(pivot.values, aspect="auto", cmap="magma")
        plt.colorbar(label=metric)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Avg Runtime by Size Bin vs Solver")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_heatmap_size.png"), dpi=200)
    plt.close()


def _advantage_region(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """计算优势区间并输出图表与汇总表。"""
    dfb = _size_bins(df)
    metric_gap = "OptimalityGap(%)"
    metric_time = "Runtime(s)"
    agg = dfb.groupby(["size_bin", "solver"], observed=False)[[metric_gap, metric_time]].mean().reset_index()

    winners = []
    for size_bin, g in agg.groupby("size_bin", observed=False):
        g_sorted = g.sort_values([metric_gap, metric_time])
        best = g_sorted.iloc[0]
        winners.append({
            "size_bin": str(size_bin),
            "best_solver": best["solver"],
            "avg_gap": best[metric_gap],
            "avg_runtime": best[metric_time]
        })

    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(os.path.join(out_dir, "advantage_region_table.csv"), index=False)

    plt.figure(figsize=(7, 3))
    colors = {s: c for s, c in zip(winners_df["best_solver"].unique(), plt.cm.tab10.colors)}
    plt.bar(winners_df["size_bin"], [1] * len(winners_df), color=[colors[s] for s in winners_df["best_solver"]])
    for idx, row in winners_df.iterrows():
        plt.text(idx, 0.5, row["best_solver"], ha="center", va="center", color="white", fontsize=9)
    plt.ylim(0, 1)
    plt.yticks([])
    plt.xlabel("size_bin")
    plt.title("Advantage Region by Size Bin")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "advantage_region.png"), dpi=200)
    plt.close()

    return winners_df


def _theory_vs_empirical(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """对比理论与经验复杂度趋势（log-log拟合）。"""
    metric = "Runtime(s)"
    records = []
    plt.figure(figsize=(7, 5))

    for solver, g in df.groupby("solver"):
        g = g.groupby("num_cities")[metric].mean().reset_index()
        x = g["num_cities"].values.astype(float)
        y = g[metric].values.astype(float)
        mask = (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            continue
        logx = np.log(x)
        logy = np.log(y)
        slope, intercept = np.polyfit(logx, logy, 1)
        records.append({"solver": solver, "empirical_exponent": slope})

        plt.scatter(logx, logy, label=f"{solver} data", alpha=0.7)
        plt.plot(logx, intercept + slope * logx, label=f"{solver} fit (b={slope:.2f})")

    plt.xlabel("log(num_cities)")
    plt.ylabel("log(Runtime)")
    plt.title("Theory vs Empirical (log-log fit)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "theory_empirical_loglog.png"), dpi=200)
    plt.close()

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(os.path.join(out_dir, "theory_empirical_summary.csv"), index=False)
    return summary_df


def _significance_analysis(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """显著性分析与箱线图输出。"""
    metric = "OptimalityGap(%)"
    pivot = df.pivot_table(index="instance", columns="solver", values=metric, aggfunc="mean")
    solvers = list(pivot.columns)

    records = []
    for i in range(len(solvers)):
        for j in range(i + 1, len(solvers)):
            a = solvers[i]
            b = solvers[j]
            paired = pivot[[a, b]].dropna()
            if paired.empty:
                continue
            diff = paired[a] - paired[b]

            if _HAS_SCIPY:
                t_stat, t_p = stats.ttest_rel(paired[a], paired[b])
                try:
                    w_stat, w_p = stats.wilcoxon(paired[a], paired[b])
                except ValueError:
                    w_stat, w_p = np.nan, np.nan
            else:
                t_stat = np.nan
                t_p = np.nan
                w_stat = np.nan
                w_p = np.nan

            ci_low, ci_high = np.percentile(diff, [2.5, 97.5]) if len(diff) >= 5 else (np.nan, np.nan)

            records.append({
                "metric": metric,
                "solver_a": a,
                "solver_b": b,
                "mean_diff(a-b)": float(diff.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "ttest_p": t_p,
                "wilcoxon_p": w_p
            })

    result_df = pd.DataFrame(records)
    result_df.to_csv(os.path.join(out_dir, "significance_tests.csv"), index=False)

    plt.figure(figsize=(7, 4))
    if _HAS_SEABORN:
        sns.boxplot(data=df, x="solver", y=metric)
    else:
        df.boxplot(column=metric, by="solver")
        plt.suptitle("")
    plt.title("Optimality Gap Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gap_boxplot.png"), dpi=200)
    plt.close()

    return result_df


def _prediction_model(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """构建经验拟合模型并输出预测曲线。"""
    metric = "Runtime(s)"
    records = []
    plt.figure(figsize=(7, 5))

    for solver, g in df.groupby("solver"):
        g = g.groupby("num_cities")[metric].mean().reset_index()
        x = g["num_cities"].values.astype(float)
        y = g[metric].values.astype(float)
        if len(x) < 2:
            continue

        logx = np.log(x)
        logy = np.log(y)
        slope, intercept = np.polyfit(logx, logy, 1)

        x_pred = np.linspace(x.min(), x.max() * 1.5, 50)
        y_pred = np.exp(intercept + slope * np.log(x_pred))

        plt.scatter(x, y, label=f"{solver} data")
        plt.plot(x_pred, y_pred, label=f"{solver} pred")

        records.append({
            "solver": solver,
            "model": "runtime = a * n^b",
            "a": float(np.exp(intercept)),
            "b": float(slope)
        })

    plt.xlabel("num_cities")
    plt.ylabel(metric)
    plt.title("Runtime Prediction Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_prediction.png"), dpi=200)
    plt.close()

    model_df = pd.DataFrame(records)
    model_df.to_csv(os.path.join(out_dir, "runtime_prediction_model.csv"), index=False)
    return model_df


def generate_analysis(results, output_dir: str = "outputs") -> Dict[str, pd.DataFrame]:
    """生成完整分析与可视化输出，并返回关键DataFrame。"""
    _ensure_dir(output_dir)
    df = build_dataframe(results)
    df.to_csv(os.path.join(output_dir, "evaluation_results_all.csv"), index=False)

    _plot_line_runtime(df, output_dir)
    _plot_bar_gap(df, output_dir)
    _plot_heatmap_category(df, output_dir)
    _plot_heatmap_size(df, output_dir)
    winners_df = _advantage_region(df, output_dir)
    theory_df = _theory_vs_empirical(df, output_dir)
    significance_df = _significance_analysis(df, output_dir)
    prediction_df = _prediction_model(df, output_dir)

    return {
        "df": df,
        "advantage": winners_df,
        "theory": theory_df,
        "significance": significance_df,
        "prediction": prediction_df
    }


def save_tour_gif(instance, tour, out_path: str, max_frames: int = 60) -> bool:
    """保存完整路线PNG并生成逐步构建的GIF。"""
    coords = np.array(instance.coords)
    if len(tour) == 0 or len(coords) == 0:
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    png_path = out_path
    if png_path.lower().endswith(".gif"):
        png_path = png_path[:-4] + ".png"
    else:
        png_path = png_path + ".png"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=12, color="#1f77b4")
    path_coords = coords[tour]
    ax.plot(path_coords[:, 0], path_coords[:, 1], color="#d62728", linewidth=1.5)
    ax.set_title("Tour (full)")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not _HAS_IMAGEIO:
        return True

    n = len(tour)
    step = max(1, n // max_frames)
    frames = []

    for k in range(2, n + 1, step):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(coords[:, 0], coords[:, 1], s=12, color="#1f77b4")

        path = tour[:k]
        path_coords = coords[path]
        ax.plot(path_coords[:, 0], path_coords[:, 1], color="#d62728", linewidth=1.5)

        ax.set_title(f"Tour progress: {k}/{n}")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    if frames:
        imageio.mimsave(out_path, frames, duration=0.15)
        return True

    return True
