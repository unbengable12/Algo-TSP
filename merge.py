import os
import pandas as pd


def merge_csv(input_dir: str = "split", output_path: str = "tsp_instances_merged.csv") -> None:
    """合并split目录下所有CSV文件为一个文件。"""
    csv_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError("split目录下未找到CSV文件")

    frames = [pd.read_csv(p) for p in sorted(csv_files)]
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output_path, index=False)


if __name__ == "__main__":
    merge_csv("split", "tsp_instances_solved.csv")
