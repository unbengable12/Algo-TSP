import os
import pandas as pd


def split_csv(input_path: str, output_dir: str = "split") -> None:
	"""按category分文件保存到指定目录。"""
	os.makedirs(output_dir, exist_ok=True)
	df = pd.read_csv(input_path)

	if "category" not in df.columns:
		raise ValueError("CSV中缺少 category 列")

	for category, group in df.groupby("category"):
		safe_name = str(category).strip().replace(" ", "_")
		if str(category).strip().lower() == "large":
			if len(group) % 2 != 0:
				group = group.iloc[:-1]
				print("警告: Large 类别行数为奇数，已丢弃最后一行以实现均分")
			mid = len(group) // 2
			part1 = group.iloc[:mid]
			part2 = group.iloc[mid:]
			part1.to_csv(os.path.join(output_dir, f"tsp_{safe_name}_part1.csv"), index=False)
			part2.to_csv(os.path.join(output_dir, f"tsp_{safe_name}_part2.csv"), index=False)
		else:
			out_path = os.path.join(output_dir, f"tsp_{safe_name}.csv")
			group.to_csv(out_path, index=False)


if __name__ == "__main__":
	split_csv("tsp_instances_solved.csv", "split")
