#!/usr/bin/env python3
"""
将 filter_result.json 拆分为多个 part，便于并行渲染。

输入：filter_result.json（含 filtered_clips 列表）
输出：filter_result_part0.json, filter_result_part1.json, ... （每个可直接用于 --filter-result）

Usage:
    python scripts/split_filter_result.py --input filter_result.json --output-dir . --num-parts 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="拆分 filter_result.json 为多个 part")
    parser.add_argument(
        "--input",
        type=str,
        default="filter_result.json",
        help="filter_result.json 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认与 input 同目录",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=8,
        help="拆分数量",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[ERROR] 文件不存在: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("filtered_clips", [])
    if not clips:
        print(f"[ERROR] filtered_clips 为空")
        return

    n = len(clips)
    num_parts = max(1, args.num_parts)
    k, m = divmod(n, num_parts)
    parts = [
        clips[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(num_parts)
    ]

    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    total_count = data.get("total_count", n)
    for i, part in enumerate(parts):
        part_data = {
            "filtered_clips": part,
            "part_index": i,
            "num_parts": num_parts,
            "total_count": total_count,
            "part_count": len(part),
        }
        out_path = out_dir / f"filter_result_part{i}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(part_data, f, ensure_ascii=False, indent=2)
        print(f"  {out_path.name}: {len(part)} clips")

    # 同时生成 filter_result_parts.json（整体索引，便于按 part 选择）
    parts_index = {"num_parts": num_parts, "part_files": [f"filter_result_part{i}.json" for i in range(num_parts)]}
    index_path = out_dir / "filter_result_parts.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(parts_index, f, indent=2)
    print(f"\n[OK] 拆分完成: {n} clips -> {num_parts} parts")
    print(f"     索引: {index_path}")


if __name__ == "__main__":
    main()
