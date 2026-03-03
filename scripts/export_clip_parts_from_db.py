#!/usr/bin/env python3
"""
从 Supabase 数据库导出 clip 列表为 clip_parts.json，供 process_ik、process_retarget、
process_retarget_egodex 等通过 --clip-parts 使用，避免扫描文件夹。

DB 表需有 path（或自定义列）、dataset_name、id。path 应为完整文件路径：
- process_retarget: path 指向 .pose3d_hand
- process_retarget_egodex: path 指向 .hdf5
- process_ik: 从 path 提取 clip_id（basename），匹配 input-dir 下的 {clip_id}.json

Usage:
    # 导出 Ego10K 数据（需配置 SUPABASE_URL、SUPABASE_KEY 环境变量）
    python scripts/export_clip_parts_from_db.py \\
        --dataset-name Ego10K \\
        --output outputs/clip_parts.json \\
        --num-parts 4

    # 导出 ml-egodex，若 path 为相对路径则加 --base-path
    python scripts/export_clip_parts_from_db.py \\
        --dataset-name ml-egodex \\
        --output outputs/ml_egodex_clip_parts.json \\
        --base-path /home/ss-oss1/data/dataset/egocentric/ml-egodex/download
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def fetch_clip_paths_from_db(
    table_name: str = "egocentric_dataset_clips",
    dataset_name: str = "Ego10K",
    path_column: str = "path",
    batch_size: int = 1000,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    base_path: str | None = None,
) -> list[str]:
    """
    从 Supabase 表批量拉取 path，按 dataset_name 过滤。
    返回去重后的 path 列表。若 base_path 非空，则 prepend 到相对路径。
    """
    from supabase import create_client

    url = supabase_url or os.environ.get("SUPABASE_URL", "")
    key = supabase_key or os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise SystemExit(
            "需要 Supabase 配置。设置环境变量 SUPABASE_URL、SUPABASE_KEY，或通过 --supabase-url、--supabase-key 传入。"
        )

    client = create_client(url, key)
    paths: list[str] = []
    start = 0

    while True:
        response = (
            client.table(table_name)
            .select(path_column)
            .eq("dataset_name", dataset_name)
            .not_.is_("id", "null")
            .range(start, start + batch_size - 1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            break
        for r in rows:
            p = r.get(path_column)
            if p and isinstance(p, str):
                if base_path and not os.path.isabs(p):
                    p = os.path.join(base_path, p)
                paths.append(p)
        start += len(rows)
        if len(rows) < batch_size:
            break

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从数据库导出 clip_parts.json，供 process_ik / process_retarget 使用"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Ego10K",
        help="数据库中的 dataset_name 过滤条件，如 Ego10K、ml-egodex、Ego4d",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="egocentric_dataset_clips",
        help="Supabase 表名",
    )
    parser.add_argument(
        "--path-column",
        type=str,
        default="path",
        help="存储文件路径的列名",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="输出 clip_parts.json 路径",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=1,
        help="划分成多少个 part（与 process_clip_dataset 一致时便于并行）。1 表示不分 part",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="若 DB 中 path 为相对路径，则 prepend 此 base_path",
    )
    parser.add_argument(
        "--supabase-url",
        type=str,
        default=None,
        help="Supabase URL（也可用 SUPABASE_URL 环境变量）",
    )
    parser.add_argument(
        "--supabase-key",
        type=str,
        default=None,
        help="Supabase anon key（也可用 SUPABASE_KEY 环境变量）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="每批从数据库拉取条数",
    )
    args = parser.parse_args()

    paths = fetch_clip_paths_from_db(
        table_name=args.table,
        dataset_name=args.dataset_name,
        path_column=args.path_column,
        batch_size=args.batch_size,
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        base_path=args.base_path,
    )

    # 去重并保持顺序
    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    n = len(unique_paths)
    if n == 0:
        print(f"[export] 无数据（dataset_name={args.dataset_name}）")
        return

    # 划分 part
    if args.num_parts <= 1:
        clip_parts = [unique_paths]
    else:
        k, m = divmod(n, args.num_parts)
        clip_parts = [
            unique_paths[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(args.num_parts)
        ]

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clip_parts, f, indent=2)

    print(f"[export] {n} clips -> {out_path} ({len(clip_parts)} parts)")


if __name__ == "__main__":
    main()
