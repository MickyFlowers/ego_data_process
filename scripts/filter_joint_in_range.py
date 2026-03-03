#!/usr/bin/env python3
"""
从 ml-egodex/ik/data 筛选前 6 个关节角度均在 [-pi, pi] 范围内的数据。

对每个 clip：若 left/right 所有帧的前 6 个关节（revolute）全部在 [-pi, pi] 内，则通过筛选。
输出 JSON 到当前目录，包含通过筛选的 clip 列表、总数、比例。使用多进程并带进度条。

Usage:
    python scripts/filter_joint_in_range.py
    python scripts/filter_joint_in_range.py --clip-parts /path/to/clip_parts.json --output filter_result.json
    python scripts/filter_joint_in_range.py --input-dir /path/to/ik/data --output filter_result.json --num-workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable

try:
    import orjson
    _USE_ORJSON = True
except ImportError:
    _USE_ORJSON = False


CLIP_PARTS_PATH = "/home/ss-oss1/data/dataset/egocentric/ml-egodex/clip_parts.json"
DATA_DIR = "/home/ss-oss1/data/dataset/egocentric/ml-egodex/ik/data"
PI = math.pi


def _load_clip_ids_from_clip_parts(path: str) -> list[str]:
    """从 clip_parts.json 提取所有 clip_id。支持 [[path, ...], ...] 或 [clip_id, ...] 格式。"""
    data = _load_json(path)
    ids = []
    for item in data:
        if isinstance(item, list):
            for p in item:
                ids.append(Path(p).name if isinstance(p, str) and p else "")
        else:
            ids.append(Path(item).name if isinstance(item, str) else str(item))
    return [x for x in ids if x]


def _load_json(path: str):
    if _USE_ORJSON:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_clip_in_range(args: tuple[str, str]) -> tuple[str, bool]:
    """Worker: 检查单个 clip 是否所有帧的前 6 关节均在 [-pi, pi]。返回 (clip_id, passed)."""
    clip_dir, traj_path = args
    try:
        data = _load_json(traj_path)
    except (json.JSONDecodeError, OSError, ValueError):
        return (clip_dir, False)

    left = data.get("left_joint_trajectory", [])
    right = data.get("right_joint_trajectory", [])

    def in_range(traj: list) -> bool:
        for frame in traj:
            for j in range(6):
                v = frame[j]
                if v < -PI or v > PI:
                    return False
        return True

    if not left and not right:
        return (clip_dir, False)
    if not in_range(left) or not in_range(right):
        return (clip_dir, False)
    return (clip_dir, True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="筛选前 6 关节在 [-pi, pi] 范围内的 IK 数据",
    )
    parser.add_argument(
        "--clip-parts",
        type=str,
        default=CLIP_PARTS_PATH,
        help="clip_parts.json 路径，从中读取所有 clip_id；data 目录下为 clip_id 子文件夹",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="IK data 根目录（clip_id 子文件夹所在处）。默认：clip_parts 同级的 ik/data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="filtered_in_range.json",
        help="输出 JSON 路径（存到当前目录）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(os.cpu_count() or 4, 16),
        help="并行进程数",
    )
    args = parser.parse_args()

    # 确定 data 目录
    if args.input_dir:
        input_path = Path(args.input_dir)
    elif args.clip_parts and Path(args.clip_parts).is_file():
        input_path = Path(args.clip_parts).parent / "ik" / "data"
    else:
        input_path = Path(DATA_DIR)
    if not input_path.is_dir():
        print(f"[ERROR] 输入目录不存在: {input_path}")
        return

    # 从 clip_parts.json 读取 clip_id 列表（避免扫描文件系统）
    if args.clip_parts and Path(args.clip_parts).is_file():
        clip_ids = _load_clip_ids_from_clip_parts(args.clip_parts)
        tasks = [(cid, str(input_path / cid / "joint_trajectory.json")) for cid in clip_ids]
        print(f"[INFO] 从 {args.clip_parts} 读取 {len(clip_ids)} 个 clip_id")
    else:
        traj_files = list(input_path.glob("*/joint_trajectory.json"))
        tasks = [(f.parent.name, str(f)) for f in traj_files]

    total = len(tasks)
    if total == 0:
        print(f"[WARN] 未找到任何 joint_trajectory.json: {input_path}")
        return

    print(f"[INFO] 共 {total} 个 clip，使用 {args.num_workers} 进程筛选...")

    if args.num_workers <= 1:
        results = []
        for t in tqdm(tasks, desc="筛选"):
            results.append(_check_clip_in_range(t))
    else:
        with multiprocessing.Pool(args.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_check_clip_in_range, tasks, chunksize=128),
                    total=total,
                    desc="筛选",
                )
            )

    filtered = [r[0] for r in results if r[1]]
    count = len(filtered)
    proportion = count / total if total > 0 else 0.0

    out_data = {
        "filtered_clips": filtered,
        "total_count": total,
        "filtered_count": count,
        "proportion": proportion,
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 通过筛选: {count} / {total} = {proportion:.2%}")
    print(f"     结果已保存: {out_path}")


if __name__ == "__main__":
    main()
