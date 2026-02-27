#!/usr/bin/env python3
"""
从 Egocentric-10K/clips 下每个 factory_* 目录各采样 5 条 .pose3d_hand 及同名的 .mp4 到 ./examples。
保证同名不冲突：目标文件名为 {folder_name}_{stem}.pose3d_hand / .mp4。

Usage:
    python scripts/sample_egocentric10k_to_examples.py
    python scripts/sample_egocentric10k_to_examples.py --base-dir /path/to/clips --out-dir ./examples --per-folder 5
"""
import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sample .pose3d_hand from each factory_* into examples")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/ss-oss1/data/dataset/egocentric/Egocentric-10K/clips",
        help="Clips 根目录（其下应有 factory_0, factory_1, ...）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./examples",
        help="输出目录",
    )
    parser.add_argument(
        "--per-folder",
        type=int,
        default=5,
        help="每个 factory_* 采样的条数",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not base.is_dir():
        raise SystemExit(f"Not a directory: {base}")

    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    factory_dirs = sorted(base.glob("factory_*"))
    if not factory_dirs:
        raise SystemExit(f"No factory_* dirs under {base}")

    total = 0
    for fd in factory_dirs:
        if not fd.is_dir():
            continue
        candidates = sorted(fd.rglob("*.pose3d_hand"))
        n = min(args.per_folder, len(candidates))
        if n == 0:
            print(f"[Skip] {fd.name}: no .pose3d_hand")
            continue
        chosen = random.sample(candidates, n)
        prefix = fd.name
        for src in chosen:
            stem = src.stem
            dest_pose = out_dir / f"{prefix}_{stem}.pose3d_hand"
            shutil.copy2(src, dest_pose)
            total += 1
            mp4_src = src.with_suffix(".mp4")
            if mp4_src.is_file():
                dest_mp4 = out_dir / f"{prefix}_{stem}.mp4"
                shutil.copy2(mp4_src, dest_mp4)
        print(f"[OK] {fd.name}: {n} -> {out_dir}")

    print(f"[Done] {total} files in {out_dir}")
    print(f"Run: python3 -m process.process_retarget --data-dir {out_dir} -o ./outputs/retarget_examples")


if __name__ == "__main__":
    main()
