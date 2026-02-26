#!/usr/bin/env python3
# Python 3.11+ compat before any other imports
import src.compat  # noqa: F401

"""
Ray parallel batch: dual-arm IK + joint KF smoothing for all JSON files in a folder.

Output layout:
    {output_dir}/
    ├── data/
    │   ├── {clip_id_1}/
    │   │   ├── joint_trajectory.json
    │   │   └── stats.json
    │   ├── {clip_id_2}/
    │   │   ├── joint_trajectory.json
    │   │   └── stats.json
    │   └── ...
    └── samples/
        ├── {sampled_clip_1}.mp4
        └── ...

Usage:
    python process_ik.py --input-dir outputs/data --output-dir outputs/ik_results --sample-ratio 0.05
"""

import argparse
import glob
import math
import os
import random
import time

import ray


@ray.remote
def process_clip_remote(json_path: str, output_dir: str, do_render: bool, urdf_path: str) -> dict:
    """Ray remote: each worker initializes PyBullet (DIRECT) and processes one clip."""
    from src.ik import process_single_clip
    return process_single_clip(
        json_path=json_path,
        output_dir=output_dir,
        do_render=do_render,
        use_gui=False,
        urdf_path=urdf_path,
        verbose=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Ray parallel batch dual-arm IK")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory of JSON files")
    parser.add_argument("--output-dir", type=str, default="outputs/ik_results", help="Output root directory")
    parser.add_argument("--sample-ratio", type=float, default=0.05, help="Ratio of clips to render (0~1)")
    parser.add_argument("--num-cpus", type=int, default=None, help="Total CPUs for Ray (default: all)")
    parser.add_argument("--num-gpus", type=int, default=None, help="Total GPUs for Ray (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--render-resolution", type=list[int], default=[640, 480], help="Render resolution")
    parser.add_argument("--urdf-path", type=str, default="./assets/aloha_new_description/urdf/dual_piper.urdf", help="URDF path")
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_files:
        print(f"[Batch] No JSON files found in {args.input_dir}")
        return

    n_total = len(json_files)
    n_sample = max(1, math.ceil(n_total * args.sample_ratio))
    random.seed(args.seed)
    sample_indices = set(random.sample(range(n_total), min(n_sample, n_total)))
    sample_clips = {os.path.splitext(os.path.basename(json_files[i]))[0] for i in sample_indices}

    print(f"[Batch] {n_total} clips -> {args.output_dir} (render {len(sample_indices)})")

    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)

    ray_kwargs = {}
    if args.num_cpus is not None:
        ray_kwargs["num_cpus"] = args.num_cpus
    if args.num_gpus is not None:
        ray_kwargs["num_gpus"] = args.num_gpus
    ray.init(**ray_kwargs)

    cluster = ray.cluster_resources()
    total_cpus = cluster.get("CPU", 1)
    total_gpus = cluster.get("GPU", 0)
    max_workers = max(1, int(total_cpus))
    cpus_per_task = max(0.5, total_cpus / max_workers)
    gpus_per_task = total_gpus / max_workers if total_gpus > 0 else 0

    task_options: dict = {"num_cpus": cpus_per_task}
    if gpus_per_task > 0:
        task_options["num_gpus"] = gpus_per_task

    futures = []
    clip_ids = []
    for i, jp in enumerate(json_files):
        cid = os.path.splitext(os.path.basename(jp))[0]
        do_render = i in sample_indices
        futures.append(process_clip_remote.options(**task_options).remote(
            jp, args.output_dir, do_render, args.urdf_path,
        ))
        clip_ids.append(cid)

    t_start = time.time()
    results = []
    n_done = 0
    for cid, future in zip(clip_ids, futures):
        try:
            stats = ray.get(future)
            results.append(stats)
        except Exception as e:
            print(f"[Batch] {cid} failed: {e}")
            results.append({"clip_id": cid, "error": str(e)})
        n_done += 1
        if n_done % 50 == 0 or n_done == n_total:
            elapsed = time.time() - t_start
            print(f"[Batch] {n_done}/{n_total} ({elapsed:.0f}s)")

    ray.shutdown()

    # Summary
    import json
    total_time = time.time() - t_start
    success_results = [r for r in results if "error" not in r]
    total_frames = sum(r.get("n_frames", 0) for r in success_results)
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        "total": n_total,
        "success": len(success_results),
        "failed": sum(1 for r in results if "error" in r),
        "sample_count": len(sample_indices),
        "sample_ratio": args.sample_ratio,
        "total_time": total_time,
        "total_frames": total_frames,
        "avg_time_per_frame": total_time / total_frames if total_frames > 0 else 0,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Batch] Done: {summary['success']}/{n_total} ok, {summary['failed']} failed | "
          f"{total_time:.0f}s, {total_frames} frames, {summary_path}")


if __name__ == "__main__":
    main()
