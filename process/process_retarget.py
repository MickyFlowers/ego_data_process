# Python 3.11+ compat before any other imports
import src.compat  # noqa: F401

"""
Ray parallel batch: retarget all .pose3d_hand files under a directory (recursive by default).
Finds all clip_ids by scanning for *.pose3d_hand; one task per unique clip_id (first path wins).
Sampled clips get replay video generated in the same worker (no separate replay pass).

Output layout:
    {output_dir}/
    ├── data/
    │   ├── {clip_id_1}.json
    │   ├── {clip_id_2}.json
    │   └── ...
    ├── video/
    │   ├── {sampled_clip_1}_retarget_replay.mp4
    │   └── ...
    └── summary.json

Usage:
    python process/process_retarget.py --data-dir /path/to/pose3d_hand --output-dir outputs/retarget
"""

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import ray


@ray.remote
def process_clip_remote(
    pose3d_path: str,
    output_dir: str,
    model_dir: str,
    methods: str,
    do_replay: bool,
    replay_frame_ratio: float,
) -> dict:
    """Ray worker: create pipeline, retarget + optional replay."""
    from src.retarget import HandRetargetPipeline

    pose3d_path = Path(pose3d_path)
    output_dir = Path(output_dir)
    clip_id = pose3d_path.stem
    t0 = time.time()

    try:
        pipeline = HandRetargetPipeline(model_dir=model_dir)
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_json = data_dir / f"{clip_id}.json"

        output_data = pipeline.retarget(
            data_path=pose3d_path,
            methods=methods,
            out_path=out_json,
            verbose=False,
        )
        n_frames = len(output_data.get("poses", {}).get("left", []))
        retarget_time = time.time() - t0

        replay_time = 0.0
        if do_replay:
            t1 = time.time()
            video_dir = output_dir / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            out_video = video_dir / f"{clip_id}.mp4"
            pipeline.replay_json(
                json_path=out_json,
                out_path=out_video,
                video_path=None,
                sample_ratio=replay_frame_ratio,
                verbose=False,
            )
            replay_time = time.time() - t1

        return {
            "clip_id": clip_id,
            "n_frames": n_frames,
            "retarget_time": retarget_time,
            "replay_time": replay_time,
            "total_time": time.time() - t0,
            "replayed": do_replay,
        }
    except Exception as e:
        return {
            "clip_id": clip_id,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Ray parallel batch retarget (.pose3d_hand -> JSON + optional replay video)"
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Input dir (searched recursively for .pose3d_hand)")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--no-recursive", action="store_true", help="Only look in data-dir, not subdirs")
    parser.add_argument("-m", "--model-dir", type=str,
                        default="./assets/mano_v1_2/models",
                        help="MANO model directory")
    parser.add_argument("--methods", type=str, default="pinch_plane", help="Retarget method")
    parser.add_argument("--sample-ratio", type=float, default=0.1, help="Ratio of clips to replay (0~1)")
    parser.add_argument("--replay-frame-ratio", type=float, default=1.0,
                        help="Frame sampling ratio per video for replay (0,1]")
    parser.add_argument("--no-replay", action="store_true", help="Do not generate replay videos")
    parser.add_argument("--num-cpus", type=int, default=None, help="Total CPUs for Ray (default: all)")
    parser.add_argument("--num-gpus", type=int, default=None, help="Total GPUs for Ray (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    input_dir = Path(args.data_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {input_dir}")

    if args.no_recursive:
        candidates = sorted(input_dir.glob("*.pose3d_hand"))
    else:
        candidates = sorted(input_dir.rglob("*.pose3d_hand"), key=lambda p: (p.stem, str(p)))
    # One path per clip_id (first found wins)
    seen: set[str] = set()
    files = []
    for p in candidates:
        if p.stem not in seen:
            seen.add(p.stem)
            files.append(p)
    if not files:
        print(f"[Batch] No .pose3d_hand files under {input_dir}")
        return

    n_total = len(files)

    # Which clips get replay (same logic as process_ik)
    if args.no_replay:
        sample_indices: set[int] = set()
    else:
        n_sample = max(1, math.ceil(n_total * args.sample_ratio))
        random.seed(args.seed)
        sample_indices = set(random.sample(range(n_total), min(n_sample, n_total)))

    sample_clips = {files[i].stem for i in sample_indices}
    print(f"[Batch] {n_total} clips -> {args.output_dir} (replay {len(sample_indices)})")

    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
    if sample_indices:
        os.makedirs(os.path.join(args.output_dir, "video"), exist_ok=True)

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
    for i, fp in enumerate(files):
        do_replay = i in sample_indices
        futures.append(process_clip_remote.options(**task_options).remote(
            str(fp), args.output_dir, args.model_dir,
            args.methods, do_replay, args.replay_frame_ratio,
        ))
        clip_ids.append(fp.stem)

    t_start = time.time()
    results = []
    n_done = 0
    for cid, future in zip(clip_ids, futures):
        try:
            stats = ray.get(future)
            results.append(stats)
        except Exception as e:
            print(f"[Batch] FAILED {cid}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            results.append({"clip_id": cid, "error": str(e)})
        n_done += 1
        if n_done % 50 == 0 or n_done == n_total:
            elapsed = time.time() - t_start
            print(f"[Batch] {n_done}/{n_total} ({elapsed:.0f}s)")

    # Summary (write before shutdown)
    total_time = time.time() - t_start
    success_results = [r for r in results if "error" not in r]
    total_frames = sum(r.get("n_frames", 0) for r in success_results)
    summary_path = os.path.join(args.output_dir, "summary.json")
    failed_list = [r for r in results if "error" in r]
    summary = {
        "total": n_total,
        "success": len(success_results),
        "failed": len(failed_list),
        "failed_clips": [{"clip_id": r["clip_id"], "error": r["error"]} for r in failed_list],
        "sample_count": len(sample_indices),
        "sample_ratio": args.sample_ratio,
        "total_time": total_time,
        "total_frames": total_frames,
        "avg_time_per_frame": total_time / total_frames if total_frames > 0 else 0,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Batch] Done: {summary['success']}/{n_total} ok, {summary['failed']} failed | "
          f"{total_time:.0f}s, {total_frames} frames, {summary_path}")

    ray.shutdown()

    if failed_list:
        print("\n[Batch] Failed clips:", file=sys.stderr)
        for r in failed_list:
            print(f"  {r['clip_id']}: {r['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
