#!/usr/bin/env python3
# Python 3.11+ compat before any other imports
import src.compat  # noqa: F401

"""
Ray parallel batch: dual-arm IK + joint KF smoothing for all JSON files in a folder.

Each Ray Actor holds a persistent PyBullet environment and processes multiple clips
sequentially, avoiding repeated env init/teardown overhead.

Crash-resilience strategy:
  - Application errors (bad data, numpy/scipy issues): caught inside the actor,
    env is reset for the next clip, the failed clip is recorded and skipped.
  - Process-level crashes (SIGSEGV, OOM): Ray auto-restarts the actor (max_restarts).
    The main loop detects RayActorError, marks the actor dead, re-queues the clip
    to a healthy actor, and spawns a replacement actor to maintain parallelism.
  - Poison clips (always crash): per-clip retry limit prevents one bad clip from
    killing all actors.

Output layout:
    {output_dir}/
    ├── data/
    │   ├── {clip_id_1}/
    │   │   ├── joint_trajectory.json
    │   │   └── stats.json
    │   └── ...
    └── samples/
        ├── video/
        │   ├── {sampled_clip_1}.mp4
        │   └── ...
        └── mask/
            ├── {sampled_clip_1}.mp4
            └── ...

Usage:
    python process_ik.py --input-dir outputs/data --output-dir outputs/ik_results --sample-ratio 0.05
"""

import argparse
import collections
import glob
import math
import os
import random
import time

import ray

MAX_CLIP_RETRIES = 2


@ray.remote(max_restarts=5, max_task_retries=0)
class IKWorkerActor:
    """Persistent PyBullet worker — reuses the same env across clips."""

    def __init__(self, urdf_path: str, camera_elevation_deg: float):
        self.urdf_path = urdf_path
        self.camera_elevation_deg = camera_elevation_deg
        self.robot_id: int | None = None
        self.robot_info: dict | None = None
        self._init_rest_poses: list[float] | None = None
        self._egl_loaded = False
        self._init_env()

    def _init_env(self):
        from src.ik import apply_robot_visual_settings, load_robot, parse_arm_info

        self.robot_id = load_robot(self.urdf_path, use_gui=False, use_egl=False)
        apply_robot_visual_settings(self.robot_id)
        self.robot_info = parse_arm_info(self.robot_id)
        self._init_rest_poses = list(self.robot_info["rest_poses"])
        self._egl_loaded = False

    def _ensure_egl(self):
        """Reinit env with EGL on first render clip.

        EGL must be loaded *before* ``p.loadURDF`` so the GPU renderer can
        register mesh data.  We therefore tear down the lightweight (no-EGL)
        env and rebuild it with EGL enabled.  This only happens once per actor
        lifetime (or after a reset).
        """
        if self._egl_loaded:
            return
        import pybullet as p

        try:
            p.disconnect()
        except Exception:
            pass
        from src.ik import apply_robot_visual_settings, load_robot, parse_arm_info

        self.robot_id = load_robot(self.urdf_path, use_gui=False, use_egl=True)
        apply_robot_visual_settings(self.robot_id)
        self.robot_info = parse_arm_info(self.robot_id)
        self._init_rest_poses = list(self.robot_info["rest_poses"])
        self._egl_loaded = True

    def _reset_env(self):
        import pybullet as p

        try:
            p.disconnect()
        except Exception:
            pass
        self._init_env()

    def process_clip(self, json_path: str, output_dir: str, do_render: bool) -> dict:
        from src.ik import process_single_clip

        if do_render:
            self._ensure_egl()
        self.robot_info["rest_poses"] = list(self._init_rest_poses)
        try:
            return process_single_clip(
                json_path=json_path,
                output_dir=output_dir,
                do_render=do_render,
                use_gui=False,
                urdf_path=self.urdf_path,
                verbose=False,
                camera_elevation_deg=self.camera_elevation_deg,
                robot_id=self.robot_id,
                robot_info=self.robot_info,
                auto_disconnect=False,
            )
        except Exception:
            try:
                self._reset_env()
            except Exception:
                pass
            raise

    def ping(self) -> bool:
        return True

    def shutdown(self):
        import pybullet as p

        try:
            p.disconnect()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Ray parallel batch dual-arm IK")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory of JSON files")
    parser.add_argument("--output-dir", type=str, default="outputs/ik_results", help="Output root directory")
    parser.add_argument("--sample-ratio", type=float, default=0.05, help="Ratio of clips to render (0~1)")
    parser.add_argument("--num-cpus", type=int, default=None, help="Total CPUs for Ray (default: all)")
    parser.add_argument("--num-gpus", type=int, default=None, help="Total GPUs for Ray (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--render-resolution", type=list[int], default=[512, 512], help="Render resolution")
    parser.add_argument("--urdf-path", type=str, default="./assets/aloha_new_description/urdf/dual_piper.urdf", help="URDF path")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers (default: num_cpus, capped at 64)")
    parser.add_argument("--camera-elevation", type=float, default=45.0, dest="camera_elevation_deg",
                        help="相机俯视角（度），默认 45")
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_files:
        print(f"[Batch] No JSON files found in {args.input_dir}")
        return

    n_total = len(json_files)
    n_sample = max(1, math.ceil(n_total * args.sample_ratio))
    random.seed(args.seed)
    sample_indices = set(random.sample(range(n_total), min(n_sample, n_total)))

    print(f"[Batch] {n_total} clips -> {args.output_dir} (render {len(sample_indices)})")

    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
    video_dir = os.path.join(args.output_dir, "samples", "video")
    mask_dir = os.path.join(args.output_dir, "samples", "mask")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Remove stale .tmp.mp4 from previous interrupted runs
    for d in (video_dir, mask_dir):
        if os.path.isdir(d):
            removed = 0
            for f in os.listdir(d):
                if f.endswith(".tmp.mp4"):
                    try:
                        os.remove(os.path.join(d, f))
                        removed += 1
                    except OSError:
                        pass
            if removed:
                print(f"[Batch] Removed {removed} stale .tmp.mp4 from {os.path.basename(d)}")

    ray_kwargs = {}
    if args.num_cpus is not None:
        ray_kwargs["num_cpus"] = args.num_cpus
    if args.num_gpus is not None:
        ray_kwargs["num_gpus"] = args.num_gpus
    ray.init(**ray_kwargs)

    cluster = ray.cluster_resources()
    total_cpus = cluster.get("CPU", 1)
    total_gpus = cluster.get("GPU", 0)
    if args.max_workers is not None:
        max_workers = max(1, args.max_workers)
    else:
        max_workers = max(1, min(int(total_cpus), 256))
    cpus_per_worker = max(0.5, total_cpus / max_workers)
    gpus_per_worker = total_gpus / max_workers if total_gpus > 0 else 0
    print(f"[Batch] {max_workers} workers ({cpus_per_worker:.1f} CPU each, "
          f"{total_cpus:.0f} CPUs total, {total_gpus:.0f} GPUs total)")

    actor_options: dict = {"num_cpus": cpus_per_worker}
    if gpus_per_worker > 0:
        actor_options["num_gpus"] = gpus_per_worker

    # ---- Actor pool with liveness tracking ----
    actors: list = [
        IKWorkerActor.options(**actor_options).remote(
            args.urdf_path, args.camera_elevation_deg
        )
        for _ in range(max_workers)
    ]
    actor_alive: list[bool] = [True] * max_workers
    n_replacements = 0

    # ---- Work-queue state ----
    clip_queue: collections.deque[int] = collections.deque(range(n_total))
    clip_retries: dict[int, int] = collections.defaultdict(int)
    # future -> (actor_idx, clip_idx, clip_id)
    future_to_meta: dict[ray.ObjectRef, tuple[int, int, str]] = {}
    results: list[dict] = []
    n_done = 0

    def _try_dispatch(ai: int) -> bool:
        """Try to dispatch the next clip to actor *ai*. Returns True on success."""
        if not clip_queue or not actor_alive[ai]:
            return False
        ci = clip_queue.popleft()
        jp = json_files[ci]
        cid = os.path.splitext(os.path.basename(jp))[0]
        do_render = ci in sample_indices
        try:
            f = actors[ai].process_clip.remote(jp, args.output_dir, do_render)
        except Exception:
            actor_alive[ai] = False
            clip_queue.appendleft(ci)
            return False
        future_to_meta[f] = (ai, ci, cid)
        return True

    def _dispatch_idle():
        """Dispatch queued clips to all idle alive actors."""
        busy = {meta[0] for meta in future_to_meta.values()}
        for ai in range(len(actors)):
            if not clip_queue:
                break
            if actor_alive[ai] and ai not in busy:
                _try_dispatch(ai)

    def _replace_actor(ai: int) -> bool:
        """Try to spawn a replacement actor in slot *ai*."""
        nonlocal n_replacements
        try:
            actors[ai] = IKWorkerActor.options(**actor_options).remote(
                args.urdf_path, args.camera_elevation_deg
            )
            actor_alive[ai] = True
            n_replacements += 1
            print(f"[Batch] Spawned replacement actor in slot {ai}")
            return True
        except Exception as e:
            print(f"[Batch] Failed to spawn replacement actor: {e}")
            return False

    # ---- Seed each actor with one clip ----
    for ai in range(len(actors)):
        _try_dispatch(ai)

    t_start = time.time()

    while future_to_meta:
        ready, _ = ray.wait(list(future_to_meta.keys()), num_returns=1)
        for f in ready:
            ai, ci, cid = future_to_meta.pop(f)

            try:
                stats = ray.get(f)
                results.append(stats)
                n_done += 1
                _try_dispatch(ai)

            except ray.exceptions.RayActorError as e:
                # Actor process crashed — mark dead, handle clip re-queue
                actor_alive[ai] = False
                print(f"[Batch] Actor {ai} died while processing {cid}: {e!r}")

                clip_retries[ci] += 1
                if clip_retries[ci] <= MAX_CLIP_RETRIES:
                    clip_queue.appendleft(ci)
                else:
                    print(f"[Batch] {cid} exceeded {MAX_CLIP_RETRIES} retries, skipping")
                    results.append({"clip_id": cid, "error": f"exceeded retries: {e}"})
                    n_done += 1

                # Try to replace the dead actor, then dispatch idle actors
                _replace_actor(ai)
                _dispatch_idle()

            except Exception as e:
                # Application-level error — actor is still alive
                print(f"[Batch] {cid} failed: {e}")
                results.append({"clip_id": cid, "error": str(e)})
                n_done += 1
                _try_dispatch(ai)

        if n_done % 50 == 0 or n_done == n_total:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_total - n_done) / rate if rate > 0 else 0
            n_alive = sum(actor_alive)
            print(f"[Batch] {n_done}/{n_total} ({elapsed:.0f}s, {rate:.1f} clips/s, "
                  f"ETA {eta:.0f}s, {n_alive}/{len(actors)} workers alive)")

    # ---- Drain remaining clips (all actors dead) ----
    while clip_queue:
        ci = clip_queue.popleft()
        cid = os.path.splitext(os.path.basename(json_files[ci]))[0]
        results.append({"clip_id": cid, "error": "all workers dead"})
        n_done += 1

    # ---- Shutdown surviving actors ----
    for ai, actor in enumerate(actors):
        if actor_alive[ai]:
            try:
                ray.get(actor.shutdown.remote(), timeout=10)
            except Exception:
                pass

    # ---- Summary ----
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
        "actor_replacements": n_replacements,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Batch] Done: {summary['success']}/{n_total} ok, {summary['failed']} failed | "
          f"{total_time:.0f}s, {total_frames} frames, {n_replacements} actor replacements, {summary_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
