#!/usr/bin/env python3
"""
PyBullet 渲染：从 filter_result.json 读取 clip 列表，加载已有 joint_trajectory，渲染得到 mask 和 video。

与 process_ik 的 render 功能一致，但不做 IK，直接使用已有的 joint_trajectory.json。
数据目录结构：data_dir/{clip_id}/ 含 meta.json + joint_trajectory.json。

Output:
    {output_dir}/video/{clip_id}.mp4  (机器人叠加重叠原视频，或仅机器人)
    {output_dir}/mask/{clip_id}.mp4   (二值分割 mask)

默认短边 512（等比例缩放，如 1920×1080 -> 910×512）。

Usage:
    python -m process.process_render_filtered --filter-result /path/to/filter_result.json --data-dir /path/to/ik/data --output-dir outputs/render
    python -m process.process_render_filtered --filter-result filter_result_part0.json --output-dir outputs/render --no-ray  # 避免 Ray GCS 连接失败
"""
from __future__ import annotations

import src.compat  # noqa: F401

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import ray

from process.process_render import (
    RENDER_RESOLUTION,
    _prepare_render_urdf,
)

FILTER_RESULT_PATH = "/home/ss-oss1/data/dataset/egocentric/ml-egodex/filter_result.json"
DATA_DIR = "/home/ss-oss1/data/dataset/egocentric/ml-egodex/ik/data"


def load_filtered_clips(path: str) -> list[str]:
    """从 filter_result.json 读取 filtered_clips 列表。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("filtered_clips", [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyBullet 渲染：从 filter_result.json 读取 clips，渲染 mask + video",
    )
    parser.add_argument(
        "--filter-result",
        type=str,
        default=FILTER_RESULT_PATH,
        help="filter_result.json 路径（含 filtered_clips）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help="IK data 根目录（其下为 clip_id 子文件夹）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/render_filtered",
        help="输出目录，将创建 video/ 和 mask/ 子目录",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="./assets/aloha_new_description/urdf/dual_piper.urdf",
        help="URDF 路径",
    )
    parser.add_argument(
        "--short-side",
        type=int,
        default=512,
        help="短边缩放（等比例），0 表示使用固定 --width/--height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=RENDER_RESOLUTION[0],
        help="渲染宽度（--short-side 为 0 时生效）",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=RENDER_RESOLUTION[1],
        help="渲染高度（--short-side 为 0 时生效）",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Ray 总 CPU 数",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Ray 总 GPU 数",
    )
    parser.add_argument(
        "--no-ray",
        action="store_true",
        help="禁用 Ray，使用 multiprocessing（避免 GCS 连接失败）。与 process_ik 不同，无 Ray 依赖。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="并行 worker 数。--no-ray 时生效；默认等于 min(num_cpus, n_render)。",
    )
    args = parser.parse_args()

    # ---- 从 filter_result.json 加载 clip 列表 ----
    filter_path = Path(args.filter_result)
    if not filter_path.is_file():
        print(f"[RenderFiltered] filter_result 不存在: {filter_path}")
        sys.exit(1)
    filtered_clips = load_filtered_clips(str(filter_path))
    if not filtered_clips:
        print(f"[RenderFiltered] filtered_clips 为空")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"[RenderFiltered] data 目录不存在: {data_dir}")
        sys.exit(1)

    # ---- 直接构建 clip 目录路径，不扫描文件系统 ----
    clip_dirs = [str(data_dir / cid) for cid in filtered_clips]
    n_render = len(clip_dirs)
    print(f"[RenderFiltered] {n_render} clips from {filter_path.name}")

    # ---- 输出目录 ----
    os.makedirs(os.path.join(args.output_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "mask"), exist_ok=True)

    use_ray = not args.no_ray
    num_workers = args.num_workers
    total_cpus = 1
    cluster = {}
    if num_workers is None:
        if use_ray:
            try:
                ray_kwargs = {}
                if args.num_cpus is not None:
                    ray_kwargs["num_cpus"] = args.num_cpus
                if args.num_gpus is not None:
                    ray_kwargs["num_gpus"] = args.num_gpus
                ray_kwargs["ignore_reinit_error"] = True
                ray.init(**ray_kwargs)
                cluster = ray.cluster_resources()
                total_cpus = int(cluster.get("CPU", 1))
                num_workers = max(1, min(total_cpus, n_render))
            except Exception as e:
                print(f"[RenderFiltered] Ray init 失败 (GCS 连接等): {e}", flush=True)
                print("[RenderFiltered] 回退到 --no-ray 模式，运行前可执行 ray stop 清理残留", flush=True)
                use_ray = False
                import multiprocessing
                total_cpus = multiprocessing.cpu_count()
                num_workers = max(1, min(total_cpus, n_render))
        else:
            import multiprocessing
            total_cpus = multiprocessing.cpu_count()
            num_workers = max(1, min(total_cpus, n_render))
    else:
        num_workers = max(1, num_workers)
        if use_ray:
            try:
                ray_kwargs = {"ignore_reinit_error": True}
                if args.num_cpus is not None:
                    ray_kwargs["num_cpus"] = args.num_cpus
                if args.num_gpus is not None:
                    ray_kwargs["num_gpus"] = args.num_gpus
                ray.init(**ray_kwargs)
                cluster = ray.cluster_resources()
                total_cpus = int(cluster.get("CPU", 1))
            except Exception as e:
                print(f"[RenderFiltered] Ray init 失败: {e}，回退到 --no-ray", flush=True)
                use_ray = False
                import multiprocessing
                total_cpus = multiprocessing.cpu_count()

    worker_clips: list[list[str]] = [[] for _ in range(num_workers)]
    for i, cd in enumerate(clip_dirs):
        worker_clips[i % num_workers].append(cd)
    active: list[tuple[int, list[str]]] = [
        (w, clips) for w, clips in enumerate(worker_clips) if clips
    ]

    print(f"[RenderFiltered] {num_workers} workers ({'Ray' if use_ray else 'multiprocessing'}), {total_cpus} CPUs")
    for w, clips in active:
        print(f"  Worker {w}: {len(clips)} clips")

    t_start = time.time()
    all_results: list[dict] = []

    if use_ray:
        cpus_per_worker = max(0.5, total_cpus / num_workers)
        gpus_per_worker = cluster.get("GPU", 0) / num_workers if cluster.get("GPU", 0) else 0
        task_opts: dict = {"num_cpus": cpus_per_worker}
        if gpus_per_worker > 0:
            task_opts["num_gpus"] = gpus_per_worker
        future_to_worker: dict = {}
        pending_futures: list = []
        for w, clips in active:
            fut = render_worker_both.options(**task_opts).remote(
                clip_dirs=clips,
                output_dir=args.output_dir,
                urdf_path=os.path.abspath(args.urdf_path),
                render_width=args.width,
                render_height=args.height,
                worker_id=w,
                short_side=args.short_side,
            )
            future_to_worker[fut] = (w, clips)
            pending_futures.append(fut)
        while pending_futures:
            done, pending_futures = ray.wait(pending_futures, num_returns=1)
            for fut in done:
                w, clips = future_to_worker[fut]
                try:
                    worker_results = ray.get(fut)
                    all_results.extend(worker_results)
                    n_ok = sum(1 for r in worker_results if "error" not in r)
                    print(
                        f"[RenderFiltered] Worker {w}: {n_ok}/{len(worker_results)} ok "
                        f"({len(all_results)}/{n_render} total)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[RenderFiltered] Worker {w} failed: {e}", file=sys.stderr, flush=True)
                    for cd in clips:
                        all_results.append({"clip_id": os.path.basename(cd), "error": str(e)})
        ray.shutdown()
    else:
        import multiprocessing
        with multiprocessing.Pool(num_workers) as pool:
            worker_args = [
                (clips, args.output_dir, os.path.abspath(args.urdf_path), args.width, args.height, w, args.short_side)
                for w, clips in active
            ]
            for (w, clips), worker_results in zip(
                active,
                pool.starmap(_render_clips_batch, worker_args),
            ):
                all_results.extend(worker_results)
                n_ok = sum(1 for r in worker_results if "error" not in r)
                print(
                    f"[RenderFiltered] Worker {w}: {n_ok}/{len(worker_results)} ok "
                    f"({len(all_results)}/{n_render} total)",
                    flush=True,
                )

    total_time = time.time() - t_start
    success = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]
    total_frames = sum(r.get("n_frames", 0) for r in success)

    summary = {
        "total": n_render,
        "success": len(success),
        "failed": len(failed),
        "failed_clips": [{"clip_id": r["clip_id"], "error": r["error"]} for r in failed],
        "total_time": total_time,
        "total_frames": total_frames,
    }
    summary_path = os.path.join(args.output_dir, "render_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[RenderFiltered] Done: {len(success)}/{n_render} ok, {len(failed)} failed | "
        f"{total_time:.0f}s, {total_frames} frames -> {summary_path}"
    )
    if failed:
        for r in failed:
            print(f"  {r['clip_id']}: {r['error']}", file=sys.stderr)

    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# 渲染逻辑（可被 Ray 或 multiprocessing 调用）
# ---------------------------------------------------------------------------

def _render_clips_batch(
    clip_dirs: list[str],
    output_dir: str,
    urdf_path: str,
    render_width: int,
    render_height: int,
    worker_id: int,
    short_side: int = 512,
) -> list[dict]:
    """渲染一批 clip，输出 video 和 mask。供 multiprocessing 或 Ray 调用。"""
    import pybullet as p
    import pybullet_data
    import imageio
    from PIL import Image

    from src.ik import (
        parse_arm_info,
        set_arm_joints,
        set_gripper,
        build_view_matrix_from_camera_matrix,
        build_projection_matrix_from_intrinsics,
        scale_intrinsics,
        short_side_resolution,
    )

    cid = p.connect(p.DIRECT)
    egl_loaded = False
    try:
        import pkgutil
        egl = pkgutil.get_loader("eglRenderer")
        if egl:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            egl_loaded = True
    except Exception:
        pass
    renderer = p.ER_BULLET_HARDWARE_OPENGL if egl_loaded else p.ER_TINY_RENDERER
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)

    render_urdf = _prepare_render_urdf(urdf_path)
    robot_id = p.loadURDF(
        render_urdf, useFixedBase=True,
        flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
    )
    robot_info = parse_arm_info(robot_id)

    tag = f"[Worker {worker_id}]"
    print(f"{tag} PyBullet ready (EGL={'yes' if egl_loaded else 'no'}), {len(clip_dirs)} clips", flush=True)

    results: list[dict] = []
    t0 = time.time()
    total_frames_written = 0

    video_dir = os.path.join(output_dir, "video")
    mask_dir = os.path.join(output_dir, "mask")

    for clip_idx, clip_dir in enumerate(clip_dirs):
        clip_path = Path(clip_dir)
        clip_id = clip_path.name
        meta_file = clip_path / "meta.json"
        traj_file = clip_path / "joint_trajectory.json"

        if not meta_file.exists() or not traj_file.exists():
            results.append({"clip_id": clip_id, "error": "missing meta.json or joint_trajectory.json"})
            continue

        try:
            with open(meta_file) as f:
                meta = json.load(f)
            with open(traj_file) as f:
                traj_data = json.load(f)
        except Exception as ex:
            results.append({"clip_id": clip_id, "error": str(ex)})
            continue

        left_traj = traj_data["left_joint_trajectory"]
        right_traj = traj_data["right_joint_trajectory"]
        n_frames = min(len(left_traj), len(right_traj))
        fps = meta.get("fps", 30)

        camera_intrinsics = np.array(meta["camera_intrinsics"], dtype=np.float64)
        camera_extrinsics = np.array(meta["camera_extrinsics"], dtype=np.float64)
        img_size = meta.get("img_size", [render_width, render_height])
        iw, ih = max(1, img_size[0]), max(1, img_size[1])
        if short_side > 0:
            rw, rh = short_side_resolution(iw, ih, short_side)
        else:
            rw, rh = render_width, render_height

        render_K = scale_intrinsics(camera_intrinsics, img_size, [rw, rh])
        view_matrix = build_view_matrix_from_camera_matrix(camera_extrinsics)
        proj_matrix = build_projection_matrix_from_intrinsics(
            render_K, rw, rh
        )

        source_reader = None
        video_path = meta.get("video_path")
        if video_path and os.path.isfile(video_path):
            try:
                source_reader = imageio.get_reader(video_path)
            except Exception:
                source_reader = None

        out_video = os.path.join(video_dir, f"{clip_id}.mp4")
        out_mask = os.path.join(mask_dir, f"{clip_id}.mp4")
        video_writer = imageio.get_writer(
            out_video, fps=fps, codec="libx264", macro_block_size=1,
            output_params=["-preset", "ultrafast"],
        )
        mask_writer = imageio.get_writer(
            out_mask, fps=fps, codec="libx264", macro_block_size=1,
            output_params=["-preset", "ultrafast"],
        )

        written = 0
        need_resize: bool | None = None

        for fi in range(n_frames):
            lq = left_traj[fi]
            rq = right_traj[fi]
            set_arm_joints(robot_info, lq[:6], rq[:6])
            lg = float(lq[6]) if len(lq) > 6 else 0.0
            rg = float(rq[6]) if len(rq) > 6 else 0.0
            set_gripper(robot_info, lg, rg)

            light_configs = [
                ([0.5, 0.0, 1.5], 0.45, 1),
                ([-1.0, 0.5, 0.8], 0.30, 0),
                ([0.0, -0.5, 1.0], 0.25, 0),
            ]
            accum = np.zeros((rh, rw, 3), dtype=np.float64)
            seg_out = None
            for l_dir, l_weight, l_shadow in light_configs:
                _, _, _rgba, _, _seg = p.getCameraImage(
                    rw, rh,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix,
                    shadow=l_shadow,
                    lightDirection=l_dir,
                    lightColor=[1.0, 1.0, 1.0],
                    lightDistance=2.0,
                    lightAmbientCoeff=0.3,
                    lightDiffuseCoeff=0.8,
                    lightSpecularCoeff=0.2,
                    renderer=renderer,
                )
                _rgb = np.asarray(_rgba, dtype=np.uint8).reshape(
                    (rh, rw, 4))[:, :, :3]
                accum += _rgb.astype(np.float64) * l_weight
                if seg_out is None:
                    seg_out = _seg

            robot_rgb = np.clip(accum, 0, 255).astype(np.uint8)
            mask = np.asarray(seg_out).reshape((rh, rw)) >= 0

            mask_uint8 = (mask.astype(np.uint8) * 255)
            mask_writer.append_data(np.stack([mask_uint8] * 3, axis=-1))

            bg = None
            if source_reader is not None:
                try:
                    bg = np.asarray(source_reader.get_data(fi))
                except (IndexError, Exception):
                    bg = None

            if bg is not None:
                if bg.ndim == 2:
                    bg = np.stack([bg] * 3, axis=-1)
                elif bg.shape[-1] == 4:
                    bg = bg[..., :3]
                bg_h, bg_w = bg.shape[:2]
                if need_resize is None:
                    need_resize = (rw, rh) != (bg_w, bg_h)
                if need_resize:
                    robot_rgb = np.asarray(
                        Image.fromarray(robot_rgb).resize((bg_w, bg_h), Image.BILINEAR)
                    )
                    mask_resized = np.asarray(
                        Image.fromarray(mask_uint8).resize((bg_w, bg_h), Image.NEAREST)
                    ) > 127
                else:
                    mask_resized = mask
                bg[mask_resized] = robot_rgb[mask_resized]
                if (bg_w, bg_h) != (rw, rh):
                    bg = np.asarray(
                        Image.fromarray(bg).resize((rw, rh), Image.BILINEAR)
                    )
                video_writer.append_data(bg[:, :, :3])
            else:
                video_writer.append_data(robot_rgb)

            written += 1

        if source_reader is not None:
            source_reader.close()
        video_writer.close()
        mask_writer.close()

        total_frames_written += written
        results.append({
            "clip_id": clip_id,
            "n_frames": written,
            "expected_frames": n_frames,
            "video_path": out_video,
            "mask_path": out_mask,
        })

        if (clip_idx + 1) % 5 == 0 or (clip_idx + 1) == len(clip_dirs):
            elapsed = time.time() - t0
            render_fps = total_frames_written / elapsed if elapsed > 0 else 0
            print(f"{tag} {clip_idx + 1}/{len(clip_dirs)} clips, "
                  f"{total_frames_written} frames, {render_fps:.1f} fps", flush=True)

    p.disconnect()
    import shutil
    tmp_dir = os.path.dirname(render_urdf)
    if tmp_dir.startswith(("/tmp", os.path.join(os.sep, "tmp"))):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"{tag} Done: {len(results)} clips, {total_frames_written} frames, {elapsed:.0f}s", flush=True)
    return results


@ray.remote
def render_worker_both(
    clip_dirs: list[str],
    output_dir: str,
    urdf_path: str,
    render_width: int,
    render_height: int,
    worker_id: int,
    short_side: int = 512,
) -> list[dict]:
    """Ray remote: 调用 _render_clips_batch。"""
    return _render_clips_batch(
        clip_dirs, output_dir, urdf_path, render_width, render_height, worker_id, short_side
    )


if __name__ == "__main__":
    main()
