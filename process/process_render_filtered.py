#!/usr/bin/env python3
"""
PyBullet 渲染：从 filter_result.json 读取 clip 列表，加载已有 joint_trajectory，渲染得到 mask 和 video。

与 process_ik 的 render 功能一致，但不做 IK，直接使用已有的 joint_trajectory.json。
数据目录结构：data_dir/{clip_id}/ 含 meta.json + joint_trajectory.json。

Output:
    {output_dir}/video/{clip_id}.mp4  (机器人叠加重叠原视频，或仅机器人)
    {output_dir}/mask/{clip_id}.mp4   (二值分割 mask)

Usage:
    python -m process.process_render_filtered --filter-result /path/to/filter_result.json --data-dir /path/to/ik/data --output-dir outputs/render
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
        "--width",
        type=int,
        default=RENDER_RESOLUTION[0],
        help="渲染宽度",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=RENDER_RESOLUTION[1],
        help="渲染高度",
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

    # ---- Ray 初始化 ----
    ray_kwargs: dict = {}
    if args.num_cpus is not None:
        ray_kwargs["num_cpus"] = args.num_cpus
    if args.num_gpus is not None:
        ray_kwargs["num_gpus"] = args.num_gpus
    ray.init(**ray_kwargs)

    cluster = ray.cluster_resources()
    total_cpus = int(cluster.get("CPU", 1))
    total_gpus = cluster.get("GPU", 0)
    num_workers = max(1, min(total_cpus, n_render))
    cpus_per_worker = max(0.5, total_cpus / num_workers)
    gpus_per_worker = total_gpus / num_workers if total_gpus > 0 else 0

    worker_clips: list[list[str]] = [[] for _ in range(num_workers)]
    for i, cd in enumerate(clip_dirs):
        worker_clips[i % num_workers].append(cd)
    active: list[tuple[int, list[str]]] = [
        (w, clips) for w, clips in enumerate(worker_clips) if clips
    ]

    print(f"[RenderFiltered] {num_workers} workers, {total_cpus} CPUs")
    for w, clips in active:
        print(f"  Worker {w}: {len(clips)} clips")

    # ---- process_render 的 render_worker 只支持 seg_only 或 video，需要同时输出 mask+video
    # 使用 seg_only=False 得到 overlay video，但 process_render 不写 mask。
    # 因此我们改为调用两次：一次 seg_only（写 mask），一次非 seg_only（写 video）。
    # 更优：扩展 render_worker 支持 output_both。这里为简化，先 run 两次。
    # 实际上单次 run 可同时写两个 writer，但需改 process_render。
    # 方案：复制 render_worker 逻辑到本地，增加 mask_writer，一次循环写两份。

    # 使用自定义 worker，输出 video + mask
    task_opts: dict = {"num_cpus": cpus_per_worker}
    if gpus_per_worker > 0:
        task_opts["num_gpus"] = gpus_per_worker

    t_start = time.time()
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
        )
        future_to_worker[fut] = (w, clips)
        pending_futures.append(fut)

    all_results: list[dict] = []
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
                    all_results.append({"clip_id": os.path.basename(cd), "error": f"Worker {w} failed: {e}"})

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

    ray.shutdown()
    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# 自定义 worker：同时输出 video 和 mask（与 process_render 逻辑一致）
# ---------------------------------------------------------------------------

@ray.remote
def render_worker_both(
    clip_dirs: list[str],
    output_dir: str,
    urdf_path: str,
    render_width: int,
    render_height: int,
    worker_id: int,
) -> list[dict]:
    """PyBullet 渲染，同时输出 video/ 和 mask/。"""
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

        render_K = scale_intrinsics(camera_intrinsics, img_size, [render_width, render_height])
        view_matrix = build_view_matrix_from_camera_matrix(camera_extrinsics)
        proj_matrix = build_projection_matrix_from_intrinsics(
            render_K, render_width, render_height
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
            accum = np.zeros((render_height, render_width, 3), dtype=np.float64)
            seg_out = None
            for l_dir, l_weight, l_shadow in light_configs:
                _, _, _rgba, _, _seg = p.getCameraImage(
                    render_width, render_height,
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
                    (render_height, render_width, 4))[:, :, :3]
                accum += _rgb.astype(np.float64) * l_weight
                if seg_out is None:
                    seg_out = _seg

            robot_rgb = np.clip(accum, 0, 255).astype(np.uint8)
            mask = np.asarray(seg_out).reshape((render_height, render_width)) >= 0

            # 写 mask
            mask_uint8 = (mask.astype(np.uint8) * 255)
            mask_writer.append_data(np.stack([mask_uint8] * 3, axis=-1))

            # 写 video（overlay 或仅机器人）
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
                    need_resize = (render_width, render_height) != (bg_w, bg_h)
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
                if (bg_w, bg_h) != (render_width, render_height):
                    bg = np.asarray(
                        Image.fromarray(bg).resize((render_width, render_height), Image.BILINEAR)
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


if __name__ == "__main__":
    main()
