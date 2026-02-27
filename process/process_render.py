#!/usr/bin/env python3
from __future__ import annotations

import src.compat  # noqa: F401

"""
PyBullet batch render with EGL: load IK results and render overlay videos.

Architecture:
  - Multi-CPU via Ray: clips are distributed round-robin across workers.
  - Each worker initialises PyBullet (DIRECT + EGL) once and reuses the
    same physics server for all its assigned clips (no per-clip overhead).
  - Rendering uses p.getCameraImage with the same camera matrices as
    process_ik (view/projection built from camera_extrinsics and
    camera_intrinsics stored in meta.json).

Input layout (from process_ik):
    {input_dir}/data/{clip_id}/ with meta.json + joint_trajectory.json

Output layout:
    {output_dir}/video/{clip_id}.mp4

Usage:
    python -m process.process_render --input-dir outputs/ik --output-dir outputs/render
    python -m process.process_render --input-dir outputs/ik --output-dir outputs/render \\
        --sample-ratio 0.1 --num-cpus 8
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import ray

RENDER_RESOLUTION = (640, 480)
MESH_RGBA = (0.055, 0.055, 0.055, 1.0)


# ---------------------------------------------------------------------------
# Work around PyBullet bug: duplicate OBJ paths in one URDF get corrupted.
# We rewrite the URDF so every visual mesh has a unique absolute file path.
# ---------------------------------------------------------------------------

def _prepare_render_urdf(urdf_path: str) -> str:
    """Return a temp URDF where every visual mesh has a unique file path.

    PyBullet corrupts the visual data of the second instance when the same
    OBJ file is referenced by multiple links.  We detect duplicates, copy
    the OBJ (+ its MTL) into a temp dir with a unique name, and rewrite the
    URDF mesh references to use absolute paths.
    """
    import shutil
    import tempfile
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    pkg_root = os.path.dirname(os.path.dirname(urdf_dir))  # assets/

    tmp_dir = tempfile.mkdtemp(prefix="pb_render_")

    def _resolve(filename: str) -> str:
        if filename.startswith("package://"):
            return os.path.join(pkg_root, filename[len("package://"):])
        if os.path.isabs(filename):
            return filename
        return os.path.join(urdf_dir, filename)

    def _mtl_refs(obj_path: str) -> list[str]:
        """Parse mtllib lines near the top of an OBJ file."""
        refs: list[str] = []
        try:
            with open(obj_path, "r") as fobj:
                for line in fobj:
                    stripped = line.strip()
                    if stripped.startswith("mtllib "):
                        refs.append(stripped[7:].strip())
                    if stripped.startswith("v ") or stripped.startswith("f "):
                        break
        except OSError:
            pass
        return refs

    # -- visual meshes: deduplicate -------------------------------------------
    vis_items: list[tuple] = []  # (Element, resolved_abs_path)
    for link_el in root.iter("link"):
        for vis in link_el.iter("visual"):
            for geom in vis.iter("geometry"):
                mesh_el = geom.find("mesh")
                if mesh_el is not None:
                    vis_items.append((mesh_el, _resolve(mesh_el.get("filename", ""))))

    seen: dict[str, int] = {}
    for mesh_el, resolved in vis_items:
        idx = seen.get(resolved, 0)
        seen[resolved] = idx + 1

        if idx == 0:
            mesh_el.set("filename", resolved)
        else:
            base = os.path.basename(resolved)
            stem, ext = os.path.splitext(base)
            dup_name = f"{stem}_dup{idx}{ext}"
            dup_path = os.path.join(tmp_dir, dup_name)
            shutil.copy2(resolved, dup_path)

            src_dir = os.path.dirname(resolved)
            for mtl_name in _mtl_refs(resolved):
                mtl_src = os.path.join(src_dir, mtl_name)
                mtl_dst = os.path.join(tmp_dir, mtl_name)
                if os.path.isfile(mtl_src) and not os.path.exists(mtl_dst):
                    shutil.copy2(mtl_src, mtl_dst)

            mesh_el.set("filename", dup_path)

    # -- collision meshes: just make paths absolute ---------------------------
    for link_el in root.iter("link"):
        for col in link_el.iter("collision"):
            for geom in col.iter("geometry"):
                mesh_el = geom.find("mesh")
                if mesh_el is not None:
                    mesh_el.set("filename", _resolve(mesh_el.get("filename", "")))

    out_urdf = os.path.join(tmp_dir, "render.urdf")
    tree.write(out_urdf, xml_declaration=True, encoding="utf-8")
    return out_urdf


# ---------------------------------------------------------------------------
# Ray remote worker
# ---------------------------------------------------------------------------

@ray.remote
def render_worker(
    clip_dirs: list[str],
    output_dir: str,
    urdf_path: str,
    render_width: int,
    render_height: int,
    worker_id: int,
) -> list[dict]:
    """Ray remote: init PyBullet once with EGL, then render all assigned clips."""
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

    # ---- PyBullet init (once per worker) ----
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
    print(f"{tag} PyBullet ready (EGL={'yes' if egl_loaded else 'no'}), "
          f"{len(clip_dirs)} clips to render", flush=True)

    # ---- Process clips ----
    results: list[dict] = []
    t0 = time.time()
    total_frames_written = 0

    for clip_idx, clip_dir in enumerate(clip_dirs):
        clip_path = Path(clip_dir)
        clip_id = clip_path.name
        meta_file = clip_path / "meta.json"
        traj_file = clip_path / "joint_trajectory.json"

        if not meta_file.exists() or not traj_file.exists():
            results.append({"clip_id": clip_id,
                            "error": "missing meta.json or joint_trajectory.json"})
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

        # ---- Camera ----
        camera_intrinsics = np.array(meta["camera_intrinsics"], dtype=np.float64)
        camera_extrinsics = np.array(meta["camera_extrinsics"], dtype=np.float64)
        img_size = meta.get("img_size", [render_width, render_height])

        render_K = scale_intrinsics(camera_intrinsics, img_size,
                                    [render_width, render_height])
        view_matrix = build_view_matrix_from_camera_matrix(camera_extrinsics)
        proj_matrix = build_projection_matrix_from_intrinsics(
            render_K, render_width, render_height)

        # ---- Source video for overlay ----
        video_path = meta.get("video_path")
        source_reader = None
        if video_path and os.path.isfile(video_path):
            try:
                source_reader = imageio.get_reader(video_path)
            except Exception:
                source_reader = None

        # ---- Output writer ----
        out_path = os.path.join(output_dir, "video", f"{clip_id}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer = imageio.get_writer(
            out_path, fps=fps, codec="libx264", macro_block_size=1,
            output_params=["-preset", "ultrafast"],
        )

        written = 0
        need_resize: bool | None = None

        for fi in range(n_frames):
            lq = left_traj[fi]
            rq = right_traj[fi]

            # Arm joints (first 6) + grippers (7th)
            set_arm_joints(robot_info, lq[:6], rq[:6])
            lg = float(lq[6]) if len(lq) > 6 else 0.0
            rg = float(rq[6]) if len(rq) > 6 else 0.0
            set_gripper(robot_info, lg, rg)

            # Multi-light render: simulate multiple light sources by
            # rendering several passes with different directions and blending.
            light_configs = [
                # (direction,       weight, shadow)  — primary key light
                ([0.5,  0.0,  1.5], 0.45,   1),
                # fill light from the left
                ([-1.0, 0.5,  0.8], 0.30,   0),
                # back/rim light
                ([0.0, -0.5,  1.0], 0.25,   0),
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

            # Read source frame
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
                        Image.fromarray(robot_rgb).resize(
                            (bg_w, bg_h), Image.BILINEAR))
                    mask = np.asarray(
                        Image.fromarray(mask.astype(np.uint8) * 255).resize(
                            (bg_w, bg_h), Image.NEAREST)) > 127

                bg[mask] = robot_rgb[mask]

                if (bg_w, bg_h) != (render_width, render_height):
                    bg = np.asarray(
                        Image.fromarray(bg).resize(
                            (render_width, render_height), Image.BILINEAR))
                writer.append_data(bg[:, :, :3])
            else:
                writer.append_data(robot_rgb)

            written += 1

        if source_reader is not None:
            source_reader.close()
        writer.close()

        total_frames_written += written
        results.append({
            "clip_id": clip_id,
            "n_frames": written,
            "expected_frames": n_frames,
            "output_path": out_path,
        })

        if (clip_idx + 1) % 5 == 0 or (clip_idx + 1) == len(clip_dirs):
            elapsed = time.time() - t0
            render_fps = total_frames_written / elapsed if elapsed > 0 else 0
            print(f"{tag} {clip_idx + 1}/{len(clip_dirs)} clips, "
                  f"{total_frames_written} frames, {render_fps:.1f} fps",
                  flush=True)

    p.disconnect()

    import shutil
    tmp_dir = os.path.dirname(render_urdf)
    if tmp_dir.startswith(("/tmp", os.path.join(os.sep, "tmp"))):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"{tag} Done: {len(results)} clips, "
          f"{total_frames_written} frames, {elapsed:.0f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main: discover clips, Ray init, distribute, launch, collect
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PyBullet batch render: multi-CPU (Ray) + EGL rendering"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="IK output dir containing data/{clip_id}/ subdirs")
    parser.add_argument("--output-dir", type=str, default="outputs/render",
                        help="Output directory for rendered videos")
    parser.add_argument("--urdf-path", type=str,
                        default="./assets/aloha_new_description/urdf/dual_piper.urdf",
                        help="URDF path for the robot")
    parser.add_argument("--sample-ratio", type=float, default=1.0,
                        help="Ratio of clips to render (0~1, default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--width", type=int, default=RENDER_RESOLUTION[0],
                        help="Render width")
    parser.add_argument("--height", type=int, default=RENDER_RESOLUTION[1],
                        help="Render height")
    parser.add_argument("--num-cpus", type=int, default=None,
                        help="Total CPUs for Ray (default: all)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Total GPUs for Ray (default: all)")
    args = parser.parse_args()

    # ---- Discover clips ----
    data_dir = os.path.join(args.input_dir, "data")
    if not os.path.isdir(data_dir):
        print(f"[Render] No data/ directory found in {args.input_dir}")
        sys.exit(1)

    clip_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and os.path.exists(os.path.join(data_dir, d, "meta.json"))
        and os.path.exists(os.path.join(data_dir, d, "joint_trajectory.json"))
    ])
    if not clip_dirs:
        print(f"[Render] No valid clip dirs in {data_dir}")
        sys.exit(1)

    n_total = len(clip_dirs)
    if args.sample_ratio < 1.0:
        n_sample = max(1, math.ceil(n_total * args.sample_ratio))
        random.seed(args.seed)
        indices = sorted(random.sample(range(n_total), min(n_sample, n_total)))
        clip_dirs = [clip_dirs[i] for i in indices]
    n_render = len(clip_dirs)

    # ---- Ray init ----
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

    # Round-robin distribution
    worker_clips: list[list[str]] = [[] for _ in range(num_workers)]
    for i, cd in enumerate(clip_dirs):
        worker_clips[i % num_workers].append(cd)
    active: list[tuple[int, list[str]]] = [
        (w, clips) for w, clips in enumerate(worker_clips) if clips
    ]

    os.makedirs(os.path.join(args.output_dir, "video"), exist_ok=True)

    print(f"[Render] {n_render}/{n_total} clips, {len(active)} workers "
          f"({total_cpus} CPUs, {total_gpus} GPUs)")
    for w, clips in active:
        print(f"  Worker {w}: {len(clips)} clips")

    t_start = time.time()

    # ---- Launch workers ----
    task_opts: dict = {"num_cpus": cpus_per_worker}
    if gpus_per_worker > 0:
        task_opts["num_gpus"] = gpus_per_worker

    future_to_worker: dict = {}
    pending_futures: list = []
    for w, clips in active:
        fut = render_worker.options(**task_opts).remote(
            clip_dirs=clips,
            output_dir=args.output_dir,
            urdf_path=os.path.abspath(args.urdf_path),
            render_width=args.width,
            render_height=args.height,
            worker_id=w,
        )
        future_to_worker[fut] = (w, clips)
        pending_futures.append(fut)

    # ---- Collect results ----
    all_results: list[dict] = []

    while pending_futures:
        done, pending_futures = ray.wait(pending_futures, num_returns=1)
        for fut in done:
            w, clips = future_to_worker[fut]
            try:
                worker_results = ray.get(fut)
                all_results.extend(worker_results)
                n_ok = sum(1 for r in worker_results if "error" not in r)
                print(f"[Render] Worker {w}: {n_ok}/{len(worker_results)} ok  "
                      f"({len(all_results)}/{n_render} total)", flush=True)
            except Exception as e:
                print(f"[Render] Worker {w} failed: {e}",
                      file=sys.stderr, flush=True)
                for cd in clips:
                    all_results.append({
                        "clip_id": os.path.basename(cd),
                        "error": f"Worker {w} failed: {e}",
                    })

    # ---- Summary ----
    total_time = time.time() - t_start
    success = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]
    total_frames = sum(r.get("n_frames", 0) for r in success)

    summary = {
        "total": n_total,
        "rendered": n_render,
        "num_workers": len(active),
        "success": len(success),
        "failed": len(failed),
        "failed_clips": [{"clip_id": r["clip_id"], "error": r["error"]}
                         for r in failed],
        "total_time": total_time,
        "total_frames": total_frames,
    }
    summary_path = os.path.join(args.output_dir, "render_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Render] Done: {len(success)}/{n_render} ok, {len(failed)} failed | "
          f"{total_time:.0f}s, {total_frames} frames -> {summary_path}")

    if failed:
        print("\n[Render] Failed clips:", file=sys.stderr)
        for r in failed:
            print(f"  {r['clip_id']}: {r['error']}", file=sys.stderr)

    ray.shutdown()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
