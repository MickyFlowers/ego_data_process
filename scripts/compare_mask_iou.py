#!/usr/bin/env python3
"""
比较两个文件夹中所有对应 clip_id 的 mask mp4 文件的交并比 (IoU)。

对每个 clip_id，若两个文件夹都有对应的 .mp4：
  - 逐帧计算二值 mask 的 IoU = |A ∩ B| / |A ∪ B|
  - 输出该 clip 的 mean/min/max IoU、帧数

若仅一方存在，则标记为 missing。

Usage:
    python scripts/compare_mask_iou.py --dir-a outputs/ik_results/samples/mask --dir-b outputs/test_render/mask --out results.json
    python scripts/compare_mask_iou.py -a /path/to/mask_a -b /path/to/mask_b -o comparison.json
"""
import argparse
import json
import os
import sys


def _load_mask_frame(frame: "np.ndarray") -> "np.ndarray":
    """将一帧转为二值 mask (0/1)。支持灰度或 RGB（取任意通道或 max）。"""
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim == 2:
        return (arr > 127).astype(np.uint8)
    if arr.ndim == 3:
        # RGB: 任一亮通道 > 127 即视为前景
        gray = np.max(arr[:, :, :3], axis=-1)
        return (gray > 127).astype(np.uint8)
    return (arr > 127).astype(np.uint8)


def compute_frame_iou(mask_a: "np.ndarray", mask_b: "np.ndarray") -> float:
    """计算两幅二值 mask 的 IoU。"""
    import numpy as np

    a = np.asarray(mask_a).flatten().astype(bool)
    b = np.asarray(mask_b).flatten().astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compare_masks(path_a: str, path_b: str, target_size: tuple[int, int] | None = None) -> dict:
    """
    比较两个 mask mp4，返回统计信息。
    比较前先将两幅 mask resize 到相同 shape（target_size 或取两者较大尺寸），用 NEAREST。
    """
    import numpy as np

    try:
        import imageio
        from PIL import Image
    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}

    try:
        reader_a = imageio.get_reader(path_a)
        reader_b = imageio.get_reader(path_b)
    except Exception as e:
        return {"error": str(e)}

    n_a = reader_a.count_frames()
    n_b = reader_b.count_frames()
    n_frames = min(n_a, n_b)
    if n_frames == 0:
        reader_a.close()
        reader_b.close()
        return {"error": "No frames", "n_frames_a": n_a, "n_frames_b": n_b}

    ious = []
    for i in range(n_frames):
        try:
            fa = np.asarray(reader_a.get_data(i))
            fb = np.asarray(reader_b.get_data(i))
        except Exception as e:
            ious.append(0.0)
            continue

        ma = _load_mask_frame(fa)
        mb = _load_mask_frame(fb)

        # 统一 resize 到相同 shape 再比较
        if target_size is not None:
            h, w = target_size
        else:
            h = max(ma.shape[0], mb.shape[0])
            w = max(ma.shape[1], mb.shape[1])
        if ma.shape != (h, w):
            ma = np.asarray(
                Image.fromarray(ma.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            ) > 127
        if mb.shape != (h, w):
            mb = np.asarray(
                Image.fromarray(mb.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            ) > 127

        iou = compute_frame_iou(ma, mb)
        ious.append(iou)

    reader_a.close()
    reader_b.close()

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    min_iou = min(ious) if ious else 0.0
    max_iou = max(ious) if ious else 0.0

    return {
        "mean_iou": round(mean_iou, 6),
        "min_iou": round(min_iou, 6),
        "max_iou": round(max_iou, 6),
        "n_frames": n_frames,
    }


def render_mask_diff_video(
    path_a: str,
    path_b: str,
    out_path: str,
    target_size: tuple[int, int],
    fps: float = 30.0,
) -> str | None:
    """
    绘制两个 mask 的差异可视化 mp4。
    颜色：黑=都无, 绿=都有(相同), 红=仅A有, 蓝=仅B有
    """
    import numpy as np
    try:
        import imageio
        from PIL import Image
    except ImportError:
        return None

    try:
        reader_a = imageio.get_reader(path_a)
        reader_b = imageio.get_reader(path_b)
    except Exception:
        return None

    n_frames = min(reader_a.count_frames(), reader_b.count_frames())
    h, w = target_size
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", macro_block_size=1)

    for i in range(n_frames):
        fa = np.asarray(reader_a.get_data(i))
        fb = np.asarray(reader_b.get_data(i))
        ma = _load_mask_frame(fa)
        mb = _load_mask_frame(fb)
        if ma.shape != (h, w):
            ma = (np.asarray(Image.fromarray(ma.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127).astype(np.uint8)
        if mb.shape != (h, w):
            mb = (np.asarray(Image.fromarray(mb.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127).astype(np.uint8)

        # 黑=都无, 绿=都有, 红=仅A, 蓝=仅B
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        both = np.logical_and(ma, mb)
        only_a = np.logical_and(ma, np.logical_not(mb))
        only_b = np.logical_and(mb, np.logical_not(ma))
        vis[both] = [0, 255, 0]    # 绿
        vis[only_a] = [255, 0, 0]  # 红
        vis[only_b] = [0, 0, 255]  # 蓝

        writer.append_data(vis)

    reader_a.close()
    reader_b.close()
    writer.close()
    return out_path


def collect_clip_ids(dir_path: str) -> set[str]:
    """收集目录下所有 .mp4 的 clip_id（不含 .tmp.mp4）。"""
    if not os.path.isdir(dir_path):
        return set()
    ids = set()
    for f in os.listdir(dir_path):
        if f.endswith(".mp4") and not f.endswith(".tmp.mp4"):
            ids.add(f[:-4])
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="比较两个文件夹中对应 clip_id 的 mask mp4 的 IoU"
    )
    parser.add_argument(
        "-a", "--dir-a",
        type=str,
        required=True,
        help="第一个 mask 目录（如 samples/mask 或 video/mask）",
    )
    parser.add_argument(
        "-b", "--dir-b",
        type=str,
        required=True,
        help="第二个 mask 目录",
    )
    parser.add_argument(
        "-o", "--out",
        type=str,
        default="mask_iou_comparison.json",
        help="输出 JSON 路径",
    )
    parser.add_argument(
        "--target-size",
        type=str,
        default=None,
        help="统一 resize 到的尺寸（默认：512x512）",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="输出 debug 可视化 mp4：绿色=相同，红色=仅A有，蓝色=仅B有",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="mask_iou_vis",
        help="--vis 时输出 mp4 的目录（默认：mask_iou_vis）",
    )
    args = parser.parse_args()

    dir_a = os.path.abspath(args.dir_a)
    dir_b = os.path.abspath(args.dir_b)

    if not os.path.isdir(dir_a):
        print(f"[Error] Not a directory: {dir_a}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(dir_b):
        print(f"[Error] Not a directory: {dir_b}", file=sys.stderr)
        sys.exit(1)

    ids_a = collect_clip_ids(dir_a)
    ids_b = collect_clip_ids(dir_b)
    common = ids_a & ids_b
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    print(f"[Compare] Dir A: {dir_a} ({len(ids_a)} clips)")
    print(f"[Compare] Dir B: {dir_b} ({len(ids_b)} clips)")
    print(f"[Compare] Common: {len(common)}, only in A: {len(only_a)}, only in B: {len(only_b)}")

    results = {}
    target_size = (512, 512)  # 默认 512x512
    if args.target_size:
        parts = args.target_size.lower().split("x")
        if len(parts) == 2:
            try:
                target_size = (int(parts[0]), int(parts[1]))
            except ValueError:
                pass

    vis_dir = os.path.abspath(args.vis_dir) if args.vis else None

    for cid in sorted(common):
        path_a = os.path.join(dir_a, f"{cid}.mp4")
        path_b = os.path.join(dir_b, f"{cid}.mp4")
        r = compare_masks(path_a, path_b, target_size=target_size)
        results[cid] = r
        if "error" in r:
            print(f"  {cid}: ERROR - {r['error']}")
        else:
            print(f"  {cid}: mean={r['mean_iou']:.4f} min={r['min_iou']:.4f} max={r['max_iou']:.4f} (n={r['n_frames']})")
        if args.vis and "error" not in r and vis_dir:
            out_mp4 = os.path.join(vis_dir, f"{cid}.mp4")
            try:
                render_mask_diff_video(path_a, path_b, out_mp4, target_size, fps=30.0)
                print(f"    -> vis: {out_mp4}")
            except Exception as e:
                print(f"    -> vis failed: {e}")

    for cid in sorted(only_a):
        results[cid] = {"status": "missing_in_b", "path_a": os.path.join(dir_a, f"{cid}.mp4")}
    for cid in sorted(only_b):
        results[cid] = {"status": "missing_in_a", "path_b": os.path.join(dir_b, f"{cid}.mp4")}

    valid_ious = [results[c]["mean_iou"] for c in common if "mean_iou" in results.get(c, {})]
    summary = {
        "dir_a": dir_a,
        "dir_b": dir_b,
        "n_common": len(common),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "mean_iou_over_common": round(sum(valid_ious) / len(valid_ious), 6) if valid_ious else 0,
    }
    output = {
        "summary": summary,
        "clips": results,
    }

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[Compare] Results written to {out_path}")


if __name__ == "__main__":
    main()
