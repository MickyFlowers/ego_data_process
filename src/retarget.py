# Python 3.11+ compat (inspect.getargspec)
import src.compat  # noqa: F401

import json

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as Rscipy
from typing import Dict, List, Tuple

from manopth.manolayer import ManoLayer

# Saved video resolution for both retarget replay and (when aligned) IK overlay; (width, height)
SAVED_VIDEO_RESOLUTION = (640, 480)


def scale_intrinsics(K: np.ndarray, src_size: list, dst_size: list) -> np.ndarray:
    """Scale 3x3 intrinsics from src_size to dst_size. Both sizes as [width, height]."""
    K = np.asarray(K, dtype=np.float64).copy()
    sx = dst_size[0] / max(1, src_size[0])
    sy = dst_size[1] / max(1, src_size[1])
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K.astype(np.float32)

# Gripper (pinch) mapping: absolute scale + nonlinear curve (more sensitivity when hand is nearly closed).
# Thumb–index distance in meters.
PINCH_DIST_CLOSED_M = 0.05
PINCH_DIST_OPEN_M = 0.10
# Power exponent for nonlinear map: output = linear^gamma. gamma < 1 => small distance changes enlarge gripper more.
PINCH_GAMMA = 0.5


def _to_json_serializable(obj):
    """Recursively convert Tensors and ndarrays to lists for JSON."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


class KeypointsExtractor:
    def __init__(self, model_dir: Path, device: torch.device | None = None,
                 flat_hand_mean: bool = True, fix_shapedirs: bool = True):
        self.device = device or self._select_device()
        self.model_dir = Path(model_dir)
        self.flat_hand_mean = flat_hand_mean
        self.fix_shapedirs = fix_shapedirs
        self.left_model = self._build_mano(is_right=False).to(self.device)
        self.right_model = self._build_mano(is_right=True).to(self.device)

    def load_data(self, data_path: Path) -> Dict:
        map_loc = "mps" if self.device.type == "mps" else "cpu"
        data = torch.load(data_path, map_location=map_loc, weights_only=False)
        return self._dict_to_device(data)

    def extract(self, data: Dict) -> Dict:
        data = self._dict_to_device(data)

        left_joints, left_valid = self._forward_model(self.left_model, data["left_hand"])
        right_joints, right_valid = self._forward_model(self.right_model, data["right_hand"])
        traj, img_focal, img_center, scale = self._parse_camera_data(data)

        return {
            "left": {"keypoints3d": left_joints, "valid": left_valid},
            "right": {"keypoints3d": right_joints, "valid": right_valid},
            "camera": {
                "traj": traj,
                "img_focal": img_focal,
                "img_center": img_center,
                "scale": scale,
            },
        }

    def _select_device(self) -> torch.device:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def _dict_to_device(self, data: Dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device)
            elif isinstance(value, dict):
                data[key] = self._dict_to_device(value)
            elif isinstance(value, np.ndarray):
                if self.device.type == "mps" and value.dtype == np.float64:
                    value = value.astype(np.float32)
                data[key] = torch.from_numpy(value).to(self.device)
            elif isinstance(value, (int, float, list, tuple)):
                data[key] = torch.tensor(value).to(self.device)
        return data

    def _build_mano(self, is_right: bool):
        side = "right" if is_right else "left"
        model = ManoLayer(
            mano_root=str(self.model_dir),
            use_pca=False,
            flat_hand_mean=self.flat_hand_mean,
            side=side,
        )
        if not is_right and self.fix_shapedirs:
            with torch.no_grad():
                if hasattr(model, "th_shapedirs"):
                    model.th_shapedirs[:, 0, :] *= -1
        return model

    def _forward_model(self, model, hand_data: Dict):
        params = hand_data["mano_params"]
        pred_valid = hand_data["pred_valid"]
        n = len(pred_valid)
        betas = params["betas"].view(n, 10)
        global_orient = params["global_orient"].view(n, 3)
        hand_pose = params["hand_pose"].view(n, 45)
        transl = params["transl"].view(n, 3)
        th_pose_coeffs = torch.cat([global_orient, hand_pose], dim=1)
        _, joints = model(th_pose_coeffs, th_betas=betas, th_trans=transl)
        return joints / 1000.0, pred_valid

    def _parse_camera_data(self, data: Dict):
        camera_data = self._dict_to_device(data["slam_data"])
        traj = camera_data["traj"]
        img_focal = camera_data["img_focal"]
        img_center = camera_data["img_center"]
        scale = camera_data["scale"]
        return traj, img_focal, img_center, scale


class CameraProjector:
    def __init__(self, device: torch.device):
        self.device = device

    def build_intrinsic(self, img_focal: torch.Tensor, img_center: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [[img_focal, 0, img_center[0]], [0, img_focal, img_center[1]], [0, 0, 1]],
            device=self.device,
        )

    def apply_scale(self, traj: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        traj = traj.clone()
        traj[:, :3] = traj[:, :3] * scale
        return traj

    def world_to_camera(self, joints: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        t = traj[:, :3]
        q = traj[:, 3:]
        R = self._quat_to_rotmat(q)
        return torch.bmm(joints - t.unsqueeze(1), R)

    def project(self, points: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
        pts_h = torch.matmul(points, intrinsic.T)
        z = pts_h[..., 2:3].clamp(min=1e-6)
        return pts_h[..., :2] / z

    def project_np(self, points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        pts_h = points @ intrinsic.T
        z = np.clip(pts_h[..., 2:3], 1e-6, None)
        return pts_h[..., :2] / z

    def _quat_to_rotmat(self, q: torch.Tensor) -> torch.Tensor:
        qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack(
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qx * qw),
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx**2 + qy**2),
            ],
            dim=1,
        ).view(-1, 3, 3)
        return R


class EEFRetargeter:
    def compute_axes(self, joints_cam: np.ndarray, method: str,
                     axis_len: float, axis_scale: float,
                     side: str = "right") -> np.ndarray:
        idx = self._get_joint_indices(joints_cam.shape[1])
        axes = np.zeros((joints_cam.shape[0], 4, 3), dtype=np.float32)
        prev_y = None
        for i in range(joints_cam.shape[0]):
            origin, R = self._compute_eef_frame(joints_cam[i], method, idx, side=side, prev_y=prev_y)
            prev_y = R[:, 1].copy()
            length = axis_len if axis_len > 0 else self._auto_axis_len(joints_cam[i], idx, axis_scale)
            axes[i, 0] = origin
            axes[i, 1] = origin + length * R[:, 0]
            axes[i, 2] = origin + length * R[:, 1]
            axes[i, 3] = origin + length * R[:, 2]
        return axes

    def compute_poses(self, joints_cam: np.ndarray, method: str,
                      side: str = "right") -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return per-frame (origin, axis_angle) list; no visualization params."""
        idx = self._get_joint_indices(joints_cam.shape[1])
        poses = []
        prev_y = None
        for i in range(joints_cam.shape[0]):
            origin, R = self._compute_eef_frame(joints_cam[i], method, idx, side=side, prev_y=prev_y)
            prev_y = R[:, 1].copy()
            axis_angle = Rscipy.from_matrix(R).as_rotvec()
            poses.append((origin, axis_angle))
        return poses

    def compute_pinch_norm(self, joints_cam: np.ndarray) -> np.ndarray:
        """Map thumb–index distance to [0,1]: absolute scale + power curve.
        When distance is small (nearly closed), a small increase in distance gives a larger
        increase in output so the gripper opens more noticeably in that range.
        """
        thumb_idx, index_idx = self._get_thumb_index_indices(joints_cam.shape[1])
        n = joints_cam.shape[0]
        dists = np.zeros(n, dtype=np.float32)
        if n < 2:
            return np.zeros(n, dtype=np.float32)
        for i in range(n):
            thumb_center = joints_cam[i, thumb_idx].mean(axis=0)
            index_center = joints_cam[i, index_idx].mean(axis=0)
            dists[i] = np.linalg.norm(thumb_center - index_center)
        d_closed = PINCH_DIST_CLOSED_M
        d_open = PINCH_DIST_OPEN_M
        span = d_open - d_closed
        if span <= 0:
            return np.zeros(n, dtype=np.float32)
        linear = np.clip((dists - d_closed) / span, 0.0, 1.0)
        # Nonlinear: linear^gamma with gamma < 1 => steeper at small distance (more gripper change)
        normed = np.power(linear, PINCH_GAMMA)
        return normed.astype(np.float32)

    def _normalize(self, v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < eps:
            return np.zeros_like(v)
        return v / n

    def _get_joint_indices(self, joint_count: int) -> Dict[str, int]:
        if joint_count >= 21:
            return {
                "wrist": 0,
                "thumb_mcp": 1,
                "thumb_tip": 4,
                "index_mcp": 5,
                "index_tip": 8,
                "middle_mcp": 9,
                "middle_tip": 12,
                "ring_mcp": 13,
                "ring_tip": 16,
                "pinky_mcp": 17,
                "pinky_tip": 20,
            }
        if joint_count == 16:
            return {
                "wrist": 0,
                "index_mcp": 1,
                "index_tip": 3,
                "middle_mcp": 4,
                "middle_tip": 6,
                "ring_mcp": 7,
                "ring_tip": 9,
                "pinky_mcp": 10,
                "pinky_tip": 12,
                "thumb_mcp": 13,
                "thumb_tip": 15,
            }
        idxs = np.linspace(0, max(0, joint_count - 1), 6).round().astype(int)
        return {
            "wrist": int(idxs[0]),
            "index_mcp": int(idxs[1]),
            "index_tip": int(idxs[2]),
            "thumb_mcp": int(idxs[3]),
            "thumb_tip": int(idxs[4]),
            "pinky_mcp": int(idxs[5]),
            "pinky_tip": int(idxs[5]),
        }

    def _get_thumb_index_indices(self, joint_count: int) -> Tuple[List[int], List[int]]:
        """Return (thumb_list, index_list) of joint indices."""
        if joint_count >= 21:
            return [1, 2, 3, 4], [5, 6, 7, 8]
        if joint_count == 16:
            return [13, 14, 15], [1, 2, 3]
        idx = self._get_joint_indices(joint_count)
        return [idx["thumb_mcp"], idx["thumb_tip"]], [idx["index_mcp"], idx["index_tip"]]

    def _compute_palm_normal(self, joints: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
        w = joints[idx["wrist"]]
        i = joints[idx["index_mcp"]]
        p = joints[idx["pinky_mcp"]]
        n = np.cross(i - w, p - w)
        n = self._normalize(n)
        if np.allclose(n, 0):
            i2 = joints[idx["index_tip"]]
            p2 = joints[idx["pinky_tip"]]
            n = self._normalize(np.cross(i2 - w, p2 - w))
        return n

    def _auto_axis_len(self, joints: np.ndarray, idx: Dict[str, int], scale: float) -> float:
        w = joints[idx["wrist"]]
        i = joints[idx["index_mcp"]]
        p = joints[idx["pinky_mcp"]]
        width = np.linalg.norm(i - p)
        if not np.isfinite(width) or width < 1e-6:
            width = np.linalg.norm(i - w)
        if not np.isfinite(width) or width < 1e-6:
            width = 0.05
        return float(width * scale)

    def _compute_eef_frame(self, joints: np.ndarray, method: str, idx: Dict[str, int],
                           side: str = "right", prev_y: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        w = joints[idx["wrist"]]
        n = self._compute_palm_normal(joints, idx)

        if method == "pinch":
            it = joints[idx["index_tip"]]
            tt = joints[idx["thumb_tip"]]
            origin = 0.5 * (it + tt)
            x_axis = self._normalize(tt - it)
            if np.allclose(x_axis, 0):
                x_axis = self._normalize(joints[idx["index_mcp"]] - w)
            z_axis = n
            y_axis = self._normalize(np.cross(z_axis, x_axis))
            x_axis = self._normalize(np.cross(y_axis, z_axis))
        elif method == "pinch_plane":
            thumb_idx, index_idx = self._get_thumb_index_indices(joints.shape[0])
            thumb_pts = joints[thumb_idx]
            index_pts = joints[index_idx]
            thumb_center = thumb_pts.mean(axis=0)
            index_center = index_pts.mean(axis=0)
            origin = 0.5 * (thumb_center + index_center)

            n_near = min(2, len(thumb_idx), len(index_idx))
            y_sum = np.zeros(3, dtype=np.float64)
            for k in range(n_near):
                vt = joints[thumb_idx[k]] - w
                vi = joints[index_idx[k]] - w
                if side == "left":
                    y_sum += np.cross(vt, vi)
                else:
                    y_sum += np.cross(vi, vt)
            y_axis = self._normalize(y_sum)

            if side == "right":
                x_raw = thumb_center - index_center
            else:
                x_raw = index_center - thumb_center
            x_raw = x_raw - np.dot(x_raw, y_axis) * y_axis
            x_axis = self._normalize(x_raw)
            if np.linalg.norm(x_axis) < 1e-8:
                x_axis = self._normalize(joints[idx["thumb_tip"]] - joints[idx["index_tip"]])
                x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
                x_axis = self._normalize(x_axis)
            z_axis = self._normalize(np.cross(x_axis, y_axis))
        elif method == "palm":
            origin = w
            x_axis = self._normalize(joints[idx["index_mcp"]] - w)
            y_axis = self._normalize(joints[idx["pinky_mcp"]] - w)
            z_axis = self._normalize(np.cross(x_axis, y_axis))
            y_axis = self._normalize(np.cross(z_axis, x_axis))
            x_axis = self._normalize(np.cross(y_axis, z_axis))
        elif method == "pca":
            origin = joints.mean(axis=0)
            centered = joints - origin
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            x_axis = self._normalize(vh[0])
            y_axis = self._normalize(vh[1])
            z_axis = self._normalize(vh[2])
            if np.dot(z_axis, n) < 0:
                z_axis = -z_axis
            y_axis = self._normalize(np.cross(z_axis, x_axis))
            x_axis = self._normalize(np.cross(y_axis, z_axis))
        else:
            origin = w
            x_axis = self._normalize(joints[idx["index_mcp"]] - w)
            z_axis = n
            y_axis = self._normalize(np.cross(z_axis, x_axis))
            x_axis = self._normalize(np.cross(y_axis, z_axis))

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        return origin, R


class HandVisualizer:
    LEFT_COLOR = (0, 255, 0)
    RIGHT_COLOR = (0, 0, 255)
    LEFT_ORIGIN = (255, 255, 0)
    RIGHT_ORIGIN = (0, 255, 255)
    AXIS_COLORS = {"x": (0, 0, 255), "y": (0, 255, 0), "z": (255, 0, 0)}

    def render_video(
        self,
        video_path: Path,
        out_path: Path,
        left_img: np.ndarray,
        right_img: np.ndarray,
        left_valid: np.ndarray,
        right_valid: np.ndarray,
        axes_uv_left: Dict[str, np.ndarray],
        axes_uv_right: Dict[str, np.ndarray],
        methods: List[str],
        out_size: Tuple[int, int],
        draw_kps: bool,
    ):
        reader = imageio.get_reader(str(video_path))
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 30))

        writers, multi = self._open_writers(out_path, methods, fps)

        out_w, out_h = out_size
        max_frames = min(left_img.shape[0], right_img.shape[0]) * 2
        written = 0

        for frame_idx, frame in enumerate(reader):
            if frame_idx >= max_frames:
                break
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))

            kps_idx = frame_idx
            base_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            for method, writer in writers.items():
                frame_bgr = base_bgr.copy()

                if draw_kps:
                    self._draw_kps(frame_bgr, left_img[kps_idx], left_valid[kps_idx], self.LEFT_COLOR)
                    self._draw_kps(frame_bgr, right_img[kps_idx], right_valid[kps_idx], self.RIGHT_COLOR)

                if left_valid[kps_idx]:
                    self._draw_axes_uv(frame_bgr, axes_uv_left[method][kps_idx], self.LEFT_ORIGIN)
                if right_valid[kps_idx]:
                    self._draw_axes_uv(frame_bgr, axes_uv_right[method][kps_idx], self.RIGHT_ORIGIN)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                writer.append_data(frame_rgb)

            written += 1

        reader.close()
        for w in writers.values():
            w.close()

        if multi:
            print(f"Wrote {written} frames for methods: {', '.join(methods)}")
        else:
            print(f"Wrote {written} frames to {out_path}")

    def _hand_connections(self, joint_count: int) -> List[Tuple[int, int]]:
        if joint_count >= 21:
            return [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17), (5, 17),
            ]
        if joint_count == 16:
            return [
                (0, 1), (1, 2), (2, 3),
                (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9),
                (0, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15),
                (1, 4), (4, 7), (7, 10), (10, 13), (1, 13),
            ]
        return []

    def _draw_kps(self, img, joints2d: np.ndarray, valid: bool, color):
        if not valid:
            return
        pts = np.asarray(joints2d, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return
        n_joints = pts.shape[0]
        connections = self._hand_connections(n_joints)
        h, w = img.shape[:2]

        def ok(pt):
            return np.isfinite(pt).all()

        for i, j in connections:
            if i >= n_joints or j >= n_joints:
                continue
            p1 = pts[i]
            p2 = pts[j]
            if not (ok(p1) and ok(p2)):
                continue
            x1, y1 = int(round(p1[0])), int(round(p1[1]))
            x2, y2 = int(round(p2[0])), int(round(p2[1]))
            if (0 <= x1 < w and 0 <= y1 < h) or (0 <= x2 < w and 0 <= y2 < h):
                cv2.line(img, (x1, y1), (x2, y2), color, 2)
        for p in pts:
            if not ok(p):
                continue
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), 3, color, -1)

    def _draw_axes_uv(self, img, axes_uv: np.ndarray, origin_color):
        if axes_uv is None or axes_uv.shape[0] < 4:
            return
        ox, oy = axes_uv[0]
        for name, idx in zip(["x", "y", "z"], [1, 2, 3]):
            x, y = axes_uv[idx]
            cv2.line(img, (int(ox), int(oy)), (int(x), int(y)), self.AXIS_COLORS[name], 2)
            cv2.circle(img, (int(x), int(y)), 3, self.AXIS_COLORS[name], -1)
        cv2.circle(img, (int(ox), int(oy)), 4, origin_color, -1)

    def _open_writers(self, out_path: Path, methods: List[str], fps: float):
        multi = len(methods) > 1
        writers = {}
        for method in methods:
            path = out_path if not multi else out_path.with_name(f"{out_path.stem}_{method}{out_path.suffix}")
            writers[method] = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1)
        return writers, multi


class HandRetargetPipeline:
    def __init__(self, model_dir: Path, device: torch.device | None = None,
                 flat_hand_mean: bool = True, fix_shapedirs: bool = True):
        self.extractor = KeypointsExtractor(model_dir, device, flat_hand_mean, fix_shapedirs)
        self.projector = CameraProjector(self.extractor.device)
        self.retargeter = EEFRetargeter()
        self.visualizer = HandVisualizer()

    def render_video(
        self,
        data_path: Path,
        video_path: Path,
        out_path: Path,
        methods: str = "pinch,palm,pca",
        axis_len: float = 0.0,
        axis_scale: float = 0.6,
        draw_kps: bool = False,
    ):
        data = self.extractor.load_data(data_path)
        result = self.extractor.extract(data)

        left_world = result["left"]["keypoints3d"]
        right_world = result["right"]["keypoints3d"]
        left_valid_t = result["left"]["valid"]
        right_valid_t = result["right"]["valid"]

        traj = self.projector.apply_scale(result["camera"]["traj"], result["camera"]["scale"])
        intrinsic = self.projector.build_intrinsic(result["camera"]["img_focal"], result["camera"]["img_center"])
        img_size = result["camera"]["img_center"] * 2
        s0 = img_size[0].item() if hasattr(img_size[0], "item") else float(img_size[0])
        s1 = img_size[1].item() if hasattr(img_size[1], "item") else float(img_size[1])
        src_size = [int(round(s0)), int(round(s1))]
        intrinsic_np = scale_intrinsics(
            intrinsic.detach().cpu().numpy(), src_size, list(SAVED_VIDEO_RESOLUTION)
        )

        left_cam_t = self.projector.world_to_camera(left_world, traj)
        right_cam_t = self.projector.world_to_camera(right_world, traj)
        left_cam = left_cam_t.detach().cpu().numpy()
        right_cam = right_cam_t.detach().cpu().numpy()
        left_img = self.projector.project_np(left_cam, intrinsic_np)
        right_img = self.projector.project_np(right_cam, intrinsic_np)

        left_valid = left_valid_t.detach().cpu().numpy().astype(bool)
        right_valid = right_valid_t.detach().cpu().numpy().astype(bool)

        method_list = [m.strip() for m in methods.split(",") if m.strip()] or ["pinch"]
        axes_uv_left = {}
        axes_uv_right = {}
        for m in method_list:
            axes_left = self.retargeter.compute_axes(left_cam, m, axis_len, axis_scale, side="left")
            axes_right = self.retargeter.compute_axes(right_cam, m, axis_len, axis_scale, side="right")
            axes_uv_left[m] = self.projector.project_np(axes_left, intrinsic_np)
            axes_uv_right[m] = self.projector.project_np(axes_right, intrinsic_np)

        self.visualizer.render_video(
            video_path,
            out_path,
            left_img,
            right_img,
            left_valid,
            right_valid,
            axes_uv_left,
            axes_uv_right,
            method_list,
            SAVED_VIDEO_RESOLUTION,
            draw_kps,
        )
    
    def retarget(self, data_path: Path, methods="pinch_plane", out_path=None, verbose: bool = True):
        """
        Retarget input result; save trajectory as 6D pose (position + axis_angle) to out_path.
        No video rendering. Per frame: both hands, each method: position and rotation(axis_angle).
        """
        data = self.extractor.load_data(data_path)
        if verbose:
            print(data.keys())
        result = self.extractor.extract(data)
        left_world = result["left"]["keypoints3d"]
        right_world = result["right"]["keypoints3d"]

        traj = self.projector.apply_scale(result["camera"]["traj"], result["camera"]["scale"])
        intrinsic = self.projector.build_intrinsic(
            result["camera"]["img_focal"], result["camera"]["img_center"]
        )
        intrinsic_np = intrinsic.detach().cpu().float().numpy()
        img_center = result["camera"]["img_center"].detach().cpu().numpy()

        left_cam = self.projector.world_to_camera(left_world, traj).detach().cpu().numpy()
        right_cam = self.projector.world_to_camera(right_world, traj).detach().cpu().numpy()

        method_list = [m.strip() for m in methods.split(",") if m.strip()] or ["pinch"]

        poses = {}
        for side, cam_data in [("left", left_cam), ("right", right_cam)]:
            for m in method_list:
                poses[(side, m)] = self.retargeter.compute_poses(cam_data, m, side=side)

        left_pinch = self.retargeter.compute_pinch_norm(left_cam)
        right_pinch = self.retargeter.compute_pinch_norm(right_cam)

        fps_val = data.get("fps", 60)
        if isinstance(fps_val, (torch.Tensor, np.ndarray)):
            fps_val = int(fps_val.item() if hasattr(fps_val, "item") else float(fps_val))
        else:
            fps_val = int(fps_val)

        output_data = {
            "video_path": str((data_path.parent / data_path.stem).with_suffix(".mp4")),
            "data_path": str(data_path),
            "fps": fps_val,
            "camera": {
                "intrinsic": intrinsic_np.tolist(),
                "img_size": [
                    int(round(float(img_center[0] * 2))),
                    int(round(float(img_center[1] * 2))),
                ],
            },
            "poses": {},
        }
        num_frames = left_cam.shape[0]

        frame_result = {"left": [], "right": []}
        for side in ["left", "right"]:
            pinch = left_pinch if side == "left" else right_pinch
            for t in range(num_frames):
                origin, axis_angle = poses[(side, m)][t]
                origin = np.asarray(origin)
                axis_angle = np.asarray(axis_angle)
                pose = np.concatenate([origin, axis_angle, [float(pinch[t])]], axis=0)
                frame_result[side].append(pose.tolist())
        output_data["poses"] = frame_result

        if out_path is not None:
            with open(out_path, "w") as f:
                json.dump(_to_json_serializable(output_data), f, ensure_ascii=False, indent=2)
        return output_data

    def replay_json(
        self,
        json_path: Path,
        out_path: Path,
        video_path: Path | None = None,
        axis_len: float = 0.03,
        sample_ratio: float = 1.0,
        verbose: bool = True,
    ):
        """Load retarget JSON (camera intrinsics + video_path), project 6D poses onto video and write.
        If video_path not given, read from JSON video_path field; sample_ratio in (0,1] controls frame sampling.
        """
        with open(json_path, "r") as f:
            all_data = json.load(f)

        if video_path is None:
            video_path = all_data.get("video_path") or (
                str(Path(all_data.get("data_path", "")).with_suffix(".mp4"))
            )
            if not video_path:
                raise ValueError("JSON has no video_path/data_path; pass video_path")
            video_path = Path(video_path)
        else:
            video_path = Path(video_path)

        intrinsic_np = np.array(all_data["camera"]["intrinsic"], dtype=np.float32)
        out_w, out_h = SAVED_VIDEO_RESOLUTION
        img_size = all_data.get("camera", {}).get("img_size")
        if img_size and len(img_size) >= 2:
            intrinsic_np = scale_intrinsics(
                intrinsic_np, [int(img_size[0]), int(img_size[1])], [out_w, out_h]
            )
        pose_dict = all_data["poses"]
        if isinstance(pose_dict.get("left"), list):
            left_poses = pose_dict.get("left", [])
            right_poses = pose_dict.get("right", [])
        else:
            method_key = next(iter(pose_dict), None)
            sub = pose_dict[method_key] if method_key else {}
            left_poses = sub.get("left", [])
            right_poses = sub.get("right", [])

        reader = imageio.get_reader(str(video_path))
        fps = all_data.get("fps", 30)

        step = max(1, round(1.0 / max(1e-6, min(1.0, float(sample_ratio)))))
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", macro_block_size=1)

        axis_colors = {"x": (0, 0, 255), "y": (0, 255, 0), "z": (255, 0, 0)}
        side_origin_colors = {"left": (255, 255, 0), "right": (0, 255, 255)}

        side_poses = {"left": left_poses, "right": right_poses}
        num_poses = max(len(left_poses), len(right_poses))
        written = 0

        for frame_idx, frame in enumerate(reader):
            if frame_idx >= num_poses:
                break
            if frame_idx % step != 0:
                continue
            # Always resize to SAVED_VIDEO_RESOLUTION for consistent output
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            pose_idx = frame_idx

            for side in ["left", "right"]:
                poses_list = side_poses[side]
                if pose_idx >= len(poses_list):
                    continue
                pose_vec = np.array(poses_list[pose_idx], dtype=np.float32)
                origin = pose_vec[:3]
                aa = pose_vec[3:6]
                R = Rscipy.from_rotvec(aa).as_matrix().astype(np.float32)

                pts_3d = np.stack([
                    origin,
                    origin + axis_len * R[:, 0],
                    origin + axis_len * R[:, 1],
                    origin + axis_len * R[:, 2],
                ])
                pts_2d = self.projector.project_np(pts_3d, intrinsic_np)

                ox, oy = int(pts_2d[0, 0]), int(pts_2d[0, 1])
                for name, k in zip(["x", "y", "z"], [1, 2, 3]):
                    ex, ey = int(pts_2d[k, 0]), int(pts_2d[k, 1])
                    cv2.line(frame_bgr, (ox, oy), (ex, ey), axis_colors[name], 2)
                    cv2.circle(frame_bgr, (ex, ey), 3, axis_colors[name], -1)
                cv2.circle(frame_bgr, (ox, oy), 4, side_origin_colors[side], -1)

            writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            written += 1

        reader.close()
        writer.close()
        if verbose:
            print(f"Wrote {written} frames (sample_ratio={sample_ratio}) to {out_path}")


if __name__ == "__main__":

    retarget = HandRetargetPipeline(model_dir="/home/cyx/projects/ego_overlay/assets/mano_v1_2/models")
    retarget.retarget(data_path=Path("/home/cyx/projects/ego-centric") / "P04_12_364_00043.pose3d_hand", methods="pinch_plane", out_path="/home/cyx/projects/test_ik/outputs/P04_12_364_00043_retarget.json")
    # retarget.render_video(data_path="/home/cyx/projects/ego-centric/P04_12_364_00043.pose3d_hand", video_path="/home/cyx/projects/ego-centric/P04_12_364_00043.mp4", out_path="/home/cyx/projects/test_ik/outputs/P04_12_364_00043_retarget.mp4", methods="pinch_plane", axis_len=0.0, axis_scale=0.6, draw_kps=True)
    retarget.replay_json(
        json_path=Path("/home/cyx/projects/test_ik/outputs/P04_12_364_00043_retarget.json"),
        out_path=Path("/home/cyx/projects/test_ik/outputs/P04_12_364_00043_retarget_replay.mp4"),
        sample_ratio=0.1,
    )
