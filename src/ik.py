#!/usr/bin/env python3
"""PyBullet env: load and visualize dual_piper.urdf, dual-arm IK and error stats."""
# Python 3.11+ compat (inspect.getargspec)
import src.compat  # noqa: F401

import json
import os
import time
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from src.kalman_filter import JointKalmanFilter, PoseExtendedKalmanFilter

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MESH_RGBA = (0.055, 0.055, 0.055, 1.0)
COLLISION_MARGIN = 0.005
EARLY_OUT_DIST = 0.5
REPLAY_FPS = 60
IK_JUMP_THRESHOLD = 0.5  # rad per frame — max joint velocity ≈ 15 rad/s @30fps
KF_EDGE_PAD = 15  # frames to replicate at edges before KF smoothing (absorbs RTS boundary effect)
# Validation render: low res, no shadows, for speed
RENDER_RESOLUTION = [640, 480]

LEFT_JOINT_NAMES = (
    "left_joint1", "left_joint2", "left_joint3",
    "left_joint4", "left_joint5", "left_joint6",
)
RIGHT_JOINT_NAMES = (
    "right_joint1", "right_joint2", "right_joint3",
    "right_joint4", "right_joint5", "right_joint6",
)
LEFT_GRIPPER_JOINT_NAMES = ("left_joint7", "left_joint8")  # left_joint7 0-0.04, left_joint8 -0.04-0
RIGHT_GRIPPER_JOINT_NAMES = ("right_joint7", "right_joint8")  # right_joint7 0-0.04, right_joint8 -0.04-0

# Gripper open/close mapped to [0, 0.04] (m), URDF joint7 [0, 0.04], joint8 [-0.04, 0]
GRIPPER_OPEN_MAX_M = 0.04
LEFT_ARM_LINK_NAMES = {
    "left_link1", "left_link2", "left_link3",
    "left_link4", "left_link5", "left_link6",
    "left_link7", "left_link8", "left_tcp",
}
RIGHT_ARM_LINK_NAMES = {
    "right_link1", "right_link2", "right_link3",
    "right_link4", "right_link5", "right_link6",
    "right_link7", "right_link8", "right_tcp",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_bbox(pose_matrix: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Compute bbox of pose matrix sequence.
    pose_matrix: (n, 4, 4)
    Returns: (bbox_min, bbox_max), each (3,); empty sequence returns (None, None).
    """
    if pose_matrix.shape[0] == 0:
        return None, None
    bbox_min = pose_matrix[:, :3, 3].min(axis=0)
    bbox_max = pose_matrix[:, :3, 3].max(axis=0)
    return bbox_min, bbox_max


def vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    """Pose vector [x,y,z, rx,ry,rz] -> 4x4 transform matrix."""
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_rotvec(vector[3:6]).as_matrix()
    matrix[:3, 3] = vector[:3]
    return matrix


def quat_distance(q1: np.ndarray | tuple, q2: np.ndarray | tuple) -> float:
    """Angular distance [0, pi] between two quaternions, q as (x, y, z, w)."""
    dot = abs(np.dot(np.asarray(q1), np.asarray(q2)))
    return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


# -----------------------------------------------------------------------------
# Pose data loading and alignment
# -----------------------------------------------------------------------------
def get_default_camera_matrix() -> np.ndarray:
    """Default camera 4x4 matrix (rotation and origin only)."""
    rot = R.from_euler("ZXZ", [-90, -135, 0.0], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    return T


def build_pose_matrices(
    left_poses: np.ndarray,
    right_poses: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert left/right pose vectors to (n, 4, 4) pose matrices (camera frame)."""
    n_left, n_right = left_poses.shape[0], right_poses.shape[0]
    left_T = np.stack([camera_matrix @ vector_to_matrix(left_poses[i]) for i in range(n_left)], axis=0)
    right_T = np.stack([camera_matrix @ vector_to_matrix(right_poses[i]) for i in range(n_right)], axis=0)
    return left_T.astype(np.float64), right_T.astype(np.float64)


def align_poses_to_workstation(
    left_pose_matrix: np.ndarray,
    right_pose_matrix: np.ndarray,
    workstation_center: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Translate left/right poses to workstation center (Y unchanged), return updated camera_matrix.
    Returns: (left_pose_matrix, right_pose_matrix, camera_matrix).
    """
    combined = np.concatenate([left_pose_matrix, right_pose_matrix], axis=0)
    bbox_min, bbox_max = compute_bbox(combined)
    if bbox_min is None:
        return left_pose_matrix, right_pose_matrix, camera_matrix
    center = (bbox_min + bbox_max) / 2
    shift = workstation_center - center
    shift[1] = 0.0

    cam = camera_matrix.copy()
    cam[:3, 3] += shift

    left_new = left_pose_matrix.copy()
    right_new = right_pose_matrix.copy()
    left_new[:, :3, 3] += shift
    right_new[:, :3, 3] += shift
    return left_new, right_new, cam


BOUNDARY_TRIM_THRESHOLD = 0.5  # max pose-6D delta to consider a boundary frame valid


def _trim_bad_boundary_frames(
    left_poses: np.ndarray,
    right_poses: np.ndarray,
    threshold: float = BOUNDARY_TRIM_THRESHOLD,
    max_trim: int = 5,
) -> tuple[int, int]:
    """
    Detect garbage frames at trajectory boundaries by checking per-frame
    pose-6D deltas.  Returns (trim_start, trim_end): number of frames to
    discard from each end.  Scans inward from each boundary, stopping at the
    first frame whose delta to its neighbour is below *threshold*.
    """
    n = min(left_poses.shape[0], right_poses.shape[0])
    if n < 3:
        return 0, 0

    def _max_delta(i: int, j: int) -> float:
        dl = np.max(np.abs(left_poses[i, :6] - left_poses[j, :6]))
        dr = np.max(np.abs(right_poses[i, :6] - right_poses[j, :6]))
        return max(float(dl), float(dr))

    trim_end = 0
    for k in range(min(max_trim, n - 1)):
        if _max_delta(n - 1 - k, n - 2 - k) > threshold:
            trim_end = k + 1
        else:
            break

    trim_start = 0
    for k in range(min(max_trim, n - 1)):
        if _max_delta(k, k + 1) > threshold:
            trim_start = k + 1
        else:
            break

    return trim_start, trim_end


def map_gripper_to_01_and_smooth(
    left_gripper_raw: np.ndarray,
    right_gripper_raw: np.ndarray,
    fps: float,
    out_min: float = 0.0,
    out_max: float = GRIPPER_OPEN_MAX_M,
    use_joint_kf: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map raw gripper signal ([0,1] = open/close) to [out_min, out_max] and apply Joint KF smoothing.
    Returns: (left_gripper_m, right_gripper_m), each (T,) in meters.
    """
    left = np.clip(np.asarray(left_gripper_raw, dtype=np.float64), 0.0, 1.0)
    right = np.clip(np.asarray(right_gripper_raw, dtype=np.float64), 0.0, 1.0)
    left_m = left * (out_max - out_min) + out_min
    right_m = right * (out_max - out_min) + out_min
    if not use_joint_kf:
        return left_m, right_m
    T = left_m.shape[0]
    obs = np.stack([left_m, right_m], axis=1)
    jkf = JointKalmanFilter(n_dim=2, dt=1.0 / fps, q_var=100.0, r_var=5e-4)
    smoothed = jkf.filter_and_smooth(obs, edge_pad=KF_EDGE_PAD)
    smoothed = np.clip(smoothed, out_min, out_max)
    return smoothed[:, 0], smoothed[:, 1]


def load_pose_data_from_json(
    json_path: str,
    workstation_center: np.ndarray | None = None,
    use_ekf: bool = True,
    use_gripper_kf: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict, np.ndarray, np.ndarray, np.ndarray, list, str | None]:
    """
    Load poses from JSON and apply alignment and filtering.
    Returns: (..., camera_intrinsics, img_size, video_path).
    """
    if workstation_center is None:
        workstation_center = np.array([0.35, 0.0, 0.1])

    data = load_json(json_path)
    fps = data["fps"]
    output_data = {"camera": data["camera"], "fps": fps}
    video_path = data.get("video_path", None)

    left_poses = np.asarray(data["poses"]["left"], dtype=np.float64)
    right_poses = np.asarray(data["poses"]["right"], dtype=np.float64)
    if left_poses.ndim == 1:
        left_poses = left_poses.reshape(1, -1)
    if right_poses.ndim == 1:
        right_poses = right_poses.reshape(1, -1)

    # Trim garbage frames at boundaries (retarget often produces reset/zero poses at clip edges)
    trim_s, trim_e = _trim_bad_boundary_frames(left_poses, right_poses)
    if trim_s or trim_e:
        end_idx = left_poses.shape[0] - trim_e if trim_e else left_poses.shape[0]
        left_poses = left_poses[trim_s:end_idx]
        right_poses = right_poses[trim_s:end_idx]

    n_left, n_right = left_poses.shape[0], right_poses.shape[0]
    pose_dim = left_poses.shape[1]
    if pose_dim >= 7:
        left_gripper_raw = left_poses[:, 6]
        right_gripper_raw = right_poses[:, 6]
    else:
        left_gripper_raw = np.full(n_left, 1.0)
        right_gripper_raw = np.full(n_right, 1.0)
    left_gripper, right_gripper = map_gripper_to_01_and_smooth(
        left_gripper_raw, right_gripper_raw, fps, out_max=GRIPPER_OPEN_MAX_M, use_joint_kf=use_gripper_kf
    )
    cam_data = data["camera"]
    camera_intrinsics = np.array(cam_data.get("intrinsics", cam_data.get("intrinsic", np.eye(3))))
    img_size = list(cam_data.get("img_size", [1920, 1080]))
    # Align lengths
    n_common = min(n_left, n_right, len(left_gripper), len(right_gripper))
    left_gripper = left_gripper[:n_common]
    right_gripper = right_gripper[:n_common]

    cam = get_default_camera_matrix()
    left_T, right_T = build_pose_matrices(left_poses, right_poses, cam)
    left_T, right_T, cam = align_poses_to_workstation(
        left_T, right_T, workstation_center, cam
    )

    if use_ekf:
        ekf = PoseExtendedKalmanFilter(
            dt=1.0 / fps, q_pos=1000.0, q_rot=1000.0, r_pos=5e-4, r_rot=1e-3,
        )
        left_T = ekf.filter_and_smooth(left_T, edge_pad=KF_EDGE_PAD)
        right_T = ekf.filter_and_smooth(right_T, edge_pad=KF_EDGE_PAD)

    return left_T, right_T, cam, fps, output_data, left_gripper, right_gripper, camera_intrinsics, img_size, video_path


# -----------------------------------------------------------------------------
# Robot loading and parsing
# -----------------------------------------------------------------------------
def load_robot(urdf_path: str, use_gui: bool = True) -> int:
    """Load URDF, connect GUI/DIRECT, return robot_id. Tries EGL plugin for GPU rendering in DIRECT mode."""
    if use_gui:
        p.connect(p.GUI)
    else:
        cid = p.connect(p.DIRECT)
        # Try loading EGL plugin for GPU rendering (faster than TinyRenderer)
        try:
            import pkgutil
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        except Exception:
            pass
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    return robot_id


def parse_arm_info(robot_id: int) -> dict[str, Any]:
    """
    Parse left/right arm joint indices, TCP, DOF, limits, collision pairs.
    Returns dict for set_arm_joints, check_cross_arm_collision, solve_frame_with_collision, etc.
    """
    num_joints = p.getNumJoints(robot_id)
    left_arm_joint_indices = []
    right_arm_joint_indices = []
    left_tcp_link_index = None
    right_tcp_link_index = None

    left_gripper_joint_indices = []
    right_gripper_joint_indices = []
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        jname = info[1].decode("utf-8") if isinstance(info[1], bytes) else info[1]
        if jname == "left_tcp_joint":
            left_tcp_link_index = j
        if jname == "right_tcp_joint":
            right_tcp_link_index = j
        if jname in LEFT_JOINT_NAMES:
            left_arm_joint_indices.append(j)
        if jname in RIGHT_JOINT_NAMES:
            right_arm_joint_indices.append(j)
        if jname in LEFT_GRIPPER_JOINT_NAMES:
            left_gripper_joint_indices.append(j)
        if jname in RIGHT_GRIPPER_JOINT_NAMES:
            right_gripper_joint_indices.append(j)

    dof_per_joint = [
        j for j in range(num_joints)
        if p.getJointInfo(robot_id, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
    ]
    left_dof_indices = [dof_per_joint.index(j) for j in left_arm_joint_indices]
    right_dof_indices = [dof_per_joint.index(j) for j in right_arm_joint_indices]

    rest_poses = [0.0] * len(dof_per_joint)
    for arm_indices in (left_arm_joint_indices, right_arm_joint_indices):
        for j in arm_indices:
            info = p.getJointInfo(robot_id, j)
            rest_poses[dof_per_joint.index(j)] = 0.5 * (info[8] + info[9])

    left_arm_links = []
    right_arm_links = []
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        child = info[12].decode("utf-8") if isinstance(info[12], bytes) else info[12]
        if child in LEFT_ARM_LINK_NAMES:
            left_arm_links.append(j)
        elif child in RIGHT_ARM_LINK_NAMES:
            right_arm_links.append(j)
    cross_arm_pairs = [(la, ra) for la in left_arm_links for ra in right_arm_links]

    left_bounds = [
        (p.getJointInfo(robot_id, j)[8], p.getJointInfo(robot_id, j)[9])
        for j in left_arm_joint_indices
    ]
    right_bounds = [
        (p.getJointInfo(robot_id, j)[8], p.getJointInfo(robot_id, j)[9])
        for j in right_arm_joint_indices
    ]
    all_bounds = left_bounds + right_bounds

    return {
        "robot_id": robot_id,
        "left_arm_joint_indices": left_arm_joint_indices,
        "right_arm_joint_indices": right_arm_joint_indices,
        "left_gripper_joint_indices": left_gripper_joint_indices,
        "right_gripper_joint_indices": right_gripper_joint_indices,
        "left_tcp_link_index": left_tcp_link_index,
        "right_tcp_link_index": right_tcp_link_index,
        "dof_per_joint": dof_per_joint,
        "left_dof_indices": left_dof_indices,
        "right_dof_indices": right_dof_indices,
        "rest_poses": rest_poses,
        "cross_arm_pairs": cross_arm_pairs,
        "all_bounds": all_bounds,
    }


def apply_robot_visual_settings(robot_id: int, rgba: tuple[float, float, float, float] = MESH_RGBA) -> None:
    """Set robot shadow, lighting and mesh color."""
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(
        lightPosition=[0.5, 0.0, 1.5],
        shadowMapResolution=4096,
        shadowMapIntensity=0.6,
    )
    for link_idx in range(-1, p.getNumJoints(robot_id) + 1):
        p.changeVisualShape(robot_id, link_idx, rgbaColor=rgba)


# -----------------------------------------------------------------------------
# Robot control (uses parse_arm_info result)
# -----------------------------------------------------------------------------
def set_arm_joints(
    robot_info: dict[str, Any],
    q_left: list[float],
    q_right: list[float],
) -> None:
    """Write left/right arm joint angles to PyBullet."""
    rid = robot_info["robot_id"]
    for k, j in enumerate(robot_info["left_arm_joint_indices"]):
        p.resetJointState(rid, j, q_left[k])
    for k, j in enumerate(robot_info["right_arm_joint_indices"]):
        p.resetJointState(rid, j, q_right[k])


def set_gripper(
    robot_info: dict[str, Any],
    left_gripper_m: float,
    right_gripper_m: float,
) -> None:
    """
    Set left/right gripper open/close in meters, range [0, GRIPPER_OPEN_MAX_M].
    URDF: joint7 in [0, 0.04], joint8 in [-0.04, 0], symmetric drive.
    """
    left_gripper_m = np.clip(float(left_gripper_m), 0.0, GRIPPER_OPEN_MAX_M)
    right_gripper_m = np.clip(float(right_gripper_m), 0.0, GRIPPER_OPEN_MAX_M)
    rid = robot_info["robot_id"]
    left_inds = robot_info["left_gripper_joint_indices"]
    right_inds = robot_info["right_gripper_joint_indices"]
    # joint7 = +g, joint8 = -g
    if len(left_inds) >= 2:
        p.resetJointState(rid, left_inds[0], left_gripper_m)
        p.resetJointState(rid, left_inds[1], -left_gripper_m)
    if len(right_inds) >= 2:
        p.resetJointState(rid, right_inds[0], right_gripper_m)
        p.resetJointState(rid, right_inds[1], -right_gripper_m)


def check_cross_arm_collision(
    robot_info: dict[str, Any],
    margin: float = COLLISION_MARGIN,
) -> list:
    """Check cross-arm collision, return list of contacts."""
    rid = robot_info["robot_id"]
    result = []
    for la, ra in robot_info["cross_arm_pairs"]:
        contacts = p.getClosestPoints(rid, rid, margin, linkIndexA=la, linkIndexB=ra)
        for c in contacts:
            if c[8] < margin:
                result.append(c)
    return result


def solve_frame_with_collision(
    robot_info: dict[str, Any],
    left_tgt_pos: np.ndarray,
    left_tgt_quat: np.ndarray,
    right_tgt_pos: np.ndarray,
    right_tgt_quat: np.ndarray,
    q0: np.ndarray,
) -> tuple[list[float], list[float]]:
    """12-DOF dual-arm scipy optimization (pose + collision penalty), returns (q_left, q_right)."""
    left_tgt_pos = np.asarray(left_tgt_pos)
    left_tgt_quat = np.asarray(left_tgt_quat)
    right_tgt_pos = np.asarray(right_tgt_pos)
    right_tgt_quat = np.asarray(right_tgt_quat)
    rid = robot_info["robot_id"]
    left_tcp = robot_info["left_tcp_link_index"]
    right_tcp = robot_info["right_tcp_link_index"]
    cross_arm_pairs = robot_info["cross_arm_pairs"]
    all_bounds = robot_info["all_bounds"]

    def cost(q: np.ndarray) -> float:
        q_l, q_r = q[:6].tolist(), q[6:].tolist()
        set_arm_joints(robot_info, q_l, q_r)
        ls = p.getLinkState(rid, left_tcp)
        rs = p.getLinkState(rid, right_tcp)
        pos_cost = (
            np.sum((np.array(ls[0]) - left_tgt_pos) ** 2)
            + np.sum((np.array(rs[0]) - right_tgt_pos) ** 2)
        )
        ori_cost = (
            quat_distance(ls[1], left_tgt_quat) ** 2
            + quat_distance(rs[1], right_tgt_quat) ** 2
        )
        col_cost = 0.0
        for la, ra in cross_arm_pairs:
            contacts = p.getClosestPoints(rid, rid, COLLISION_MARGIN, linkIndexA=la, linkIndexB=ra)
            for c in contacts:
                d = c[8]
                if d < COLLISION_MARGIN:
                    col_cost += (COLLISION_MARGIN - d) ** 2
        smooth_cost = np.sum((q - q0) ** 2)
        return 100.0 * pos_cost + 10.0 * ori_cost + 500.0 * col_cost + 0.1 * smooth_cost

    res = minimize(cost, q0, method="L-BFGS-B", bounds=all_bounds, options={"maxiter": 60, "ftol": 1e-9})
    return res.x[:6].tolist(), res.x[6:].tolist()


# -----------------------------------------------------------------------------
# Dual-arm IK trajectory solving
# -----------------------------------------------------------------------------
def _ik_solve_single_arm(
    rid: int,
    tcp_link: int,
    pos: list[float],
    quat: list[float],
    dof_indices: list[int],
    arm_joint_indices: list[int],
    dof_per_joint: list[int],
    rest_poses: list[float],
    prev_q: list[float] | None,
    jump_threshold: float,
) -> tuple[list[float], bool]:
    """
    Solve IK for one arm with branch-jump protection (rate-limited).

    If the new solution jumps more than *jump_threshold* from prev_q,
    reset joint states to prev_q and re-solve. If still exceeding the
    threshold, rate-limit each joint (move toward the solution by at
    most *jump_threshold* per frame) instead of freezing.

    Returns (q_arm, was_rate_limited).
    """
    is_first = prev_q is None
    max_iters = 500 if is_first else 100
    res_thr = 1e-6 if is_first else 1e-5

    q_all = p.calculateInverseKinematics(
        rid, tcp_link, pos, quat,
        maxNumIterations=max_iters, residualThreshold=res_thr, restPoses=rest_poses,
    )
    q = [q_all[k] for k in dof_indices]

    # First frame: commit immediately then re-solve to let IK refine from its own solution
    if is_first:
        for k, j in enumerate(arm_joint_indices):
            p.resetJointState(rid, j, q[k])
            rest_poses[dof_per_joint.index(j)] = q[k]
        q_all = p.calculateInverseKinematics(
            rid, tcp_link, pos, quat,
            maxNumIterations=500, residualThreshold=1e-7, restPoses=rest_poses,
        )
        q = [q_all[k] for k in dof_indices]

    rate_limited = False
    if prev_q is not None:
        max_d = max(abs(q[k] - prev_q[k]) for k in range(len(q)))
        if max_d > jump_threshold:
            # Retry with joints reset to previous frame
            for k, j in enumerate(arm_joint_indices):
                p.resetJointState(rid, j, prev_q[k])
            q_all2 = p.calculateInverseKinematics(
                rid, tcp_link, pos, quat,
                maxNumIterations=300, residualThreshold=1e-6, restPoses=rest_poses,
            )
            q2 = [q_all2[k] for k in dof_indices]
            d2 = max(abs(q2[k] - prev_q[k]) for k in range(len(q2)))
            if d2 < max_d:
                q = q2
                max_d = d2
            if max_d > jump_threshold:
                # Rate-limit: move toward solution but cap step per joint
                for k in range(len(q)):
                    delta = q[k] - prev_q[k]
                    if abs(delta) > jump_threshold:
                        q[k] = prev_q[k] + jump_threshold * (1.0 if delta > 0 else -1.0)
                rate_limited = True

    # Commit: update joint states and restPoses
    for k, j in enumerate(arm_joint_indices):
        p.resetJointState(rid, j, q[k])
        rest_poses[dof_per_joint.index(j)] = q[k]

    return q, rate_limited


def solve_dual_arm_ik(
    robot_info: dict[str, Any],
    left_pose_matrix: np.ndarray,
    right_pose_matrix: np.ndarray,
    early_out_dist: float = EARLY_OUT_DIST,
    jump_threshold: float = IK_JUMP_THRESHOLD,
    verbose: bool = True,
) -> tuple[list[list[float]], list[list[float]], int]:
    """
    Solve dual-arm IK frame-by-frame alternately with branch-jump protection.
    Returns: (left_joint_trajectory, right_joint_trajectory, collision_count).
    """
    n_frames = min(left_pose_matrix.shape[0], right_pose_matrix.shape[0])
    left_traj: list[list[float]] = []
    right_traj: list[list[float]] = []
    collision_count = 0
    jump_clamp_count = 0
    rest_poses = robot_info["rest_poses"]
    dof_per_joint = robot_info["dof_per_joint"]
    left_dof_indices = robot_info["left_dof_indices"]
    right_dof_indices = robot_info["right_dof_indices"]
    left_arm_joint_indices = robot_info["left_arm_joint_indices"]
    right_arm_joint_indices = robot_info["right_arm_joint_indices"]
    rid = robot_info["robot_id"]
    left_tcp = robot_info["left_tcp_link_index"]
    right_tcp = robot_info["right_tcp_link_index"]

    if verbose:
        print(f"[IK] Solving dual-arm IK for {n_frames} frames (jump_thresh={jump_threshold:.2f} rad) ...")
    for i in range(n_frames):
        T_l = left_pose_matrix[i]
        T_r = right_pose_matrix[i]
        l_pos = T_l[:3, 3].tolist()
        l_quat = R.from_matrix(T_l[:3, :3]).as_quat().tolist()
        r_pos = T_r[:3, 3].tolist()
        r_quat = R.from_matrix(T_r[:3, :3]).as_quat().tolist()

        # First frame: no jump protection (free jump from rest poses)
        prev_l = left_traj[-1] if left_traj else None
        prev_r = right_traj[-1] if right_traj else None

        # --- Left arm IK (with jump protection) ---
        q_left, l_clamped = _ik_solve_single_arm(
            rid, left_tcp, l_pos, l_quat,
            left_dof_indices, left_arm_joint_indices, dof_per_joint,
            rest_poses, prev_l, jump_threshold,
        )
        # --- Right arm IK (with jump protection) ---
        q_right, r_clamped = _ik_solve_single_arm(
            rid, right_tcp, r_pos, r_quat,
            right_dof_indices, right_arm_joint_indices, dof_per_joint,
            rest_poses, prev_r, jump_threshold,
        )

        if l_clamped or r_clamped:
            jump_clamp_count += 1
            if verbose:
                sides = []
                if l_clamped:
                    sides.append("LEFT")
                if r_clamped:
                    sides.append("RIGHT")
                print(f"  frame {i}: {'+'.join(sides)} arm IK jump clamped")

        # --- Collision check ---
        l_tcp_state = p.getLinkState(rid, left_tcp)
        r_tcp_state = p.getLinkState(rid, right_tcp)
        tcp_dist = np.linalg.norm(np.array(l_tcp_state[0]) - np.array(r_tcp_state[0]))
        arm_contacts = check_cross_arm_collision(robot_info) if tcp_dist < early_out_dist else []

        if arm_contacts:
            collision_count += 1
            worst_dist = min(c[8] for c in arm_contacts)
            q0 = np.array(q_left + q_right)
            q_left, q_right = solve_frame_with_collision(
                robot_info, l_pos, l_quat, r_pos, r_quat, q0
            )

            # Post-collision jump guard: rate-limit instead of freezing
            if prev_l is not None:
                for k in range(6):
                    delta = q_left[k] - prev_l[k]
                    if abs(delta) > jump_threshold:
                        q_left[k] = prev_l[k] + jump_threshold * (1.0 if delta > 0 else -1.0)
            if prev_r is not None:
                for k in range(6):
                    delta = q_right[k] - prev_r[k]
                    if abs(delta) > jump_threshold:
                        q_right[k] = prev_r[k] + jump_threshold * (1.0 if delta > 0 else -1.0)

            set_arm_joints(robot_info, q_left, q_right)
            for k, j in enumerate(left_arm_joint_indices):
                rest_poses[dof_per_joint.index(j)] = q_left[k]
            for k, j in enumerate(right_arm_joint_indices):
                rest_poses[dof_per_joint.index(j)] = q_right[k]
            remaining = check_cross_arm_collision(robot_info)
            if verbose:
                status = "resolved" if not remaining else f"remain {min(c[8] for c in remaining):.4f}m"
                print(f"  frame {i}: collision {worst_dist:.4f}m -> {status}")

        left_traj.append(q_left)
        right_traj.append(q_right)
        if verbose and i % 200 == 0:
            print(f"  frame {i}/{n_frames}, collisions so far: {collision_count}")

    if verbose:
        print(f"[IK] Done: {collision_count}/{n_frames} collisions, {jump_clamp_count} jump clamps")
    return left_traj, right_traj, collision_count


# -----------------------------------------------------------------------------
# Joint trajectory jump repair
# -----------------------------------------------------------------------------
def repair_joint_trajectory_jumps(
    traj: np.ndarray,
    velocity_threshold: float | None = None,
    max_outlier_segment: int = 100,
    max_passes: int = 5,
    verbose: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Detect and repair IK solution jumps (branch switches) in joint trajectories.

    Uses per-frame max joint velocity to find jump transitions, pairs them into
    outlier segments, and replaces those segments with linear interpolation.
    Multiple passes handle residual violations at repair boundaries.

    Args:
        traj: (n_frames, n_joints) joint angle trajectory.
        velocity_threshold: max allowed per-frame joint angle change (rad).
            None = adaptive (median + 10*MAD, min 0.15 rad).
        max_outlier_segment: max length of a single outlier segment (frames).
        max_passes: number of detection + repair passes.

    Returns:
        (repaired_trajectory, total_repaired_frames)
    """
    traj = np.array(traj, dtype=float).copy()
    n = len(traj)
    if n < 3:
        return traj, 0

    diffs_init = np.max(np.abs(np.diff(traj, axis=0)), axis=1)
    if velocity_threshold is None:
        med = np.median(diffs_init)
        mad = np.median(np.abs(diffs_init - med))
        velocity_threshold = max(med + 10.0 * mad, 0.15)
        if verbose:
            print(
                f"  [JumpRepair] adaptive threshold={velocity_threshold:.4f} rad/frame "
                f"(median={med:.4f}, MAD={mad:.4f})"
            )

    outlier_mask = np.zeros(n, dtype=bool)
    total_repaired = 0

    for _pass in range(max_passes):
        diffs = np.max(np.abs(np.diff(traj, axis=0)), axis=1)
        transitions = np.where(diffs > velocity_threshold)[0]
        if len(transitions) == 0:
            break

        new_outliers = 0
        i = 0
        while i < len(transitions):
            t = transitions[i]
            paired = False
            if i + 1 < len(transitions):
                t2 = transitions[i + 1]
                if t2 - t <= max_outlier_segment:
                    for f in range(t + 1, t2 + 1):
                        if not outlier_mask[f]:
                            outlier_mask[f] = True
                            new_outliers += 1
                    i += 2
                    paired = True
            if not paired:
                if t + 1 < n and not outlier_mask[t + 1]:
                    outlier_mask[t + 1] = True
                    new_outliers += 1
                i += 1

        if new_outliers == 0:
            break
        total_repaired += new_outliers

        outlier_mask[0] = False
        outlier_mask[-1] = False
        valid_idx = np.where(~outlier_mask)[0]
        if len(valid_idx) < 2:
            break
        for j in range(traj.shape[1]):
            traj[:, j] = np.interp(np.arange(n), valid_idx, traj[valid_idx, j])

    if verbose and total_repaired > 0:
        print(
            f"  [JumpRepair] Repaired {total_repaired}/{n} frames "
            f"({total_repaired / n * 100:.1f}%)"
        )
    return traj, total_repaired


# -----------------------------------------------------------------------------
# IK errors (forward kinematics)
# -----------------------------------------------------------------------------
def compute_ik_errors(
    robot_info: dict[str, Any],
    left_joint_trajectory: list[list[float]],
    right_joint_trajectory: list[list[float]],
    left_pose_matrix: np.ndarray,
    right_pose_matrix: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-frame position/rotation error via forward kinematics.
    Returns: (ik_stats, left_pos_errs, right_pos_errs, left_rot_errs, right_rot_errs).
    """
    n_frames = len(left_joint_trajectory)
    rid = robot_info["robot_id"]
    left_tcp = robot_info["left_tcp_link_index"]
    right_tcp = robot_info["right_tcp_link_index"]

    left_pos_errs = np.zeros(n_frames)
    right_pos_errs = np.zeros(n_frames)
    left_rot_errs = np.zeros(n_frames)
    right_rot_errs = np.zeros(n_frames)

    for i in range(n_frames):
        set_arm_joints(robot_info, left_joint_trajectory[i], right_joint_trajectory[i])
        ls = p.getLinkState(rid, left_tcp)
        rs = p.getLinkState(rid, right_tcp)

        left_pos_errs[i] = np.linalg.norm(np.array(ls[0]) - left_pose_matrix[i, :3, 3])
        right_pos_errs[i] = np.linalg.norm(np.array(rs[0]) - right_pose_matrix[i, :3, 3])

        left_rot_err = np.linalg.inv(left_pose_matrix[i, :3, :3]) @ R.from_quat(ls[1]).as_matrix()
        right_rot_err = np.linalg.inv(right_pose_matrix[i, :3, :3]) @ R.from_quat(rs[1]).as_matrix()
        left_rot_errs[i] = np.linalg.norm(R.from_matrix(left_rot_err).as_rotvec())
        right_rot_errs[i] = np.linalg.norm(R.from_matrix(right_rot_err).as_rotvec())

    ik_stats = {
        "left_avg_pos_err": float(left_pos_errs.mean()),
        "right_avg_pos_err": float(right_pos_errs.mean()),
        "left_avg_rot_err": float(left_rot_errs.mean()),
        "right_avg_rot_err": float(right_rot_errs.mean()),
        "left_max_pos_err": float(left_pos_errs.max()),
        "right_max_pos_err": float(right_pos_errs.max()),
        "left_max_rot_err": float(left_rot_errs.max()),
        "right_max_rot_err": float(right_rot_errs.max()),
    }
    return ik_stats, left_pos_errs, right_pos_errs, left_rot_errs, right_rot_errs


def print_ik_error_report(
    left_pos_errs: np.ndarray,
    right_pos_errs: np.ndarray,
    left_rot_errs: np.ndarray,
    right_rot_errs: np.ndarray,
) -> None:
    """Print IK error statistics."""
    print("\n[IK Error] ===== Left arm =====")
    print(f"  Position (m)   avg={left_pos_errs.mean():.6f}  max={left_pos_errs.max():.6f}  min={left_pos_errs.min():.6f}")
    print(f"  Rotation (rad) avg={left_rot_errs.mean():.6f}  max={left_rot_errs.max():.6f}  min={left_rot_errs.min():.6f}")
    print("[IK Error] ===== Right arm =====")
    print(f"  Position (m)   avg={right_pos_errs.mean():.6f}  max={right_pos_errs.max():.6f}  min={right_pos_errs.min():.6f}")
    print(f"  Rotation (rad) avg={right_rot_errs.mean():.6f}  max={right_rot_errs.max():.6f}  min={right_rot_errs.min():.6f}")
    print(f"  Rotation (deg) left_avg={np.degrees(left_rot_errs.mean()):.4f}  right_avg={np.degrees(right_rot_errs.mean()):.4f}\n")


# -----------------------------------------------------------------------------
# Rendering: view/projection from camera_matrix and camera_intrinsics
# -----------------------------------------------------------------------------
def build_projection_matrix_from_intrinsics(
    camera_intrinsics: np.ndarray,
    width: int,
    height: int,
    near: float = 0.01,
    far: float = 10.0,
) -> list[float]:
    """
    Build PyBullet/OpenGL projection matrix (column-major 16-element list) from 3x3 intrinsics K (fx,fy,cx,cy) and image size.
    """
    K = np.asarray(camera_intrinsics, dtype=np.float64)
    if K.shape == (3, 3):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    else:
        fx = fy = K.flat[0] if K.size >= 1 else 600.0
        cx, cy = width / 2.0, height / 2.0
    A = (near + far) / (near - far)
    B = 2.0 * near * far / (near - far)
    # Projection matrix consistent with OpenGL/PyBullet (see cvK2BulletP)
    P = np.array([
        [2.0 / width * fx, 0, 0, 0],
        [0, 2.0 / height * fy, 0, 0],
        [(width - 2.0 * cx) / width, (2.0 * cy - height) / height, A, B],
        [0, 0, -1, 0],
    ], dtype=np.float64)
    return P.T.ravel().tolist()


def build_view_matrix_from_camera_matrix(camera_matrix: np.ndarray) -> list[float]:
    """
    camera_matrix is 4x4, camera pose in world (camera-to-world).
    Returns PyBullet/OpenGL view matrix (world-to-camera, column-major 16-element list).
    OpenGL convention Y down; Tc for coordinate transform.
    """
    T_world_from_cam = np.asarray(camera_matrix, dtype=np.float64)
    T_cam_from_world = np.linalg.inv(T_world_from_cam)
    # OpenCV/common vision frame -> OpenGL (Y down, etc.)
    Tc = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)
    view_4x4 = Tc @ T_cam_from_world
    return view_4x4.T.ravel().tolist()


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def draw_camera_axis(camera_matrix: np.ndarray, axis_len: float = 0.15) -> None:
    """Draw camera frame (X red, Y green, Z blue)."""
    cam_pos = camera_matrix[:3, 3]
    cam_rot = camera_matrix[:3, :3]
    p.addUserDebugLine(cam_pos, cam_pos + cam_rot[:, 0] * axis_len, [1, 0, 0], lineWidth=3)
    p.addUserDebugLine(cam_pos, cam_pos + cam_rot[:, 1] * axis_len, [0, 1, 0], lineWidth=3)
    p.addUserDebugLine(cam_pos, cam_pos + cam_rot[:, 2] * axis_len, [0, 0, 1], lineWidth=3)
    p.addUserDebugText("cam", cam_pos.tolist(), [1, 1, 1], textSize=1.2)


def draw_trajectory_lines(
    left_pose_matrix: np.ndarray,
    right_pose_matrix: np.ndarray,
    n_frames: int,
) -> None:
    """Draw left/right target trajectory lines."""
    for i in range(n_frames - 1):
        p.addUserDebugLine(
            left_pose_matrix[i, :3, 3].tolist(),
            left_pose_matrix[i + 1, :3, 3].tolist(),
            [1, 0.4, 0.4], lineWidth=2,
        )
        p.addUserDebugLine(
            right_pose_matrix[i, :3, 3].tolist(),
            right_pose_matrix[i + 1, :3, 3].tolist(),
            [0.4, 0.4, 1], lineWidth=2,
        )


def build_full_joint_trajectory(
    left_joint_trajectory: list[list[float]],
    right_joint_trajectory: list[list[float]],
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
) -> np.ndarray:
    """
    Concatenate arm joints and grippers into full joint trajectory: per frame [q_left(6), left_gripper, q_right(6), right_gripper], shape (n_frames, 14).
    """
    n_frames = min(
        len(left_joint_trajectory),
        len(right_joint_trajectory),
        len(left_gripper),
        len(right_gripper),
    )
    left_rows = []
    right_rows = []
    for i in range(n_frames):
        left_row = (
            list(left_joint_trajectory[i])
            + [float(left_gripper[i])]
        )
        right_row = (
            list(right_joint_trajectory[i])
            + [float(right_gripper[i])]
        )
        left_rows.append(left_row)
        right_rows.append(right_row)
    
    joint_traj = {"left_joint_trajectory": left_rows, "right_joint_trajectory": right_rows}
    return joint_traj


def render_and_overlay(
    robot_info: dict[str, Any],
    left_joint_trajectory: list[list[float]],
    right_joint_trajectory: list[list[float]],
    left_gripper_trajectory: np.ndarray,
    right_gripper_trajectory: np.ndarray,
    camera_matrix: np.ndarray,
    camera_intrinsics: np.ndarray,
    render_size: list[int],
    original_video_path: str,
    output_path: str,
    fps: float,
    verbose: bool = True,
) -> None:
    """
    Single-pass streaming: render robot per frame -> seg mask -> overlay onto original video -> write.
    No intermediate frame storage, low memory.
    """
    import imageio
    from PIL import Image

    n_frames = min(len(left_joint_trajectory), len(right_joint_trajectory))
    rw, rh = int(render_size[0]), int(render_size[1])
    view_matrix = build_view_matrix_from_camera_matrix(camera_matrix)
    projection_matrix = build_projection_matrix_from_intrinsics(camera_intrinsics, rw, rh)
    rid = robot_info["robot_id"]
    left_arm = robot_info["left_arm_joint_indices"]
    right_arm = robot_info["right_arm_joint_indices"]
    n_use = min(n_frames, len(left_gripper_trajectory), len(right_gripper_trajectory))

    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.setGravity(0, 0, 0)

    reader = imageio.get_reader(original_video_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)

    need_resize = None  # Decided on first frame
    if verbose:
        print(f"[Render+Overlay] {n_frames} frames, render {rw}x{rh}, video -> {output_path}")

    for i in range(n_frames):
        # --- Set joints ---
        q_left = left_joint_trajectory[i]
        q_right = right_joint_trajectory[i]
        for k, j in enumerate(left_arm):
            p.resetJointState(rid, j, q_left[k])
        for k, j in enumerate(right_arm):
            p.resetJointState(rid, j, q_right[k])
        if i < n_use:
            set_gripper(robot_info, left_gripper_trajectory[i], right_gripper_trajectory[i])

        # --- Render ---
        _, _, rgba, _, seg = p.getCameraImage(
            rw, rh,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            shadow=0,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if hasattr(p, "ER_BULLET_HARDWARE_OPENGL") else p.ER_TINY_RENDERER,
        )
        robot_rgb = np.asarray(rgba, dtype=np.uint8).reshape((rh, rw, 4))[:, :, :3]
        mask = np.asarray(seg).reshape((rh, rw)) >= 0

        # --- Read original frame ---
        try:
            bg = np.asarray(reader.get_data(i))
        except IndexError:
            break
        if bg.ndim == 2:
            bg = np.stack([bg] * 3, axis=-1)
        bg_h, bg_w = bg.shape[:2]

        # --- Resize (only if resolution differs) ---
        if need_resize is None:
            need_resize = (rw, rh) != (bg_w, bg_h)
        if need_resize:
            robot_rgb = np.asarray(Image.fromarray(robot_rgb).resize((bg_w, bg_h), Image.BILINEAR))
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((bg_w, bg_h), Image.NEAREST)) > 127

        # --- Overlay then resize to render_size so saved video uses this resolution ---
        bg[mask] = robot_rgb[mask]
        out_w, out_h = int(render_size[0]), int(render_size[1])
        if (bg_w, bg_h) != (out_w, out_h):
            bg = np.asarray(Image.fromarray(bg).resize((out_w, out_h), Image.BILINEAR))
        writer.append_data(bg[:, :, :3])

        if verbose and ((i + 1) % 200 == 0 or i == n_frames - 1):
            print(f"  {i + 1}/{n_frames}")

    reader.close()
    writer.close()
    if verbose:
        print(f"[Render+Overlay] Done: {output_path} ({n_frames} frames, {fps} FPS)")


def run_replay_loop(
    robot_info: dict[str, Any],
    left_joint_trajectory: list[list[float]],
    right_joint_trajectory: list[list[float]],
    left_pose_matrix: np.ndarray,
    right_pose_matrix: np.ndarray,
    left_gripper_trajectory: np.ndarray | None = None,
    right_gripper_trajectory: np.ndarray | None = None,
    fps: int = REPLAY_FPS,
    tcp_axis_len: float = 0.06,
) -> None:
    """Replay dual-arm + gripper trajectory at fps and draw target frames (visualization only, no render)."""
    n_frames = len(left_joint_trajectory)
    dt = 1.0 / fps
    p.setGravity(0, 0, 0)
    print(f"[Replay] Dual-arm {n_frames} frames, {fps} FPS")

    rid = robot_info["robot_id"]
    left_arm_joint_indices = robot_info["left_arm_joint_indices"]
    right_arm_joint_indices = robot_info["right_arm_joint_indices"]
    if left_gripper_trajectory is None:
        left_gripper_trajectory = np.zeros(n_frames)
    if right_gripper_trajectory is None:
        right_gripper_trajectory = np.zeros(n_frames)
    n_use = min(n_frames, len(left_gripper_trajectory), len(right_gripper_trajectory))

    line_ids = [p.addUserDebugLine([0, 0, 0], [0, 0, 0], [0, 0, 0]) for _ in range(6)]
    frame_index = 0

    while True:
        if frame_index >= n_frames:
            frame_index = 0

        q_left = left_joint_trajectory[frame_index]
        q_right = right_joint_trajectory[frame_index]
        for k, j in enumerate(left_arm_joint_indices):
            p.resetJointState(rid, j, q_left[k], targetVelocity=0.0)
        for k, j in enumerate(right_arm_joint_indices):
            p.resetJointState(rid, j, q_right[k], targetVelocity=0.0)
        if frame_index < n_use:
            set_gripper(
                robot_info,
                left_gripper_trajectory[frame_index],
                right_gripper_trajectory[frame_index],
            )

        T_l = left_pose_matrix[frame_index]
        T_r = right_pose_matrix[frame_index]
        for T, base in [(T_l, 0), (T_r, 3)]:
            o = T[:3, 3]
            rx = T[:3, :3] @ np.array([tcp_axis_len, 0, 0])
            ry = T[:3, :3] @ np.array([0, tcp_axis_len, 0])
            rz = T[:3, :3] @ np.array([0, 0, tcp_axis_len])
            line_ids[base + 0] = p.addUserDebugLine(
                o.tolist(), (o + rx).tolist(), [1, 0, 0], lineWidth=2,
                replaceItemUniqueId=line_ids[base + 0],
            )
            line_ids[base + 1] = p.addUserDebugLine(
                o.tolist(), (o + ry).tolist(), [0, 1, 0], lineWidth=2,
                replaceItemUniqueId=line_ids[base + 1],
            )
            line_ids[base + 2] = p.addUserDebugLine(
                o.tolist(), (o + rz).tolist(), [0, 0, 1], lineWidth=2,
                replaceItemUniqueId=line_ids[base + 2],
            )

        p.stepSimulation()
        time.sleep(dt)
        frame_index += 1


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def scale_intrinsics(
    camera_intrinsics: np.ndarray,
    src_size: list[int],
    dst_size: list[int],
) -> np.ndarray:
    """Scale intrinsics by resolution (src_size/dst_size both [w, h])."""
    K = camera_intrinsics.copy()
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def process_single_clip(
    json_path: str,
    output_dir: str,
    do_render: bool = False,
    render_resolution: list[int] | None = None,
    use_gui: bool = False,
    urdf_path: str = "",
    verbose: bool = True,
) -> dict:
    """
    Process single clip: IK solve -> Joint KF -> error stats -> save.
    Optionally render overlay video.

    Output files:
        {output_dir}/data/{clip_id}/joint_trajectory.json
        {output_dir}/data/{clip_id}/stats.json
        {output_dir}/samples/{clip_id}.mp4  (only when do_render=True)

    Returns: stats dict (same as stats.json content).
    """
    if render_resolution is None:
        render_resolution = RENDER_RESOLUTION
    clip_id = os.path.splitext(os.path.basename(json_path))[0]

    # --- Load data ---
    (
        left_pose_matrix, right_pose_matrix, camera_matrix,
        fps, _, left_gripper, right_gripper,
        camera_intrinsics, img_size, video_path,
    ) = load_pose_data_from_json(json_path, use_ekf=True, use_gripper_kf=True)

    n_frames = min(
        left_pose_matrix.shape[0], right_pose_matrix.shape[0],
        len(left_gripper), len(right_gripper),
    )
    left_pose_matrix = left_pose_matrix[:n_frames]
    right_pose_matrix = right_pose_matrix[:n_frames]
    left_gripper = left_gripper[:n_frames]
    right_gripper = right_gripper[:n_frames]

    # --- PyBullet init ---
    robot_id = load_robot(urdf_path, use_gui=use_gui)
    apply_robot_visual_settings(robot_id)
    robot_info = parse_arm_info(robot_id)

    # --- IK ---
    t0 = time.time()
    left_joint_traj, right_joint_traj, collision_count = solve_dual_arm_ik(
        robot_info, left_pose_matrix, right_pose_matrix, verbose=verbose
    )
    ik_time = time.time() - t0

    # --- IK errors (raw, before repair) ---
    ik_stats, *_ = compute_ik_errors(
        robot_info, left_joint_traj, right_joint_traj,
        left_pose_matrix, right_pose_matrix,
    )

    # --- Repair IK jumps (branch-switch detection + interpolation) ---
    left_arr, n_left_repaired = repair_joint_trajectory_jumps(np.array(left_joint_traj), verbose=verbose)
    right_arr, n_right_repaired = repair_joint_trajectory_jumps(np.array(right_joint_traj), verbose=verbose)

    # --- Joint KF smoothing (r_var controls smoothing strength; higher = smoother) ---
    jkf = JointKalmanFilter(n_dim=6, dt=1.0 / fps, q_var=50.0, r_var=5e-3)
    left_joint_traj = jkf.filter_and_smooth(left_arr, edge_pad=KF_EDGE_PAD)
    right_joint_traj = jkf.filter_and_smooth(right_arr, edge_pad=KF_EDGE_PAD)

    # --- Per-joint max frame-to-frame delta (on final smoothed trajectory) ---
    left_final = np.asarray(left_joint_traj)
    right_final = np.asarray(right_joint_traj)
    if left_final.shape[0] > 1:
        left_deltas = np.max(np.abs(np.diff(left_final, axis=0)), axis=0)
        right_deltas = np.max(np.abs(np.diff(right_final, axis=0)), axis=0)
    else:
        left_deltas = np.zeros(6)
        right_deltas = np.zeros(6)
    max_joint_delta = {
        "left": {f"joint{k+1}": float(left_deltas[k]) for k in range(6)},
        "right": {f"joint{k+1}": float(right_deltas[k]) for k in range(6)},
    }

    # --- Build full joint trajectory ---
    joint_traj = build_full_joint_trajectory(
        left_joint_traj, right_joint_traj, left_gripper, right_gripper
    )

    # --- Save joint_trajectory.json ---
    clip_dir = os.path.join(output_dir, "data", clip_id)
    os.makedirs(clip_dir, exist_ok=True)
    traj_path = os.path.join(clip_dir, "joint_trajectory.json")
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(joint_traj, f)

    # --- Save stats.json ---
    stats = {
        "clip_id": clip_id,
        "json_path": json_path,
        "n_frames": n_frames,
        "fps": fps,
        "ik_time": ik_time,
        "ik_time_per_frame": ik_time / max(n_frames, 1),
        "collision_count": collision_count,
        "ik_stats": ik_stats,
        "max_joint_delta": max_joint_delta,
        "jump_repair": {
            "left_repaired_frames": n_left_repaired,
            "right_repaired_frames": n_right_repaired,
        },
    }
    stats_path = os.path.join(clip_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # --- Optional render overlay ---
    if do_render:
        if video_path and os.path.isfile(video_path):
            render_K = scale_intrinsics(camera_intrinsics, img_size, render_resolution)
            sample_dir = os.path.join(output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            overlay_path = os.path.join(sample_dir, f"{clip_id}.mp4")
            render_and_overlay(
                robot_info, left_joint_traj, right_joint_traj,
                left_gripper, right_gripper,
                camera_matrix, render_K, render_resolution,
                video_path, overlay_path, fps,
                verbose=verbose,
            )
        else:
            print(f"[{clip_id}] Render skipped: source video not found ({video_path})")

    # --- Disconnect PyBullet ---
    p.disconnect()

    if verbose:
        print(f"[{clip_id}] Done: {n_frames} frames, IK {ik_time:.1f}s, collisions {collision_count}")
    return stats

def main() -> None:
    json_path = os.path.abspath("outputs/epfl_retarget/data/YH2002_2023_12_04_10_15_23__11585_12023_439.json")
    urdf = os.path.abspath("assets/aloha_new_description/urdf/dual_piper.urdf")
    process_single_clip(json_path, output_dir="outputs/epfl_ik", do_render=False, use_gui=True, urdf_path=urdf)

if __name__ == "__main__":
    main()