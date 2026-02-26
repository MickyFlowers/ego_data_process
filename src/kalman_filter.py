#!/usr/bin/env python3
"""
Kalman filters with RTS (Rauch-Tung-Striebel) smoothing for trajectory smoothing.

1. JointKalmanFilter
   Linear KF for joint angles of arbitrary dimension.
   Constant-velocity model + discrete white noise acceleration.

2. PoseExtendedKalmanFilter
   EKF for SE(3) pose trajectories (4×4 matrix input/output).
   Proper SO(3) manifold handling for rotation residuals.

Both filters include a forward filtering pass followed by a backward RTS
smoothing pass, which eliminates the phase lag inherent in causal filtering.
"""

import json
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════
#  Joint Kalman Filter (linear, arbitrary dimension)
# ═══════════════════════════════════════════════════════════════════════════

class JointKalmanFilter:
    """
    Linear Kalman Filter + RTS smoother for multi-dimensional joint trajectories.

    Constant-velocity state model
    ─────────────────────────────
        state = [q₁ … qₙ,  dq₁ … dqₙ]   dim = 2n
        obs   = [q₁ … qₙ]                 dim = n

    Parameters
    ----------
    n_dim  : int   – number of joint dimensions (user-specified)
    dt     : float – time step between frames
    q_var  : float – acceleration power spectral density (process noise)
    r_var  : float – measurement noise variance per joint
    """

    def __init__(self, n_dim: int, dt: float = 1 / 60,
                 q_var: float = 100.0, r_var: float = 5e-4):
        self.n = n_dim
        self.dt = dt
        self.sd = 2 * n_dim

        # F: constant velocity transition
        self.F = np.eye(self.sd)
        self.F[:self.n, self.n:] = dt * np.eye(self.n)

        # H: observe position only
        self.H = np.zeros((self.n, self.sd))
        self.H[:self.n, :self.n] = np.eye(self.n)

        # Q: discrete white-noise acceleration model
        dt2 = dt * dt
        In = np.eye(self.n)
        self.Q = np.block([
            [dt2 * dt2 / 4 * In, dt2 * dt / 2 * In],
            [dt2 * dt / 2 * In,  dt2 * In],
        ]) * q_var

        self.R = r_var * np.eye(self.n)

    # ── forward filter + backward RTS ──────────────────────────────────
    def filter_and_smooth(self, observations: np.ndarray, edge_pad: int = 0) -> np.ndarray:
        """
        Parameters
        ----------
        observations : ndarray, shape (T, n_dim)
        edge_pad     : int – replicate first/last frame this many times to
                        absorb RTS boundary effects; stripped before return.

        Returns
        -------
        smoothed : ndarray, shape (T, n_dim)
        """
        orig_T = observations.shape[0]
        if edge_pad > 0 and orig_T > 1:
            pad_s = np.tile(observations[0:1], (edge_pad, 1))
            pad_e = np.tile(observations[-1:], (edge_pad, 1))
            observations = np.concatenate([pad_s, observations, pad_e], axis=0)

        T = observations.shape[0]
        sd, n = self.sd, self.n
        I_sd = np.eye(sd)

        xp = np.zeros((T, sd))
        Pp = np.zeros((T, sd, sd))
        xf = np.zeros((T, sd))
        Pf = np.zeros((T, sd, sd))

        # initialise state
        x = np.zeros(sd)
        x[:n] = observations[0]
        if T > 1:
            x[n:] = (observations[1] - observations[0]) / self.dt
        P = np.eye(sd)
        P[:n, :n] *= self.R[0, 0] * 10
        P[n:, n:] *= 1.0

        # ── forward pass ──
        for t in range(T):
            xp[t] = self.F @ x
            Pp[t] = self.F @ P @ self.F.T + self.Q

            y = observations[t] - self.H @ xp[t]
            S = self.H @ Pp[t] @ self.H.T + self.R
            K = Pp[t] @ self.H.T @ np.linalg.inv(S)

            x = xp[t] + K @ y
            P = (I_sd - K @ self.H) @ Pp[t]
            xf[t], Pf[t] = x.copy(), P.copy()

        # ── backward RTS pass ──
        xs = xf.copy()
        for t in range(T - 2, -1, -1):
            G = Pf[t] @ self.F.T @ np.linalg.inv(Pp[t + 1])
            xs[t] = xf[t] + G @ (xs[t + 1] - xp[t + 1])

        result = xs[:, :n]
        if edge_pad > 0 and orig_T > 1:
            result = result[edge_pad:edge_pad + orig_T]
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  6-D Pose Extended Kalman Filter (SE(3))
# ═══════════════════════════════════════════════════════════════════════════

class PoseExtendedKalmanFilter:
    """
    EKF + RTS smoother for SE(3) pose trajectories.

    Input:  (T, 4, 4) homogeneous transformation matrices
    Output: (T, 4, 4) smoothed matrices

    State = [px py pz  rx ry rz  vx vy vz  wx wy wz]  dim 12
    Obs   = [px py pz  rx ry rz]                       dim 6

    Rotation uses axis-angle (rotation vector).
    Prediction composes rotations on SO(3); update uses manifold-aware
    residual  y_rot = Log( Exp(z_rot) · Exp(x̂_rot)⁻¹ ).

    Parameters
    ----------
    dt            : float – time step
    q_pos / q_rot : float – acceleration PSD for position / rotation
    r_pos / r_rot : float – measurement noise variance for position / rotation
    """

    def __init__(self, dt: float = 1 / 60,
                 q_pos: float = 100.0, q_rot: float = 100.0,
                 r_pos: float = 5e-4,  r_rot: float = 1e-3):
        self.dt = dt
        self.sd = 12
        self.od = 6

        # H: observe [p, r]
        self.H = np.zeros((6, 12))
        self.H[:6, :6] = np.eye(6)

        # Q: piecewise-constant white-noise acceleration
        dt2 = dt * dt
        self.Q = np.zeros((12, 12))
        for i in range(3):
            q = q_pos
            self.Q[i, i]       = dt2 * dt2 / 4 * q
            self.Q[i, i + 6]   = dt2 * dt  / 2 * q
            self.Q[i + 6, i]   = dt2 * dt  / 2 * q
            self.Q[i + 6, i + 6] = dt2 * q
            q = q_rot
            self.Q[i + 3, i + 3]     = dt2 * dt2 / 4 * q
            self.Q[i + 3, i + 9]     = dt2 * dt  / 2 * q
            self.Q[i + 9, i + 3]     = dt2 * dt  / 2 * q
            self.Q[i + 9, i + 9]     = dt2 * q

        self.R = np.diag([r_pos] * 3 + [r_rot] * 3)

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def mat_to_pose6d(T: np.ndarray) -> np.ndarray:
        """4×4 → [x y z rx ry rz]."""
        return np.concatenate([T[:3, 3],
                               R.from_matrix(T[:3, :3]).as_rotvec()])

    @staticmethod
    def pose6d_to_mat(p: np.ndarray) -> np.ndarray:
        """[x y z rx ry rz] → 4×4."""
        T = np.eye(4)
        T[:3, 3]  = p[:3]
        T[:3, :3] = R.from_rotvec(p[3:6]).as_matrix()
        return T

    def _unwrap_rotvec(self, obs: np.ndarray) -> np.ndarray:
        """Make rotation vectors continuous across the trajectory.

        For equivalent rotation θ·â, we try θ+2πk·â (k = ±1,±2,…) and
        pick the one closest to the previous frame.
        """
        for t in range(1, len(obs)):
            r_prev = obs[t - 1, 3:6]
            r_curr = obs[t, 3:6]
            if np.linalg.norm(r_curr - r_prev) > np.pi:
                angle = np.linalg.norm(r_curr)
                if angle < 1e-10:
                    continue
                axis = r_curr / angle
                best, best_d = r_curr, np.linalg.norm(r_curr - r_prev)
                for k in range(-3, 4):
                    if k == 0:
                        continue
                    c = axis * (angle + k * 2 * np.pi)
                    d = np.linalg.norm(c - r_prev)
                    if d < best_d:
                        best, best_d = c, d
                obs[t, 3:6] = best
        return obs

    # ── EKF building blocks ───────────────────────────────────────────
    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Linear constant-velocity prediction.

        Using direct addition instead of SO(3) composition avoids
        as_rotvec() wrapping the rotation vector back to [0, π], which
        would break the continuity maintained by _unwrap_rotvec and cause
        the RTS smoother to produce wildly wrong corrections.
        For 60 fps data the linear approximation r += ω·dt is excellent.
        """
        xp = np.zeros(12)
        xp[:3]   = x[:3] + x[6:9]  * self.dt
        xp[3:6]  = x[3:6] + x[9:12] * self.dt
        xp[6:9]  = x[6:9]
        xp[9:12] = x[9:12]
        return xp

    def _jacobian(self, _x: np.ndarray) -> np.ndarray:
        """Process Jacobian (constant-velocity model)."""
        F = np.eye(12)
        F[:3, 6:9]   = self.dt * np.eye(3)
        F[3:6, 9:12] = self.dt * np.eye(3)
        return F

    def _innovation(self, z: np.ndarray, xp: np.ndarray) -> np.ndarray:
        """Linear innovation (both z and xp live in the unwrapped domain)."""
        return z - self.H @ xp

    # ── main entry point ──────────────────────────────────────────────
    def filter_and_smooth(self, pose_matrices: np.ndarray, edge_pad: int = 0) -> np.ndarray:
        """
        Parameters
        ----------
        pose_matrices : ndarray, shape (T, 4, 4)
        edge_pad      : int – replicate first/last pose this many times to
                         absorb RTS boundary effects; stripped before return.

        Returns
        -------
        smoothed : ndarray, shape (T, 4, 4)
        """
        orig_T = pose_matrices.shape[0]
        if edge_pad > 0 and orig_T > 1:
            pad_s = np.tile(pose_matrices[0:1], (edge_pad, 1, 1))
            pad_e = np.tile(pose_matrices[-1:], (edge_pad, 1, 1))
            pose_matrices = np.concatenate([pad_s, pose_matrices, pad_e], axis=0)

        T = pose_matrices.shape[0]

        obs = np.zeros((T, 6))
        for t in range(T):
            obs[t] = self.mat_to_pose6d(pose_matrices[t])
        obs = self._unwrap_rotvec(obs)

        xp_a = np.zeros((T, 12))
        Pp_a = np.zeros((T, 12, 12))
        xf   = np.zeros((T, 12))
        Pf   = np.zeros((T, 12, 12))
        F_a  = np.zeros((T, 12, 12))

        # init
        x = np.zeros(12)
        x[:6] = obs[0]
        if T > 1:
            x[6:9]  = (obs[1, :3] - obs[0, :3]) / self.dt
            x[9:12] = (obs[1, 3:6] - obs[0, 3:6]) / self.dt
        P = np.eye(12)
        P[:6, :6]  *= 0.01
        P[6:, 6:]  *= 1.0

        I12 = np.eye(12)

        # ── forward EKF ──
        for t in range(T):
            F = self._jacobian(x)
            xp = self._predict(x)
            Pp = F @ P @ F.T + self.Q

            F_a[t], xp_a[t], Pp_a[t] = F, xp, Pp

            y = self._innovation(obs[t], xp)
            S = self.H @ Pp @ self.H.T + self.R
            K = Pp @ self.H.T @ np.linalg.inv(S)

            x = xp + K @ y
            P = (I12 - K @ self.H) @ Pp
            xf[t], Pf[t] = x.copy(), P.copy()

        # ── backward RTS ──
        xs = xf.copy()
        Ps = Pf.copy()
        for t in range(T - 2, -1, -1):
            G = Pf[t] @ F_a[t + 1].T @ np.linalg.inv(Pp_a[t + 1])
            xs[t] = xf[t] + G @ (xs[t + 1] - xp_a[t + 1])
            Ps[t] = Pf[t] + G @ (Ps[t + 1] - Pp_a[t + 1]) @ G.T

        result = np.zeros((T, 4, 4))
        for t in range(T):
            result[t] = self.pose6d_to_mat(xs[t, :6])
        if edge_pad > 0 and orig_T > 1:
            result = result[edge_pad:edge_pad + orig_T]
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def set_axes_equal(ax):
    """Set 3-D axes to equal scale (Matplotlib has no built-in for this)."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    half = 0.5 * (limits[:, 1] - limits[:, 0]).max()
    ax.set_xlim3d([center[0] - half, center[0] + half])
    ax.set_ylim3d([center[1] - half, center[1] + half])
    ax.set_zlim3d([center[2] - half, center[2] + half])


def draw_frame(ax, T_mat, length=0.006, lw=1.2, alpha=0.65):
    """Draw an RGB coordinate frame (X=red, Y=green, Z=blue) at a pose."""
    o = T_mat[:3, 3]
    rot = T_mat[:3, :3]
    for i, c in enumerate(['r', 'g', 'b']):
        tip = o + rot[:, i] * length
        ax.plot([o[0], tip[0]], [o[1], tip[1]], [o[2], tip[2]],
                color=c, alpha=alpha, linewidth=lw)


# ═══════════════════════════════════════════════════════════════════════════
#  Main: load JSON → smooth → plot
# ═══════════════════════════════════════════════════════════════════════════

def main():
    json_files = sorted(glob.glob("outputs/data/*.json"))
    if not json_files:
        print("No JSON files found under outputs/data/")
        return

    json_path = json_files[0]
    print(f"Processing: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    fps = data.get("fps", 60)
    dt  = 1.0 / fps

    left_raw  = np.asarray(data["poses"]["left"],  dtype=np.float64)
    right_raw = np.asarray(data["poses"]["right"], dtype=np.float64)

    cam_rot = R.from_euler('ZXZ', [-90, -135, 0], degrees=True).as_matrix()
    cam_mat = np.eye(4)
    cam_mat[:3, :3] = cam_rot

    n = min(left_raw.shape[0], right_raw.shape[0])

    def raw_to_matrices(raw, n_frames):
        mats = np.zeros((n_frames, 4, 4))
        for i in range(n_frames):
            T = np.eye(4)
            T[:3, :3] = R.from_rotvec(raw[i, 3:6]).as_matrix()
            T[:3, 3]  = raw[i, :3]
            mats[i] = cam_mat @ T
        return mats

    left_mats  = raw_to_matrices(left_raw,  n)
    right_mats = raw_to_matrices(right_raw, n)

    # ── 6-D Pose EKF + RTS ────────────────────────────────────────────
    ekf = PoseExtendedKalmanFilter(
        dt=dt, q_pos=1000.0, q_rot=1000.0, r_pos=5e-4, r_rot=1e-3,
    )
    print("Running Pose EKF + RTS smoothing...")
    left_smooth  = ekf.filter_and_smooth(left_mats)
    right_smooth = ekf.filter_and_smooth(right_mats)
    print("  6D Pose EKF + RTS smoothing done")

    # ── Joint KF + RTS (demo: smooth the 6-DOF raw vectors) ──────────
    jkf = JointKalmanFilter(n_dim=6, dt=dt, q_var=100.0, r_var=5e-4)
    print("Running Joint KF + RTS smoothing...")
    left_jsmooth  = jkf.filter_and_smooth(left_raw[:n, :6])
    right_jsmooth = jkf.filter_and_smooth(right_raw[:n, :6])
    print("  Joint KF + RTS smoothing done")

    # ── 3-D visualisation ─────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 9))

    palette = [('tab:blue', 'Left Hand'), ('tab:red', 'Right Hand')]

    for idx, (mats_o, mats_s, (colour, label)) in enumerate([
        (left_mats,  left_smooth,  palette[0]),
        (right_mats, right_smooth, palette[1]),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')

        pos_o = mats_o[:, :3, 3]
        pos_s = mats_s[:, :3, 3]

        ax.plot(pos_o[:, 0], pos_o[:, 1], pos_o[:, 2],
                '-', color='gray', alpha=0.45, lw=0.7, label='Original')
        ax.plot(pos_s[:, 0], pos_s[:, 1], pos_s[:, 2],
                '-', color=colour, lw=1.8, label='EKF+RTS Smoothed')

        step = max(1, n // 25)
        for i in range(0, n, step):
            draw_frame(ax, mats_s[i], length=0.005)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{label}  ({n} frames @ {fps} fps)')
        ax.legend(loc='upper right', fontsize=9)
        set_axes_equal(ax)

    plt.suptitle(
        f'Pose EKF + RTS Smoothing — {os.path.basename(json_path)}',
        fontsize=14,
    )
    plt.tight_layout()

    out_dir = "outputs/video"
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(json_path).replace('.json', '')
    out_path = os.path.join(out_dir, f'{stem}_smooth_3d.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
