#!/usr/bin/env python3
"""
检查 IK 输出目录下 joint_trajectory.json 中的关节轨迹是否超出 URDF 定义的关节限位。

用法:
    python scripts/check_joint_limits.py outputs/test_ik/data
    python scripts/check_joint_limits.py outputs/test_ik/data --urdf assets/aloha_new_description/urdf/dual_piper_origin.urdf
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_joint_limits_from_urdf(urdf_path: str) -> dict[str, tuple[float, float]]:
    """
    从 URDF 解析关节限位，返回 {joint_name: (lower, upper)}。
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limits = {}
    for joint in root.findall(".//joint"):
        jname = joint.get("name")
        if jname is None:
            continue
        limit_elem = joint.find("limit")
        if limit_elem is not None:
            lower = float(limit_elem.get("lower", 0))
            upper = float(limit_elem.get("upper", 0))
            limits[jname] = (lower, upper)
    return limits


# joint_trajectory 格式: 每帧 [q1,q2,q3,q4,q5,q6,gripper]
# gripper 对应 joint7 开合 [0, 0.04]，joint8 = -gripper 故 [-0.04, 0]
LEFT_ARM_JOINTS = ["left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6"]
RIGHT_ARM_JOINTS = ["right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_joint5", "right_joint6"]
GRIPPER_JOINT = "left_joint7"  # gripper 值对应 joint7 范围 [0, 0.04]，左右臂 gripper 限位相同


def check_trajectory(
    traj_path: str,
    limits: dict[str, tuple[float, float]],
    verbose: bool = True,
) -> dict:
    """
    检查单个 joint_trajectory.json 是否超出限位。
    返回: {clip_id, violations: [{frame, side, joint, value, lower, upper}, ...], total_violations}
    """
    with open(traj_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    left_traj = data.get("left_joint_trajectory", [])
    right_traj = data.get("right_joint_trajectory", [])
    clip_id = Path(traj_path).parent.name

    violations = []
    n_frames = max(len(left_traj), len(right_traj))

    for frame_idx in range(n_frames):
        # Left arm: joints 0-5 + gripper 6
        if frame_idx < len(left_traj):
            row = left_traj[frame_idx]
            for j_idx, jname in enumerate(LEFT_ARM_JOINTS):
                if j_idx >= len(row):
                    break
                val = row[j_idx]
                if jname in limits:
                    lo, hi = limits[jname]
                    if val < lo or val > hi:
                        violations.append({
                            "frame": frame_idx,
                            "side": "left",
                            "joint": jname,
                            "value": val,
                            "lower": lo,
                            "upper": hi,
                        })
            # gripper (index 6)
            if len(row) >= 7:
                gval = row[6]
                lo, hi = limits.get("left_joint7", (0.0, 0.04))
                if gval < lo or gval > hi:
                    violations.append({
                        "frame": frame_idx,
                        "side": "left",
                        "joint": "gripper",
                        "value": gval,
                        "lower": lo,
                        "upper": hi,
                    })

        # Right arm
        if frame_idx < len(right_traj):
            row = right_traj[frame_idx]
            for j_idx, jname in enumerate(RIGHT_ARM_JOINTS):
                if j_idx >= len(row):
                    break
                val = row[j_idx]
                if jname in limits:
                    lo, hi = limits[jname]
                    if val < lo or val > hi:
                        violations.append({
                            "frame": frame_idx,
                            "side": "right",
                            "joint": jname,
                            "value": val,
                            "lower": lo,
                            "upper": hi,
                        })
            if len(row) >= 7:
                gval = row[6]
                lo, hi = limits.get("right_joint7", (0.0, 0.04))
                if gval < lo or gval > hi:
                    violations.append({
                        "frame": frame_idx,
                        "side": "right",
                        "joint": "gripper",
                        "value": gval,
                        "lower": lo,
                        "upper": hi,
                    })

    result = {"clip_id": clip_id, "violations": violations, "total_violations": len(violations), "n_frames": n_frames}
    if verbose and violations:
        print(f"\n[{clip_id}] {len(violations)} 处超限 (共 {n_frames} 帧):")
        for v in violations[:20]:  # 最多打印 20 条
            print(f"  frame {v['frame']} {v['side']} {v['joint']}: {v['value']:.4f} (限位 [{v['lower']:.4f}, {v['upper']:.4f}])")
        if len(violations) > 20:
            print(f"  ... 还有 {len(violations) - 20} 处")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 IK 输出关节轨迹是否超出 URDF 关节限位")
    parser.add_argument("data_dir", type=str, help="IK 输出 data 目录，如 outputs/test_ik/data")
    parser.add_argument(
        "--urdf",
        type=str,
        default="assets/aloha_new_description/urdf/dual_piper_origin.urdf",
        help="URDF 文件路径（用于读取关节限位）",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="仅输出汇总，不打印每条违规")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    urdf_path = Path(args.urdf)
    if not urdf_path.is_absolute():
        urdf_path = Path.cwd() / urdf_path

    if not urdf_path.exists():
        print(f"错误: URDF 不存在: {urdf_path}")
        return

    limits = parse_joint_limits_from_urdf(str(urdf_path))
    print(f"从 {urdf_path} 加载关节限位: {len(limits)} 个关节")
    for j in LEFT_ARM_JOINTS + ["left_joint7", "left_joint8"] + RIGHT_ARM_JOINTS + ["right_joint7", "right_joint8"]:
        if j in limits:
            print(f"  {j}: [{limits[j][0]:.4f}, {limits[j][1]:.4f}]")

    traj_files = sorted(data_dir.glob("*/joint_trajectory.json"))
    if not traj_files:
        print(f"在 {data_dir} 下未找到 joint_trajectory.json")
        return

    print(f"\n检查 {len(traj_files)} 个 clip ...")
    total_violations = 0
    clips_with_violations = 0

    for tf in traj_files:
        res = check_trajectory(str(tf), limits, verbose=not args.quiet)
        total_violations += res["total_violations"]
        if res["total_violations"] > 0:
            clips_with_violations += 1

    print(f"\n===== 汇总 =====")
    print(f"共检查 {len(traj_files)} 个 clip，{clips_with_violations} 个存在超限，总违规 {total_violations} 处")
    if clips_with_violations == 0:
        print("全部在限位内 ✓")


if __name__ == "__main__":
    main()
