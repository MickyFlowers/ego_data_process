"""Isaac Sim environment for loading and visualizing the Aloha URDF robot."""

from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET

import numpy as np

from isaacsim import SimulationApp


def _parse_vec3(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split(",")])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isaac Sim Aloha URDF Environment")
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="/home/cyx/projects/mobile_aloha_sim/aloha_new_description/urdf/dual_piper.urdf",
    )
    parser.add_argument("--fix-base", action="store_true", default=True)
    parser.add_argument("--merge-fixed-joints", action="store_true", default=False)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument(
        "--view-eye",
        type=str,
        default="0.56,0.0,0.56",
        help="Third-person camera eye in world coordinates (fallback).",
    )
    parser.add_argument(
        "--view-target",
        type=str,
        default="1.267,0.0,-0.147",
        help="Third-person camera target in world coordinates (fallback). +X forward, 45-deg down.",
    )
    parser.add_argument("--view-up", type=str, default="0.7071,0.0,0.7071")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--no-dae-materials",
        action="store_true",
        help="Disable post-import DAE material binding (use if it errors or stalls)",
    )
    return parser


class IsaacSimEnv:
    """Manages an Isaac Sim stage with a URDF robot loaded and a third-person camera."""

    def __init__(self, args: argparse.Namespace | None = None):
        if args is None:
            args = build_parser().parse_args([])
        self.args = args

        self._app = SimulationApp(
            {"headless": False, "width": args.width, "height": args.height}
        )

        import omni.kit.commands
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdLux, UsdPhysics

        self._world = World(stage_units_in_meters=1.0, physics_dt=args.dt, rendering_dt=args.dt)
        stage = omni.usd.get_context().get_stage()

        # --- physics scene ---
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
        # scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        # scene.CreateGravityMagnitudeAttr().Set(9.81)

        # --- ground plane ---
        PhysicsSchemaTools.addGroundPlane(
            stage, "/World/groundPlane", "Z", 1500, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5)
        )

        # --- lighting ---
        light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
        light.CreateIntensityAttr(3000)
        light.CreateAngleAttr(0.53)

        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr(500)

        # --- import URDF ---
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = args.merge_fixed_joints
        import_config.fix_base = args.fix_base
        import_config.import_inertia_tensor = True
        import_config.convex_decomp = False
        import_config.collision_from_visuals = False
        import_config.create_physics_scene = False
        # Try enabling mesh materials from DAE; some Isaac Sim versions support this
        for attr in ("use_mesh_materials", "useMeshMaterials", "import_mesh_materials"):
            if hasattr(import_config, attr):
                setattr(import_config, attr, True)
                print(f"[IsaacSimEnv] Set {attr}=True for DAE/material import")
                break

        result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=args.urdf_path,
            import_config=import_config,
        )
        self.robot_prim_path: str = result[1] if result and result[1] else "/aloha_tracer2_four_d435_dark"
        print(f"[IsaacSimEnv] Robot loaded at: {self.robot_prim_path}")

        # Left/right arm joint names (7 each: joint1..joint7, no gripper joint8)
        self._left_joint_names = [f"fl_joint{i}" for i in range(1, 8)]
        self._right_joint_names = [f"fr_joint{i}" for i in range(1, 8)]
        self._articulation = None
        self._left_dof_indices = None
        self._right_dof_indices = None

        # After import: bind DAE materials to meshes (run next frame to avoid blocking)
        if not getattr(args, "no_dae_materials", False):
            import omni.kit.app
            app = omni.kit.app.get_app()
            stage_ref = stage
            robot_path_ref = self.robot_prim_path
            urdf_path_ref = args.urdf_path
            self_ref = self

            sub_box = [None]  # Mutable in closure; holds subscription handle

            def _run_once(_e=None):
                self_ref._apply_mesh_materials_from_dae(stage_ref, robot_path_ref, urdf_path_ref)
                if sub_box[0] is not None:
                    sub_box[0].unsubscribe()
                    sub_box[0] = None

            if hasattr(app, "post_update_call"):
                app.post_update_call(lambda: _run_once())
            else:
                # Some versions lack post_update_call: use post-update stream, run once then unsubscribe
                stream = app.get_post_update_event_stream()
                sub_box[0] = stream.create_subscription_to_pop(_run_once, name="IsaacSimEnv_apply_dae_materials")

        # --- camera ---
        eye = _parse_vec3(args.view_eye)
        target = _parse_vec3(args.view_target)
        up = _parse_vec3(args.view_up)
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
        self._apply_camera_up(stage, "/OmniverseKit_Persp", eye, target, up)

        self._world.reset()

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_camera_up(stage, cam_path: str, eye: np.ndarray, target: np.ndarray, up: np.ndarray):
        """Refine the default perspective camera orientation to honour a custom up vector."""
        from pxr import Gf, UsdGeom

        forward = target - eye
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-12)
        true_up = np.cross(right, forward)
        back = -forward

        # Gf.Matrix3d takes 9 doubles (row-major), not three Vec3d
        rot = Gf.Matrix3d(
            right[0], right[1], right[2],
            true_up[0], true_up[1], true_up[2],
            back[0], back[1], back[2],
        )
        # Gf.Rotation does not take Matrix3d; build from axis-angle (Vec3d, double)
        trace_r = rot[0][0] + rot[1][1] + rot[2][2]
        angle = np.arccos(np.clip((trace_r - 1.0) / 2.0, -1.0, 1.0))
        sin_a = np.sin(angle)
        if sin_a > 1e-8:
            ax = (rot[1][2] - rot[2][1]) / (2 * sin_a)
            ay = (rot[2][0] - rot[0][2]) / (2 * sin_a)
            az = (rot[0][1] - rot[1][0]) / (2 * sin_a)
            axis = Gf.Vec3d(ax, ay, az)
        else:
            axis = Gf.Vec3d(1, 0, 0)
        rot_mat = Gf.Matrix4d()
        rot_mat.SetRotateOnly(Gf.Rotation(axis, angle))
        rot_mat.SetTranslateOnly(Gf.Vec3d(*eye.tolist()))

        cam_prim = stage.GetPrimAtPath(cam_path)
        if cam_prim.IsValid():
            xform = UsdGeom.Xformable(cam_prim)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(rot_mat)

    # ------------------------------------------------------------------
    @staticmethod
    def _link_to_dae_paths(urdf_path: str) -> dict[str, list[str]]:
        """Parse each link's visual DAE paths from URDF (package:// resolved relative to urdf)."""
        out: dict[str, list[str]] = {}
        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        # Package root is usually parent of urdf dir (description package root)
        pkg_root = os.path.normpath(os.path.join(urdf_dir, ".."))

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except Exception:
            return out

        for link in root.findall(".//link"):
            name = link.get("name")
            if not name:
                continue
            paths = []
            for visual in link.findall("visual"):
                geom = visual.find("geometry/mesh")
                if geom is None:
                    continue
                fn = geom.get("filename")
                if not fn or not fn.lower().endswith(".dae"):
                    continue
                if fn.startswith("package://"):
                    # package://pkg_name/rel_path -> pkg_root/rel_path (assume current pkg is pkg_name)
                    rest = fn.replace("package://", "").split("/", 1)[-1]
                    full = os.path.normpath(os.path.join(pkg_root, rest))
                else:
                    full = os.path.normpath(os.path.join(urdf_dir, fn))
                if os.path.isfile(full):
                    paths.append(full)
            if paths:
                out[name] = paths
        return out

    @staticmethod
    def _link_to_mesh_prims(stage, robot_prim_path: str):
        """Walk robot prims; return link name -> list of Mesh prims under that link."""
        from pxr import UsdGeom

        out = {}
        robot_path = stage.GetPrimAtPath(robot_prim_path)
        if not robot_path or not robot_path.IsValid():
            return out

        def walk(prim, link_name: str | None):
            if link_name is None:
                # Direct child prim name is treated as link name (common URDF importer layout)
                for child in prim.GetChildren():
                    walk(child, child.GetName())
                return
            if prim.IsA(UsdGeom.Mesh):
                out.setdefault(link_name, []).append(prim)
            for child in prim.GetChildren():
                walk(child, link_name)

        for child in robot_path.GetChildren():
            walk(child, child.GetName())
        return out

    def _apply_mesh_materials_from_dae(self, stage, robot_prim_path: str, urdf_path: str):
        """After import: load materials from DAE and bind to meshes (if Asset Converter available)."""
        from pxr import Sdf, UsdGeom, UsdShade

        link_to_dae = self._link_to_dae_paths(urdf_path)
        link_to_meshes = self._link_to_mesh_prims(stage, robot_prim_path)
        if not link_to_dae or not link_to_meshes:
            return

        try:
            import omni.kit.asset_converter as conv
        except Exception:
            print("[IsaacSimEnv] omni.kit.asset_converter not available, skip DAE material binding")
            return

        # Create conversion context, keep materials (do not ignore)
        ctx = None
        if hasattr(conv, "AssetConverterContext"):
            ctx = conv.AssetConverterContext()
            if hasattr(ctx, "ignore_materials"):
                ctx.ignore_materials = False

        materials_scope = Sdf.Path(robot_prim_path).AppendPath("ImportedMaterials")
        stage.OverridePrim(materials_scope)

        for link_name, dae_paths in link_to_dae.items():
            mesh_prims = link_to_meshes.get(link_name, [])
            if not mesh_prims or not dae_paths:
                continue
            dae_path = dae_paths[0]
            safe_name = link_name.replace("/", "_")
            out_usd = os.path.join(os.path.dirname(dae_path), f"_temp_mtl_{safe_name}.usd")
            try:
                task_manager = conv.get_instance()
                if ctx is not None and hasattr(task_manager, "create_converter_task"):
                    try:
                        task = task_manager.create_converter_task(dae_path, out_usd, None, ctx)
                    except TypeError:
                        task = task_manager.create_converter_task(dae_path, out_usd)
                else:
                    task = task_manager.create_converter_task(dae_path, out_usd)
                # Synchronous wait: poll status (2 or 4 often means done/success)
                for _ in range(300):
                    st = getattr(task, "get_status", lambda: 0)()
                    if st in (2, 4) or getattr(task, "is_done", lambda: False)():
                        break
                    if hasattr(task, "update"):
                        task.update()
                    time.sleep(0.02)
            except Exception as e:
                print(f"[IsaacSimEnv] DAE conversion failed {dae_path}: {e}")
                continue

            if not os.path.isfile(out_usd):
                continue
            try:
                from pxr import Usd
                tmp_stage = Usd.Stage.Open(out_usd)
                mtl_prim = None
                for p in tmp_stage.Traverse():
                    if p.IsA(UsdShade.Material):
                        mtl_prim = p
                        break
                if mtl_prim is None:
                    continue
                mtl_path = mtl_prim.GetPath()
                # Reference the material in this stage
                ref_prim_path = materials_scope.AppendPath(safe_name)
                ref_prim = stage.OverridePrim(ref_prim_path)
                ref_prim.GetReferences().AddReference(out_usd, mtl_path)
                mtl = UsdShade.Material(stage.GetPrimAtPath(ref_prim_path))
                binding_api = UsdShade.MaterialBindingAPI
                for mesh_prim in mesh_prims:
                    binding_api.Apply(mesh_prim.GetPrim()).Bind(mtl)
            except Exception as e:
                print(f"[IsaacSimEnv] Material bind failed {link_name}: {e}")
            finally:
                try:
                    os.remove(out_usd)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def _ensure_articulation(self):
        """Lazy-create Articulation and resolve left/right arm DOF indices (on first set_joint; step at least once first)."""
        if self._articulation is not None:
            return
        names = None
        try:
            from isaacsim.core.prims import SingleArticulation
            self._articulation = SingleArticulation(self.robot_prim_path)
            self._articulation.initialize()
            names = getattr(self._articulation, "dof_names", None) or getattr(
                self._articulation, "get_dof_names", lambda: None
            )()
        except Exception:
            try:
                from isaacsim.core.prims import Articulation
                self._articulation = Articulation(prim_paths_expr=self.robot_prim_path)
                self._articulation.initialize()
                names = getattr(self._articulation, "dof_names", None) or getattr(
                    self._articulation, "get_dof_names", lambda: None
                )()
            except Exception as e2:
                print(f"[IsaacSimEnv] Articulation init failed: {e2}")
                raise
        if names is None:
            raise RuntimeError("[IsaacSimEnv] Could not get articulation dof_names")
        name_to_idx = {n: i for i, n in enumerate(names)}
        self._left_dof_indices = np.array(
            [name_to_idx[n] for n in self._left_joint_names if n in name_to_idx],
            dtype=np.int64,
        )
        self._right_dof_indices = np.array(
            [name_to_idx[n] for n in self._right_joint_names if n in name_to_idx],
            dtype=np.int64,
        )
        if len(self._left_dof_indices) != 7 or len(self._right_dof_indices) != 7:
            print(
                f"[IsaacSimEnv] Warning: left DOFs {len(self._left_dof_indices)}, right {len(self._right_dof_indices)}; "
                "expected 7 each. Check URDF and joint names."
            )

    def set_joint(
        self,
        left_joint_angle: np.ndarray,
        right_joint_angle: np.ndarray,
    ) -> None:
        """Set left/right arm joint angles (7 each); no physics, write current pose.

        - left_joint_angle: shape (7,) radians, fl_joint1..fl_joint7
        - right_joint_angle: shape (7,) radians, fr_joint1..fr_joint7
        """
        left_joint_angle = np.asarray(left_joint_angle, dtype=np.float64)
        right_joint_angle = np.asarray(right_joint_angle, dtype=np.float64)
        if left_joint_angle.shape != (7,) or right_joint_angle.shape != (7,):
            raise ValueError(
                "set_joint requires left_joint_angle and right_joint_angle shape (7,); "
                f"got {left_joint_angle.shape} and {right_joint_angle.shape}"
            )
        self._ensure_articulation()
        indices = np.concatenate([self._left_dof_indices, self._right_dof_indices])
        positions = np.concatenate([left_joint_angle, right_joint_angle])
        self._articulation.set_joint_positions(positions, indices)

    # ------------------------------------------------------------------
    @property
    def world(self):
        return self._world

    @property
    def app(self) -> SimulationApp:
        return self._app

    @property
    def stage(self):
        import omni.usd
        return omni.usd.get_context().get_stage()

    def step(self, render: bool = True):
        self._world.step(render=render)

    def is_running(self) -> bool:
        return self._app.is_running()

    def close(self):
        self._world.stop()
        self._app.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = build_parser().parse_args()
    env = IsaacSimEnv(args)

    print("[IsaacSimEnv] Simulation running.  Close the GUI window to exit.")
    while env.is_running():
        env.step()

    env.close()
