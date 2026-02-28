"""Isaac Sim environment for loading and visualizing the Aloha URDF robot.

Batch mode: Ray multi-process, one GPU per worker, each worker initializes
SimulationApp once and processes all assigned clips sequentially.

    python tests/isaac_sim_env.py --input-dir outputs/ik/data --output-dir outputs/render --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import ray

# SimulationApp imported lazily in IsaacSimEnv / IsaacSimWorkerActor to avoid
# loading Omniverse in the Ray main process when using batch mode


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
    parser.add_argument("--headless", action="store_true", help="Headless mode for fast batch render")
    parser.add_argument("--output-dir", type=str, default=None, help="Save output to this dir (enables render mode)")
    parser.add_argument("--no-overlay", action="store_true", help="Output seg only; default is overlay robot onto original video")
    parser.add_argument("--video-path", type=str, default=None, help="Override source video path for overlay (default: from meta.json)")
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--render-size", type=int, default=512, help="Square resolution for render (default: 512x512)")
    parser.add_argument("--render-scale", type=float, default=1.0, help="Scale resolution for speed (e.g. 0.5 = half res)")
    return parser


class IsaacSimEnv:
    """Manages an Isaac Sim stage with a URDF robot loaded and a third-person camera."""

    def __init__(self, args: argparse.Namespace | None = None):
        from isaacsim import SimulationApp

        if args is None:
            args = build_parser().parse_args([])
        self.args = args

        headless = getattr(args, "headless", False)
        app_config = {
            "headless": headless,
            "width": args.width,
            "height": args.height,
        }
        if headless:
            app_config["windowless"] = True
        self._app = SimulationApp(app_config)

        import omni.kit.commands
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics

        self._world = World(stage_units_in_meters=1.0, physics_dt=args.dt, rendering_dt=args.dt)
        stage = omni.usd.get_context().get_stage()

        # --- physics scene (kinematic only: no gravity, no ground) ---
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(0.0)  # gravity disabled

        # --- ground plane (disabled for kinematic visualization) ---
        # PhysicsSchemaTools.addGroundPlane(
        #     stage, "/World/groundPlane", "Z", 1500, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5)
        # )

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
        import_config.import_inertia_tensor = False  # kinematic only
        import_config.convex_decomp = False
        import_config.collision_from_visuals = False
        import_config.create_physics_scene = False
        if hasattr(import_config, "self_collision"):
            import_config.self_collision = False
        elif hasattr(import_config, "selfCollision"):
            import_config.selfCollision = False
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

        # Kinematic only: disable self-collision and dynamics
        robot_prim = stage.GetPrimAtPath(self.robot_prim_path)
        if robot_prim.IsValid():
            physx_api = PhysxSchema.PhysxArticulationAPI.Get(stage, self.robot_prim_path)
            if not physx_api:
                physx_api = PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
            attr = physx_api.GetEnabledSelfCollisionsAttr()
            if attr:
                attr.Set(False)
            else:
                physx_api.CreateEnabledSelfCollisionsAttr(False)

            # Add semantic label for Replicator instance segmentation
            try:
                for schema_name in ("SemanticsLabelsAPI", "SemanticsAPI"):
                    try:
                        if not robot_prim.HasAPI(schema_name, "class"):
                            robot_prim.ApplyAPI(schema_name, "class")
                        sem_attr = robot_prim.GetAttribute("semantics:labels:class")
                        if not sem_attr:
                            sem_attr = robot_prim.CreateAttribute(
                                "semantics:labels:class", Sdf.ValueTypeNames.TokenArray
                            )
                        sem_attr.Set(["robot"])
                        break
                    except Exception:
                        continue
            except Exception as e:
                print(f"[IsaacSimEnv] Semantic label failed (seg may not work): {e}")

        # Left/right arm joint names (7 each: joint1..joint7, no gripper joint8)
        self._left_joint_names = [f"left_joint{i}" for i in range(1, 8)]
        self._right_joint_names = [f"right_joint{i}" for i in range(1, 8)]
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

        import carb
        if not getattr(args, "headless", False):
            carb.settings.get_settings().set("/app/viewport/showStatistics", True)

        self._world.reset()

        # Replicator render product for rgb + segmentation (first-person camera view)
        self._render_product = None
        self._rgb_annotator = None
        self._seg_annotator = None
        self._setup_replicator_render(args)

    # ------------------------------------------------------------------
    def _setup_replicator_render(self, args: argparse.Namespace) -> None:
        """Create render product from viewport camera with instance segmentation (seg only for speed)."""
        try:
            import omni.replicator.core as rep

            scale = getattr(args, "render_scale", 1.0)
            rw = int(getattr(args, "render_width", 512) * scale)
            rh = int(getattr(args, "render_height", 512) * scale)
            rw, rh = max(64, rw), max(64, rh)
            cam_path = "/OmniverseKit_Persp"

            if hasattr(rep.settings, "set_render_rasterized"):
                rep.settings.set_render_rasterized()
                print("[IsaacSimEnv] Set render rasterized")
            elif hasattr(rep.settings, "set_render_rtx_realtime"):
                rep.settings.set_render_rtx_realtime()

            self._render_product = rep.create.render_product(cam_path, (rw, rh))
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._rgb_annotator.attach(self._render_product)
            self._seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
            self._seg_annotator.attach(self._render_product)
            print(f"[IsaacSimEnv] Replicator render {rw}x{rh} (rgb + seg)")
        except Exception as e:
            print(f"[IsaacSimEnv] Replicator setup failed: {e}")

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
    def _set_camera_intrinsics_opencv(
        cam_path: str,
        K: np.ndarray,
        width: int,
        height: int,
        near: float,
        far: float,
        *,
        render_product_path: str | None = None,
        pixel_size_um: float = 3.0,
    ) -> None:
        """Set camera intrinsics via OpenCV pinhole model.

        Focal length and aperture derived from K: horizontal_aperture = pixel_size*width,
        focal_length = pixel_size*(fx+fy)/2, all in meters (stage units).
        """
        from isaacsim.sensors.camera import Camera

        (fx, _, cx), (_, fy, cy), (_, _, _) = K[0:3, 0:3]
        fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)

        # pixel_size = pixel_size_um * 1e-6
        # horizontal_aperture = pixel_size * width
        # vertical_aperture = pixel_size * height
        # fx, fy 可不同（非正方形像素），set_focal_length 僅接受單值，用 fx 與 horizontal_aperture 對應
        # 實際投影由 set_opencv_pinhole_properties 的 fx, fy 分別控制
        # focal_length = pixel_size * fy

        cam = Camera(
            prim_path=cam_path,
            resolution=(width, height),
            frequency=None,
            render_product_path=render_product_path,
        )
        cam.initialize(attach_rgb_annotator=False)

        # cam.set_focal_length(focal_length)
        # cam.set_lens_aperture(0.0)
        # cam.set_horizontal_aperture(horizontal_aperture, maintain_square_pixels=False)
        # cam.set_vertical_aperture(vertical_aperture, maintain_square_pixels=False)
        cam.set_clipping_range(float(near), float(far))
        cam.set_opencv_pinhole_properties(cx=cx, cy=cy, fx=fx, fy=fy, pinhole=[0.0] * 12)

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

    def render_frame(
        self,
        left_joint_angle: np.ndarray,
        right_joint_angle: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Set joints, render from first-person camera, return (rgb, seg_mask).

        rgb: (H, W, 4) uint8 RGBA
        seg_mask: (H, W) uint32 or binary (robot > 0)
        Returns None if Replicator not setup.
        """
        if self._render_product is None or self._seg_annotator is None or self._rgb_annotator is None:
            return None
        self.set_joint(left_joint_angle, right_joint_angle)
        self._world.step(render=True)
        import omni.replicator.core as rep

        rep.orchestrator.step(rt_subframes=1)
        rgb = self._rgb_annotator.get_data()
        seg_data = self._seg_annotator.get_data()
        seg_arr = seg_data["data"] if isinstance(seg_data, dict) else seg_data
        seg_mask = (seg_arr > 0).astype(np.uint8) if np.issubdtype(seg_arr.dtype, np.integer) else seg_arr
        return rgb, seg_mask

    def is_running(self) -> bool:
        return self._app.is_running()

    def close(self):
        self._world.stop()
        self._app.close()


# ----------------------------------------------------------------------
def _resolve_clip_root(input_dir: str) -> str:
    """Resolve input_dir to the directory that directly contains clip subdirs.

    - If input_dir is IK output root (e.g. outputs/ik), clips are in input_dir/data.
    - If input_dir already contains clip subdirs (e.g. outputs/ik/data), use as is.
    """
    if not os.path.isdir(input_dir):
        return input_dir
    # Check if this dir has clip subdirs (any subdir with meta.json)
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "meta.json")):
            return input_dir  # Already has clips
    # No clips here; try input_dir/data (IK output layout)
    data_sub = os.path.join(input_dir, "data")
    if os.path.isdir(data_sub):
        return data_sub
    return input_dir


def _gather_clip_dirs(input_dir: str) -> list[str]:
    """Return sorted list of valid clip directories (each has meta.json + joint_trajectory.json)."""
    root = _resolve_clip_root(input_dir)
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, "meta.json")) and os.path.isfile(
            os.path.join(path, "joint_trajectory.json")
        ):
            out.append(os.path.abspath(path))
    return out


def _render_single_clip(env: IsaacSimEnv, data_dir: str, opts: argparse.Namespace | dict) -> dict | None:
    """Render one clip; returns stats dict or None on failure."""
    import json

    try:
        with open(os.path.join(data_dir, "meta.json")) as f:
            meta = json.load(f)
        with open(os.path.join(data_dir, "joint_trajectory.json")) as f:
            traj_data = json.load(f)
    except Exception as e:
        print(f"[IsaacSimEnv] Failed to load {data_dir}: {e}")
        return None

    def _get(key: str, default=None):
        return opts.get(key, default) if isinstance(opts, dict) else getattr(opts, key, default)

    # Camera from extrinsics (4x4 camera-to-world, OpenCV convention)
    cam_ext = np.array(meta["camera_extrinsics"], dtype=np.float64)
    eye = cam_ext[:3, 3]
    forward = cam_ext[:3, 2]
    up = -cam_ext[:3, 1]
    target = eye + forward

    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    IsaacSimEnv._apply_camera_up(env.stage, "/OmniverseKit_Persp", eye, target, up)

    # Scale intrinsics to render resolution (ego_overlay convention)
    render_w = _get("render_width", 512)
    render_h = _get("render_height", 512)
    scale = _get("render_scale", 1.0)
    if scale != 1.0:
        render_w = int(render_w * scale)
        render_h = int(render_h * scale)
    img_size = meta.get("img_size", [512, 512])
    img_w, img_h = max(1, img_size[0]), max(1, img_size[1])
    K = np.array(meta.get("camera_intrinsics", [[512, 0, 256], [0, 512, 256], [0, 0, 1]]), dtype=np.float64)
    scale_x = render_w / img_w
    scale_y = render_h / img_h
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y

    fx, fy = float(K_scaled[0, 0]), float(K_scaled[1, 1])
    cx, cy = float(K_scaled[0, 2]), float(K_scaled[1, 2])
    # Omniverse 相機使用方像素，fx!=fy 時需渲染尺寸調整
    render_h_actual = render_h
    needs_resize = False
    if abs(fx - fy) > 1e-3:
        render_h_actual = max(1, int(round(render_h * fx / fy)))
        K_scaled[1, 1] = fx
        K_scaled[1, 2] = cy * render_h_actual / render_h
        needs_resize = render_h_actual != render_h

    near_plane = float(meta.get("camera_near", 0.01))
    far_plane = float(meta.get("camera_far", 10.0))
    rp_path = None
    if env._render_product is not None:
        rp_path = getattr(env._render_product, "path", env._render_product)
    IsaacSimEnv._set_camera_intrinsics_opencv(
        "/OmniverseKit_Persp",
        K_scaled,
        width=render_w,
        height=render_h_actual,
        near=near_plane,
        far=far_plane,
        render_product_path=rp_path,
    )

    left_traj = traj_data["left_joint_trajectory"]
    right_traj = traj_data["right_joint_trajectory"]
    n_frames = min(len(left_traj), len(right_traj))

    output_dir = _get("output_dir")
    do_render = output_dir and env._render_product is not None

    do_overlay = False
    source_reader = None
    writer = None
    fps_meta = meta.get("fps", 30)
    clip_id = meta.get("clip_id", os.path.basename(data_dir.rstrip("/")))
    output_video_path = os.path.join(output_dir, f"{clip_id}.mp4") if output_dir else None

    render_w = _get("render_width", 512)
    render_h = _get("render_height", 512)
    scale = _get("render_scale", 1.0)
    if scale != 1.0:
        render_w = int(render_w * scale)
        render_h = int(render_h * scale)

    if do_render:
        os.makedirs(output_dir, exist_ok=True)
        video_path = _get("video_path") or meta.get("video_path")
        if not _get("no_overlay", False) and video_path and os.path.isfile(video_path):
            try:
                import imageio

                source_reader = imageio.get_reader(video_path)
                do_overlay = True
            except Exception:
                source_reader = None
        try:
            import imageio

            writer = imageio.get_writer(output_video_path, fps=fps_meta, codec="libx264", macro_block_size=1)
        except ImportError:
            writer = None

    t_start = time.time()
    for frame_idx in range(n_frames):
        if not env.is_running():
            break
        left_q = np.array(left_traj[frame_idx], dtype=np.float64)
        right_q = np.array(right_traj[frame_idx], dtype=np.float64)

        if do_render and writer is not None:
            result = env.render_frame(left_q, right_q)
            if result is not None:
                rgb, seg = result
                robot_rgb = np.asarray(rgb[:, :, :3], dtype=np.uint8)
                mask = (seg > 0).astype(np.uint8)
                if needs_resize and (robot_rgb.shape[0], robot_rgb.shape[1]) != (render_h, render_w):
                    try:
                        from PIL import Image

                        robot_rgb = np.asarray(
                            Image.fromarray(robot_rgb).resize((render_w, render_h), Image.BILINEAR)
                        )
                        mask = np.asarray(
                            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                                (render_w, render_h), Image.NEAREST
                            )
                        ) > 127
                    except ImportError:
                        pass

                try:
                    from PIL import Image
                except ImportError:
                    Image = None

                if do_overlay and source_reader is not None and Image is not None:
                    try:
                        bg = np.asarray(source_reader.get_data(frame_idx)).copy()
                    except (IndexError, Exception):
                        bg = None
                    if bg is not None:
                        if bg.ndim == 2:
                            bg = np.stack([bg] * 3, axis=-1)
                        elif bg.shape[-1] == 4:
                            bg = bg[..., :3]
                        bg_h, bg_w = bg.shape[:2]
                        if (bg_w, bg_h) != (render_w, render_h):
                            bg = np.asarray(Image.fromarray(bg).resize((render_w, render_h), Image.BILINEAR))
                        if robot_rgb.shape[:2] != (render_h, render_w):
                            robot_rgb = np.asarray(
                                Image.fromarray(robot_rgb).resize((render_w, render_h), Image.BILINEAR)
                            )
                            mask = (
                                np.asarray(
                                    Image.fromarray(mask * 255).resize((render_w, render_h), Image.NEAREST)
                                )
                                > 127
                            )
                        bg = np.asarray(bg, dtype=np.uint8).copy()
                        bg[mask.astype(bool)] = robot_rgb[mask.astype(bool)]
                        out_frame = bg
                    else:
                        out_frame = np.stack([mask * 255] * 3, axis=-1)
                else:
                    out_frame = np.stack([mask * 255] * 3, axis=-1)
                writer.append_data(out_frame[:, :, :3])
        elif not do_render:
            env.set_joint(left_q, right_q)
            env.step()

    if source_reader is not None:
        source_reader.close()
    t_elapsed = time.time() - t_start

    if writer is not None:
        writer.close()
        render_fps = n_frames / t_elapsed if t_elapsed > 0 else 0.0
        return {
            "clip_id": clip_id,
            "n_frames": n_frames,
            "elapsed_seconds": round(t_elapsed, 2),
            "render_fps": round(render_fps, 2),
            "output_video_fps": fps_meta,
            "overlay": do_overlay,
        }
    return None


# ----------------------------------------------------------------------
@ray.remote(max_restarts=3, max_task_retries=0)
class IsaacSimWorkerActor:
    """One Isaac Sim app per process; init once, process multiple clips."""

    def __init__(
        self,
        urdf_path: str,
        headless: bool,
        render_size: int,
        render_scale: float,
        no_overlay: bool,
        video_path: str | None = None,
    ):
        self._app = None
        self._env = None
        self._urdf_path = urdf_path
        self._headless = headless
        self._render_size = render_size
        self._render_scale = render_scale
        self._no_overlay = no_overlay
        self._video_path = video_path
        self._init_app()

    def _init_app(self) -> None:
        from isaacsim import SimulationApp

        app_config = {
            "headless": self._headless,
            "width": 1280,
            "height": 720,
        }
        if self._headless:
            app_config["windowless"] = True
        self._app = SimulationApp(app_config)

        parser = build_parser()
        args = parser.parse_args([])
        args.urdf_path = self._urdf_path
        args.headless = self._headless
        args.render_size = self._render_size
        args.render_scale = self._render_scale
        args.render_width = args.render_height = self._render_size
        args.no_overlay = self._no_overlay
        self._env = IsaacSimEnv(args)

    def process_clips(
        self,
        clip_dirs: list[str],
        output_dir: str,
        worker_id: int,
    ) -> list[dict]:
        opts = {
            "output_dir": output_dir,
            "render_width": self._render_size,
            "render_height": self._render_size,
            "render_scale": self._render_scale,
            "no_overlay": self._no_overlay,
            "video_path": self._video_path,
        }
        results = []
        tag = f"[IsaacSim GPU{worker_id}]"
        for i, data_dir in enumerate(clip_dirs):
            clip_name = os.path.basename(data_dir.rstrip("/"))
            print(f"{tag} [{i+1}/{len(clip_dirs)}] {clip_name}", flush=True)
            try:
                stats = _render_single_clip(self._env, data_dir, opts)
                if stats is not None:
                    results.append(stats)
                    print(f"  Saved, FPS: {stats['render_fps']:.1f}", flush=True)
                else:
                    results.append({"clip_id": clip_name, "error": "render failed"})
            except Exception as e:
                results.append({"clip_id": clip_name, "error": str(e)})
                print(f"  Error: {e}", flush=True)
        return results

    def shutdown(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env = None
        self._app = None


def _run_ray_batch(args: argparse.Namespace, clip_dirs: list[str]) -> None:
    """Ray multi-GPU batch: one process per GPU, each init app once."""
    ray_kwargs = {}
    if getattr(args, "num_gpus", None) is not None:
        ray_kwargs["num_gpus"] = args.num_gpus
    ray.init(**ray_kwargs)

    cluster = ray.cluster_resources()
    total_gpus = cluster.get("GPU", 0)
    num_workers = max(1, int(total_gpus)) if total_gpus > 0 else 1
    gpus_per_worker = total_gpus / num_workers if total_gpus > 0 else 0

    worker_clips: list[list[str]] = [[] for _ in range(num_workers)]
    for i, cd in enumerate(clip_dirs):
        worker_clips[i % num_workers].append(cd)
    active: list[tuple[int, list[str]]] = [
        (w, clips) for w, clips in enumerate(worker_clips) if clips
    ]

    print(f"[IsaacSim] Ray batch: {len(clip_dirs)} clips, {len(active)} workers "
          f"({total_gpus} GPUs)")
    for w, clips in active:
        print(f"  Worker {w}: {len(clips)} clips")

    actor_options = {"num_cpus": 1.0}
    if gpus_per_worker > 0:
        actor_options["num_gpus"] = gpus_per_worker

    futures = []
    for w, clips in active:
        actor = IsaacSimWorkerActor.options(**actor_options).remote(
            urdf_path=args.urdf_path,
            headless=args.headless,
            render_size=getattr(args, "render_size", 512),
            render_scale=getattr(args, "render_scale", 1.0),
            no_overlay=args.no_overlay,
            video_path=getattr(args, "video_path", None),
        )
        fut = actor.process_clips.remote(clips, args.output_dir, w)
        futures.append((fut, w, actor))

    all_results = []
    for fut, w, actor in futures:
        try:
            worker_results = ray.get(fut)
            all_results.extend(worker_results)
            n_ok = sum(1 for r in worker_results if "error" not in r)
            print(f"[IsaacSim] Worker {w}: {n_ok}/{len(worker_results)} ok")
        except Exception as e:
            print(f"[IsaacSim] Worker {w} failed: {e}")
        try:
            ray.get(actor.shutdown.remote(), timeout=5)
        except Exception:
            pass

    summary = {
        "total": len(clip_dirs),
        "success": sum(1 for r in all_results if "error" not in r),
        "failed": sum(1 for r in all_results if "error" in r),
        "clips": all_results,
    }
    summary_path = os.path.join(args.output_dir, "render_stats.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[IsaacSim] Done: {summary['success']}/{summary['total']} ok -> {summary_path}")
    ray.shutdown()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import json

    parser = build_parser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Single clip directory (meta.json + joint_trajectory.json)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="IK output dir (outputs/ik) or data dir (outputs/ik/data). Process all clips within.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Total GPUs for Ray (default: all). One worker per GPU.",
    )
    args = parser.parse_args()

    # Determine clip directories
    clip_dirs: list[str] = []
    if args.data_dir and os.path.isdir(args.data_dir):
        meta_path = os.path.join(args.data_dir, "meta.json")
        if os.path.isfile(meta_path):
            clip_dirs = [os.path.abspath(args.data_dir)]
        else:
            # data_dir might be IK root (outputs/ik) or data parent (outputs/ik/data)
            clip_dirs = _gather_clip_dirs(args.data_dir)
            if clip_dirs:
                print(f"[IsaacSimEnv] Resolved --data-dir to {len(clip_dirs)} clips")
    elif args.input_dir and os.path.isdir(args.input_dir):
        clip_dirs = _gather_clip_dirs(args.input_dir)
    else:
        ik_root = os.path.join(os.path.dirname(__file__), "..", "outputs", "ik")
        clip_dirs = _gather_clip_dirs(ik_root)
        if not clip_dirs:
            raise FileNotFoundError(f"No clip directories in {ik_root} or {os.path.join(ik_root, 'data')}")
        if not args.input_dir and not args.data_dir:
            clip_dirs = [clip_dirs[0]]
            print(f"[IsaacSimEnv] Auto-selected single clip: {os.path.basename(clip_dirs[0])}")

    if not clip_dirs:
        raise FileNotFoundError("No valid clip directories")

    # Ray batch: input-dir + output-dir -> one process per GPU
    use_ray = args.input_dir and args.output_dir and len(clip_dirs) > 0
    if use_ray:
        _run_ray_batch(args, clip_dirs)
    else:
        # Single-process: one env, process clips sequentially
        data_dir = clip_dirs[0]
        with open(os.path.join(data_dir, "meta.json")) as f:
            meta = json.load(f)
        img_size = meta.get("img_size", [512, 512])
        render_size = getattr(args, "render_size", None)
        if render_size is not None and render_size > 0:
            args.render_width = args.render_height = render_size
        else:
            args.render_width = img_size[0]
            args.render_height = img_size[1]

        env = IsaacSimEnv(args)
        output_dir = getattr(args, "output_dir", None)
        do_render = output_dir and env._render_product is not None
        if do_render:
            os.makedirs(output_dir, exist_ok=True)

        all_stats = []
        for i, clip_dir in enumerate(clip_dirs):
            clip_name = os.path.basename(clip_dir.rstrip("/"))
            print(f"[IsaacSimEnv] [{i+1}/{len(clip_dirs)}] {clip_name}")
            stats = _render_single_clip(env, clip_dir, args)
            if stats is not None:
                all_stats.append(stats)
                if do_render:
                    print(f"  Saved, FPS: {stats['render_fps']:.1f}")

        if do_render and output_dir and all_stats:
            summary = {"clips": all_stats, "total_clips": len(all_stats), "failed": len(clip_dirs) - len(all_stats)}
            with open(os.path.join(output_dir, "render_stats.json"), "w") as f:
                json.dump(summary, f, indent=2)
        env.close()
