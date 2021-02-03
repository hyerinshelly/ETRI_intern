import carb
import omni
import numpy as np

from pxr import UsdGeom, Gf
from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
from omni.isaac.dynamic_control import _dynamic_control

class STR:
    def __init__(self, omni_kit):
        self.omni_kit = omni_kit

        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return
        self.usd_path = nucleus_server + "/Isaac/Robots/STR/STR_V4_Physics_Caster.usd"
        self.robot_prim = None
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.ar = None

    # rotation is in degrees
    def spawn(self, location, rotation):
        stage = self.omni_kit.get_stage()
        prefix = "/World/Robot/STR"
        prim_path = omni.kit.utils.get_stage_next_free_path(stage, prefix, False)
        print(prim_path)
        self.robot_prim = stage.DefinePrim(prim_path, "Xform")
        self.robot_prim.GetReferences().AddReference(self.usd_path)
        xform = UsdGeom.Xformable(self.robot_prim)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        mat = Gf.Matrix4d().SetTranslate(location)
        mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation))
        xform_op.Set(mat)

        self.camera_path = prim_path + "/STR/Vehicle/str_camera"

    def teleport(self, location, rotation, settle=False):
        if self.ar is None:
            self.ar = self.dc.get_rigid_body(self.robot_prim.GetPath().pathString + "/Vehicle")
            self.chassis = self.ar
        self.dc.wake_up_rigid_body(self.ar)
        rot_quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation).GetQuaternion()

        tf = _dynamic_control.Transform(
            location,
            (rot_quat.GetImaginary()[0], rot_quat.GetImaginary()[1], rot_quat.GetImaginary()[2], rot_quat.GetReal()),
        )
        self.dc.set_rigid_body_pose(self.chassis, tf)
        self.dc.set_rigid_body_linear_velocity(self.chassis, [0, 0, 0])
        self.dc.set_rigid_body_angular_velocity(self.chassis, [0, 0, 0])
        self.command((0, 0))
        if settle:
            frame = 0
            velocity = 1
            print("Settling robot...")
            while velocity > 0.1 and frame < 120:
                self.omni_kit.update(1.0 / 60.0)
                lin_vel = self.dc.get_rigid_body_linear_velocity(self.chassis)
                velocity = np.linalg.norm([lin_vel.x, lin_vel.y, lin_vel.z])
                # print("velocity magnitude is: ", velocity)
                frame = frame + 1
            # print("done after frame: HERE", frame)

    def activate_camera(self):
        vpi = omni.kit.viewport.get_viewport_interface()
        vpi.get_viewport_window().set_active_camera(str(self.camera_path))

    def command(self, motor_value):
        if self.ar is None:
            self.ar = self.dc.get_articulation(self.robot_prim.GetPath().pathString)
            self.chassis = self.dc.get_articulation_root_body(self.ar)
            self.wheel_left = self.dc.find_articulation_dof(self.ar, "left_wheel_joint")
            self.wheel_right = self.dc.find_articulation_dof(self.ar, "right_wheel_joint")
        self.dc.wake_up_articulation(self.ar)
        left_speed = self.wheel_speed_from_motor_value(motor_value[0])
        right_speed = self.wheel_speed_from_motor_value(motor_value[1])
        self.dc.set_dof_velocity_target(self.wheel_left, np.clip(left_speed, -1000, 1000))
        self.dc.set_dof_velocity_target(self.wheel_right, np.clip(right_speed, -1000, 1000))

    # idealized motor model
    def wheel_speed_from_motor_value(self, input):
        return input * 2.0

    def observations(self):
        if self.ar is None:
            self.ar = self.dc.get_rigid_body(self.robot_prim.GetPath().pathString + "/Vehicle")
            self.chassis = self.ar
        dc_pose = self.dc.get_rigid_body_pose(self.chassis)
        dc_lin_vel = self.dc.get_rigid_body_linear_velocity(self.chassis)
        dc_local_lin_vel = self.dc.get_rigid_body_local_linear_velocity(self.chassis)
        dc_ang_vel = self.dc.get_rigid_body_angular_velocity(self.chassis)
        return {
            "pose": (dc_pose.p.x, dc_pose.p.y, dc_pose.p.z, dc_pose.r.w, dc_pose.r.x, dc_pose.r.y, dc_pose.r.z),
            "linear_velocity": (dc_lin_vel.x, dc_lin_vel.y, dc_lin_vel.z),
            "local_linear_velocity": (dc_local_lin_vel.x, dc_local_lin_vel.y, dc_local_lin_vel.z),
            "angular_velocity": (dc_ang_vel.x, dc_ang_vel.y, dc_ang_vel.z),
        }