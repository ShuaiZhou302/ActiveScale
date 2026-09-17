"""Pinocchio/CasADi inverse kinematics used by the Quest 2 teleop server."""

import os

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from tf.transformations import quaternion_from_euler


class PinocchioIKSolver:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        absolute_path = os.path.abspath(urdf_path)
        package_index = absolute_path.find(os.sep + "piper_description" + os.sep)
        package_dirs = [absolute_path[:package_index]] if package_index >= 0 else []

        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=["joint7", "joint8"],
            reference_configuration=np.zeros(self.robot.model.nq),
        )

        ee_quaternion = quaternion_from_euler(0.0, -np.pi / 2.0, 0.0)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "ee",
                self.reduced_robot.model.getJointId("joint6"),
                pin.SE3(
                    pin.Quaternion(
                        ee_quaternion[3],
                        ee_quaternion[0],
                        ee_quaternion[1],
                        ee_quaternion[2],
                    ),
                    np.zeros(3),
                ),
                pin.FrameType.OP_FRAME,
            )
        )

        self.model = self.reduced_robot.model
        self.data = self.model.createData()
        self.reduced_robot.data = self.data
        self.init_data = np.zeros(self.model.nq)
        self.history_data = np.zeros(self.model.nq)

        self.geom_model = pin.buildGeomFromUrdf(
            self.robot.model,
            urdf_path,
            pin.GeometryType.COLLISION,
            package_dirs=package_dirs,
        )
        geometry_count = len(self.geom_model.geometryObjects)
        for first in range(4, 10):
            for second in range(3):
                if first < geometry_count and second < geometry_count and first != second:
                    self.geom_model.addCollisionPair(pin.CollisionPair(first, second))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.ee_frame_id = self.model.getFrameId("ee")
        self.error_func = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.ee_frame_id].inverse() * cpin.SE3(self.cTf)
                    ).vector
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        error = self.error_func(self.var_q, self.param_tf)
        pose_cost = casadi.sumsqr(error[:3]) + casadi.sumsqr(0.1 * error[3:])
        self.opti.subject_to(
            self.opti.bounded(
                self.model.lowerPositionLimit,
                self.var_q,
                self.model.upperPositionLimit,
            )
        )
        self.opti.minimize(20.0 * pose_cost + 0.01 * casadi.sumsqr(self.var_q))
        self.opti.solver(
            "ipopt",
            {
                "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-4},
                "print_time": False,
            },
        )

    def check_self_collision(self, joints, gripper=0.0):
        gripper_joints = np.array([gripper / 2.0, -gripper / 2.0])
        full_joints = np.concatenate([np.asarray(joints), gripper_joints])
        pin.forwardKinematics(self.robot.model, self.robot.data, full_joints)
        pin.updateGeometryPlacements(
            self.robot.model,
            self.robot.data,
            self.geom_model,
            self.geometry_data,
        )
        return pin.computeCollisions(self.geom_model, self.geometry_data, False)

    def solve(self, xyz, rpy, gripper=0.0, motorstate=None, allow_collision=False):
        quaternion = quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        target = pin.SE3(
            pin.Quaternion(quaternion[3], quaternion[0], quaternion[1], quaternion[2]),
            np.asarray(xyz),
        )
        if motorstate is not None:
            self.init_data = np.asarray(motorstate, dtype=float).copy()
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf, target.homogeneous)

        try:
            self.opti.solve_limited()
            joints = np.asarray(self.opti.value(self.var_q)).reshape(-1)
        except Exception:
            return None, False, "ik_failed"

        self.init_data = joints.copy()
        self.history_data = joints.copy()
        collision = self.check_self_collision(joints, gripper)
        if collision and not allow_collision:
            return None, False, "self_collision"
        return joints[:6], True, "ok"
