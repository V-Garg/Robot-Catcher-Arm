#!/usr/bin/env python3
"""
IK solver — one confident command per throw.
Waits for trajectory predictor confidence then sends
a single well-timed command and lets the arm execute.
"""
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener
import numpy as np


class IKSolver(Node):
    def __init__(self):
        super().__init__(
            'ik_solver',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, True)
            ]
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointStamped,
            '/ball_intercept_point',
            self.intercept_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        self.joint_names = [
            'BaseYaw',
            'ShoulderPitch',
            'ElbowPitch',
            'WristPitch',
            'HoopRotate',
        ]

        self.joint_limits = {
            'BaseYaw':       (-np.pi,   np.pi),
            'ShoulderPitch': (-np.pi/2, np.pi/2),
            'ElbowPitch':    (-np.pi/2, np.pi/2),
            'WristPitch':    (-np.pi/2, np.pi/2),
            'HoopRotate':    (-np.pi,   np.pi),
        }

        self.current_joints     = {name: 0.0 for name in self.joint_names}
        self.current_velocities = {name: 0.0 for name in self.joint_names}

        # Link lengths
        self.L1 = 0.207
        self.L2 = 0.250
        self.L3 = 0.250
        self.L4 = 0.184

        # Execution time — long enough for arm to reach without overshooting
        self.execution_time = 0.40

        # Intercept point history for confidence check
        # Only command once the predicted intercept has stabilized
        self.intercept_history  = []
        self.intercept_times    = []
        self.history_window     = 3      # need 5 consistent predictions
        self.stability_threshold = 0.12  # intercept must be stable within 5cm

        # Once commanded, don't resend unless intercept shifts significantly
        self.last_commanded_target  = None
        self.resend_threshold       = 0.15  # 8cm shift triggers resend
        self.last_command_time      = 0.0
        self.min_resend_interval    = 0.25  # don't resend faster than 350ms
        #  — give arm time to actually move before new command

        # Empirical correction
        self.correction_x = 0.0
        self.correction_y = 0.0
        self.correction_z = 0.0

        self.get_logger().info(
            f'IK solver ready. Max reach: {self.L2+self.L3+self.L4:.3f}m'
        )

    def joint_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.current_joints:
                self.current_joints[name]     = msg.position[i]
                self.current_velocities[name] = msg.velocity[i]

    def get_hoop_position(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'world', 'Hoop',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            return np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ])
        except Exception:
            return None

    def is_intercept_stable(self, tx, ty, tz):
        """
        Return True if the last N intercept predictions
        are all within stability_threshold of each other.
        Ensures we only command on a confident prediction.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        self.intercept_history.append([tx, ty, tz])
        self.intercept_times.append(now)

        # Keep only recent predictions
        while len(self.intercept_times) > 0 and \
              now - self.intercept_times[0] > 1.0:
            self.intercept_history.pop(0)
            self.intercept_times.pop(0)

        if len(self.intercept_history) < self.history_window:
            self.get_logger().info(
                f'Building confidence '
                f'({len(self.intercept_history)}/{self.history_window})'
            )
            return False

        # Check spread of last N predictions
        recent = np.array(self.intercept_history[-self.history_window:])
        spread = np.max(recent, axis=0) - np.min(recent, axis=0)
        max_spread = np.max(spread)

        if max_spread > self.stability_threshold:
            self.get_logger().info(
                f'Intercept unstable — spread {max_spread:.3f}m '
                f'> {self.stability_threshold}m'
            )
            return False

        return True

    def target_shifted_significantly(self, tx, ty, tz):
        """
        Check if the intercept point has moved enough from the
        last commanded position to warrant a correction command.
        """
        if self.last_commanded_target is None:
            return True
        diff = np.linalg.norm(
            np.array([tx, ty, tz]) - self.last_commanded_target
        )
        return diff > self.resend_threshold

    def solve_ik(self, tx, ty, tz):
        """
        Geometric IK with corrected joint directions.
        BaseYaw: positive = clockwise → j0 = atan2(-tx, -ty)
        ElbowPitch: positive = inward → j2 = pi - arccos(cos_j2)
        ShoulderPitch: negative = forward → sign unchanged
        """
        tx += self.correction_x
        ty += self.correction_y
        tz += self.correction_z

        # J0 BaseYaw — clockwise positive
        j0 = np.arctan2(-tx, -ty)

        r_horiz = np.sqrt(tx**2 + ty**2)
        dz = tz - self.L1
        dr = r_horiz
        d  = np.sqrt(dr**2 + dz**2)

        max_d = self.L2 + self.L3
        min_d = abs(self.L2 - self.L3) + 0.001

        if d > max_d:
            self.get_logger().warn(
                f'Target d={d:.3f}m > max {max_d:.3f}m — extending fully'
            )
            d = max_d - 0.005

        if d < min_d:
            self.get_logger().warn(f'Target too close d={d:.3f}m')
            return None

        # J2 ElbowPitch — positive = inward
        cos_j2 = (self.L2**2 + self.L3**2 - d**2) / (2 * self.L2 * self.L3)
        cos_j2 = np.clip(cos_j2, -1.0, 1.0)
        j2 = np.pi - np.arccos(cos_j2)

        # J1 ShoulderPitch — negative = forward
        alpha    = np.arctan2(dr, dz)
        cos_beta = (self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta     = np.arccos(cos_beta)
        j1       = -(alpha - beta)

        # J3 WristPitch — keep hoop level
        j3 = -(j1 + j2)

        j4 = 0.0

        angles = {
            'BaseYaw':       j0,
            'ShoulderPitch': j1,
            'ElbowPitch':    j2,
            'WristPitch':    j3,
            'HoopRotate':    j4,
        }

        for joint in angles:
            lo, hi  = self.joint_limits[joint]
            raw     = angles[joint]
            clamped = np.clip(raw, lo, hi)
            if abs(clamped - raw) > 0.05:
                self.get_logger().warn(
                    f'{joint} clamped {np.degrees(raw):.1f}° '
                    f'→ {np.degrees(clamped):.1f}°'
                )
            angles[joint] = clamped

        self.get_logger().info(
            f'IK: J0={np.degrees(j0):.1f}° '
            f'J1={np.degrees(j1):.1f}° '
            f'J2={np.degrees(j2):.1f}° '
            f'J3={np.degrees(j3):.1f}°'
        )

        return angles

    def intercept_callback(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9

        tx = msg.point.x
        ty = msg.point.y
        tz = msg.point.z

        # Log hoop vs target
        hoop_pos = self.get_hoop_position()
        if hoop_pos is not None:
            error = np.array([tx, ty, tz]) - hoop_pos
            self.get_logger().info(
                f'Hoop: ({hoop_pos[0]:.3f},{hoop_pos[1]:.3f},{hoop_pos[2]:.3f}) '
                f'Target: ({tx:.3f},{ty:.3f},{tz:.3f}) '
                f'Error: ({error[0]:.3f},{error[1]:.3f},{error[2]:.3f})'
            )

        # Gate 1 — wait for stable intercept prediction
        if not self.is_intercept_stable(tx, ty, tz):
            return

        # Gate 2 — don't resend too fast
        if now - self.last_command_time < self.min_resend_interval:
            return

        # Gate 3 — only resend if target shifted significantly
        if not self.target_shifted_significantly(tx, ty, tz):
            self.get_logger().info(
                f'Target stable — no resend needed '
                f'(shift < {self.resend_threshold}m)'
            )
            return

        angles = self.solve_ik(tx, ty, tz)
        if angles is None:
            return

        self.last_command_time     = now
        self.last_commanded_target = np.array([tx, ty, tz])
        self.send_trajectory(angles)

    def send_trajectory(self, angles):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names  = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [
            float(angles[name]) for name in self.joint_names
        ]
        point.velocities = [0.0] * len(self.joint_names)
        point.time_from_start.sec     = 0
        point.time_from_start.nanosec = int(self.execution_time * 1e9)

        msg.points = [point]
        self.traj_pub.publish(msg)

        self.get_logger().info(
            f'COMMAND SENT: '
            f'[{", ".join(f"{np.degrees(p):.1f}°" for p in point.positions)}] '
            f'in {self.execution_time*1000:.0f}ms'
        )


def main(args=None):
    rclpy.init(args=args)
    node = IKSolver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()