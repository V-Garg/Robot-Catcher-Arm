#!/usr/bin/env python3
"""
Trajectory predictor — improved accuracy with throw detection.
"""
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
import numpy as np


class TrajectoryPredictor(Node):
    def __init__(self):
        super().__init__(
            'trajectory_predictor',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, True)
            ]
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointStamped,
            '/ball_position',
            self.ball_callback,
            10
        )

        self.intercept_pub = self.create_publisher(
            PointStamped, '/ball_intercept_point', 10
        )
        self.predicted_pub = self.create_publisher(
            PointStamped, '/ball_predicted_position', 10
        )
        self.world_ball_pub = self.create_publisher(
            PointStamped, '/ball_position_world', 10
        )

        # Kalman state [x, y, z, vx, vy, vz]
        self.state = None
        self.P = None
        self.initialized = False
        self.last_time = None

        self.g = 9.81

        self.R = np.diag([0.005, 0.005, 0.005])
        self.Q_pos = 0.0001
        self.Q_vel = 0.05

        # Rolling window
        self.position_history = []
        self.time_history = []
        self.window_size = 8
        self.min_measurements = 4

        # Throw detection thresholds
        # Ball must be moving horizontally above this speed to be intercepted
        self.min_horizontal_speed = 0.3  # m/s

        # Ball must be moving toward the arm (negative y in world frame)
        # or at least have significant horizontal motion
        self.min_speed_total = 0.5  # m/s total speed

        # If ball jumps more than this between frames, reset filter
        self.jump_threshold = 0.3  # m

        # Arm reach
        self.arm_max_reach = 0.684
        self.min_predict_ahead = 0.1
        self.max_predict_ahead = 0.5

        self.get_logger().info('Trajectory predictor started')

    def reset_filter(self, x, y, z, now):
        """Reset Kalman filter — call when new throw detected."""
        self.state = np.array([x, y, z, 0.0, 0.0, 0.0])
        self.P = np.eye(6) * 1.0
        self.last_time = now
        self.initialized = True
        self.position_history = [[x, y, z]]
        self.time_history = [now]
        self.get_logger().info('Filter reset — new throw detected')

    def transform_to_world(self, point_stamped):
        try:
            transform = self.tf_buffer.lookup_transform(
                'world',
                point_stamped.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            world_point = do_transform_point(point_stamped, transform)
            world_point.header.stamp = self.get_clock().now().to_msg()
            return world_point
        except Exception as e:
            self.get_logger().warn(f'TF transform failed: {e}')
            return None

    def estimate_velocity_from_window(self):
        """Linear regression velocity estimate from position window."""
        if len(self.position_history) < 3:
            return None

        times = np.array(self.time_history)
        positions = np.array(self.position_history)
        t = times - times[0]

        velocities = []
        for axis in range(3):
            weights = np.linspace(0.5, 1.0, len(t))
            A = np.vstack([t, np.ones(len(t))]).T
            W = np.diag(weights)
            result = np.linalg.lstsq(W @ A, W @ positions[:, axis], rcond=None)
            velocities.append(result[0][0])

        return np.array(velocities)

    def compute_intercept_time(self, px, py, pz, vx, vy, vz):
        """Find earliest time ball is within arm reach and above ground."""
        for t in np.arange(self.min_predict_ahead,
                           self.max_predict_ahead + 0.01, 0.02):
            fx = px + vx * t
            fy = py + vy * t
            fz = pz + vz * t - 0.5 * self.g * t**2

            if fz < 0.15:
                continue

            dist = np.sqrt(fx**2 + fy**2 + fz**2)
            if dist < self.arm_max_reach:
                return t

        return 0.3

    def ball_callback(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9

        world_msg = self.transform_to_world(msg)
        if world_msg is None:
            return

        self.world_ball_pub.publish(world_msg)

        x = world_msg.point.x
        y = world_msg.point.y
        z = world_msg.point.z

        # Detect jump — ball was picked up or respawned
        if self.initialized and len(self.position_history) > 0:
            last_pos = np.array(self.position_history[-1])
            curr_pos = np.array([x, y, z])
            jump = np.linalg.norm(curr_pos - last_pos)
            if jump > self.jump_threshold:
                self.get_logger().info(
                    f'Ball jumped {jump:.3f}m — resetting filter'
                )
                self.reset_filter(x, y, z, now)
                return

        # Add to rolling window
        self.position_history.append([x, y, z])
        self.time_history.append(now)

        if len(self.position_history) > self.window_size:
            self.position_history.pop(0)
            self.time_history.pop(0)

        if not self.initialized:
            self.reset_filter(x, y, z, now)
            return

        dt = now - self.last_time
        self.last_time = now

        if dt <= 0 or dt > 1.0:
            return

        # === PREDICT STEP ===
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.state = F @ self.state

        Q = np.zeros((6, 6))
        Q[0, 0] = self.Q_pos * dt**2
        Q[1, 1] = self.Q_pos * dt**2
        Q[2, 2] = self.Q_pos * dt**2
        Q[3, 3] = self.Q_vel * dt
        Q[4, 4] = self.Q_vel * dt
        Q[5, 5] = self.Q_vel * dt
        self.P = F @ self.P @ F.T + Q

        # === UPDATE STEP ===
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        z_meas = np.array([x, y, z])
        y_innov = z_meas - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_innov
        self.P = (np.eye(6) - K @ H) @ self.P

        # Blend with window regression
        if len(self.position_history) >= self.min_measurements:
            window_vel = self.estimate_velocity_from_window()
            if window_vel is not None:
                alpha = 0.4
                self.state[3] = (1-alpha)*self.state[3] + alpha*window_vel[0]
                self.state[4] = (1-alpha)*self.state[4] + alpha*window_vel[1]
                self.state[5] = (1-alpha)*self.state[5] + alpha*window_vel[2]

        vx = self.state[3]
        vy = self.state[4]
        vz = self.state[5]

        horiz_speed = np.sqrt(vx**2 + vy**2)
        total_speed = np.sqrt(vx**2 + vy**2 + vz**2)

        self.get_logger().info(
            f'Ball: ({x:.3f},{y:.3f},{z:.3f}) '
            f'vel: ({vx:.3f},{vy:.3f},{vz:.3f}) '
            f'horiz={horiz_speed:.3f} total={total_speed:.3f}'
        )

        # Gate 1 — need enough measurements
        if len(self.position_history) < self.min_measurements:
            self.get_logger().info(
                f'Waiting for measurements '
                f'({len(self.position_history)}/{self.min_measurements})'
            )
            return

        # Gate 4 — ball must be above minimum height
        if z < 0.1:
            self.get_logger().info('Ball below minimum height')
            return

        self.predict_intercept()

    def predict_intercept(self):
        if not self.initialized:
            return

        px = self.state[0]
        py = self.state[1]
        pz = self.state[2]
        vx = self.state[3]
        vy = self.state[4]
        vz = self.state[5]

        t = self.compute_intercept_time(px, py, pz, vx, vy, vz)

        future_x = px + vx * t
        future_y = py + vy * t
        future_z = pz + vz * t - 0.5 * self.g * t**2

        if future_z < 0.05:
            self.get_logger().warn(
                f'Ball hits ground in {t:.2f}s — cannot intercept'
            )
            return

        stamp = self.get_clock().now().to_msg()

        pred_msg = PointStamped()
        pred_msg.header.stamp = stamp
        pred_msg.header.frame_id = 'world'
        pred_msg.point.x = future_x
        pred_msg.point.y = future_y
        pred_msg.point.z = future_z
        self.predicted_pub.publish(pred_msg)

        intercept_msg = PointStamped()
        intercept_msg.header.stamp = stamp
        intercept_msg.header.frame_id = 'world'
        intercept_msg.point.x = future_x
        intercept_msg.point.y = future_y
        intercept_msg.point.z = future_z
        self.intercept_pub.publish(intercept_msg)

        self.get_logger().info(
            f'Intercept in {t:.2f}s: '
            f'({future_x:.3f},{future_y:.3f},{future_z:.3f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()