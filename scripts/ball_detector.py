#!/usr/bin/env python3
"""
Ball detector node — detects orange ball in camera feed
and publishes its 3D position using the point cloud.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')

        self.color_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            10
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/image_raw/points',
            self.cloud_callback,
            10
        )

        self.ball_pub = self.create_publisher(
            PointStamped,
            '/ball_position',
            10
        )

        self.bridge = CvBridge()
        self.latest_cloud = None
        self.get_logger().info('Ball detector node started')

    def color_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Wider orange range to handle shadows and lighting variation
        # Shadows darken the ball — lower V threshold catches them
        lower_orange = np.array([0,  100, 80])
        upper_orange = np.array([30, 255, 255])

        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Also catch darker orange in shadows
        lower_dark = np.array([0, 80, 40])
        upper_dark = np.array([20, 255, 120])
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

        # Combine masks
        mask = cv2.bitwise_or(mask, mask_dark)

        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask,  kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        debug = cv_image.copy()

        if not contours:
            cv2.imshow('Ball Detection', debug)
            cv2.waitKey(1)
            return

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 100:
            cv2.imshow('Ball Detection', debug)
            cv2.waitKey(1)
            return

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        (bx, by), radius = cv2.minEnclosingCircle(largest)
        cv2.circle(debug, (int(bx), int(by)), int(radius), (0, 255, 0), 2)
        cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            debug, f'({cx},{cy})',
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1
        )

        cv2.imshow('Ball Detection', debug)
        cv2.waitKey(1)

        self.get_logger().info(
            f'Ball detected at pixel ({cx},{cy}) area={area:.0f}'
        )

        if self.latest_cloud is not None:
            self.lookup_3d(cx, cy)

    def cloud_callback(self, msg):
        self.latest_cloud = msg

    def lookup_3d(self, u, v):
        """Get 3D position by reading raw bytes from PointCloud2."""
        try:
            cloud = self.latest_cloud
            w = cloud.width
            h = cloud.height
            point_step = cloud.point_step
            row_step = cloud.row_step

            # Find x y z field offsets
            field_offsets = {}
            for field in cloud.fields:
                field_offsets[field.name] = field.offset

            x_off = field_offsets['x']
            y_off = field_offsets['y']
            z_off = field_offsets['z']

            data = cloud.data

            valid = []

            # Sample 5x5 patch around centroid
            for du in range(-2, 3):
                for dv in range(-2, 3):
                    su = u + du
                    sv = v + dv
                    if not (0 <= su < w and 0 <= sv < h):
                        continue

                    # Calculate byte offset for this pixel
                    offset = sv * row_step + su * point_step

                    # Read x y z as float32
                    x = struct.unpack_from('f', data, offset + x_off)[0]
                    y = struct.unpack_from('f', data, offset + y_off)[0]
                    z = struct.unpack_from('f', data, offset + z_off)[0]

                    # Skip NaN and out of range
                    if np.isnan(x) or np.isnan(y) or np.isnan(z):
                        continue
                    if z < 0.05 or z > 5.0:
                        continue

                    valid.append((x, y, z))

            if not valid:
                self.get_logger().warn('No valid depth near ball centroid')
                return

            x = float(np.mean([p[0] for p in valid]))
            y = float(np.mean([p[1] for p in valid]))
            z = float(np.mean([p[2] for p in valid]))

            ball_msg = PointStamped()
            ball_msg.header.stamp = self.get_clock().now().to_msg()
            ball_msg.header.frame_id = 'camera_optical'
            ball_msg.point.x = x
            ball_msg.point.y = y
            ball_msg.point.z = z

            self.ball_pub.publish(ball_msg)

            self.get_logger().info(
                f'Ball 3D pos: x={x:.3f} y={y:.3f} z={z:.3f}m '
                f'({len(valid)} samples)'
            )

        except Exception as e:
            self.get_logger().error(f'Point cloud lookup error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    node = BallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()