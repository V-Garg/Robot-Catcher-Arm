import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg = get_package_share_directory('arm')
    urdf = os.path.join(pkg, 'urdf', 'arm.urdf')
    world = os.path.join(pkg, 'worlds', 'ball_world.sdf')

    with open(urdf, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([

        # Set resource path so Gazebo finds meshes
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=os.path.join(
                os.path.expanduser('~'),
                'ros2_ws/install/arm/share'
            )
        ),

        # Set plugin path so Gazebo finds ros2_control plugin
        SetEnvironmentVariable(
            name='GZ_SIM_SYSTEM_PLUGIN_PATH',
            value='/opt/ros/jazzy/lib'
        ),

        # Start Gazebo with ball world
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world],
            output='screen'
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True
            }]
        ),

        # Bridge Gazebo topics to ROS2
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                '/camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                '/camera/depth/image_raw/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            ],
            output='screen'
        ),

        # Spawn arm in Gazebo
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'arm',
                '-topic', 'robot_description',
                '-x', '0', '-y', '0', '-z', '1.0'
            ],
            output='screen'
        ),

        # Wait for Gazebo + plugin to initialize then spawn controllers
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=[
                        'joint_state_broadcaster',
                        '--controller-manager', '/controller_manager'
                    ],
                    output='screen'
                ),
            ]
        ),

        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=[
                        'arm_controller',
                        '--controller-manager', '/controller_manager'
                    ],
                    output='screen'
                ),
            ]
        ),

        # Wait for controllers then start perception pipeline
        TimerAction(
            period=12.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'python3',
                        os.path.join(
                            os.path.expanduser('~'),
                            'ros2_ws/src/arm/scripts/ball_detector.py'
                        )
                    ],
                    output='screen'
                ),
            ]
        ),

        TimerAction(
            period=13.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'python3',
                        os.path.join(
                            os.path.expanduser('~'),
                            'ros2_ws/src/arm/scripts/trajectory_predictor.py'
                        )
                    ],
                    output='screen'
                ),
            ]
        ),

        TimerAction(
            period=14.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'python3',
                        os.path.join(
                            os.path.expanduser('~'),
                            'ros2_ws/src/arm/scripts/ik_solver.py'
                        )
                    ],
                    output='screen'
                ),
            ]
        ),
    ])