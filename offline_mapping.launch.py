import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

# --- CONFIGURATION ---
# UPDATE THIS PATH to match your actual file location inside WSL
BAG_FILE = "/mnt/c/Users/Daniel/PycharmProjects/fakeBuilding/scan_imu_20260203_145509_0.mcap"

def generate_launch_description():
    
    # 1. Static Transform: odom -> base_link (The Fake Wheels)
    # We strictly enforce use_sim_time here
    fake_wheels = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
        parameters=[{'use_sim_time': True}]
    )

    # 2. Static Transform: base_link -> laser (The Sensor Mount)
    sensor_mount = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'laser'],
        parameters=[{'use_sim_time': True}]
    )

    # 3. SLAM Toolbox
    slam = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'base_frame': 'base_link',
            'odom_frame': 'odom',
            'map_frame': 'map',
            'scan_topic': '/scan'
        }]
    )

    # 4. Bag Player
    # We delay this by 5 seconds to ensure SLAM and TFs are ready
    play_bag = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'bag', 'play', BAG_FILE, '--clock'],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        fake_wheels,
        sensor_mount,
        slam,
        play_bag
    ])
