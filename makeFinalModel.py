import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

# --- USER CONFIGURATION ---
BAG_FILE = "/mnt/c/Users/Daniel/PycharmProjects/fakeBuilding/scan_imu_20260203_145509_0.mcap"

# ROTATION 1: LIDAR (Targeting ROLL)
# Since Pitch didn't work, we go back to ROLL.
# Now that we've silenced the ghost, this should finally take effect.
LIDAR_ARGS = ['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '1.57']

# ROTATION 2: IMU (Targeting ROLL)
IMU_ARGS = ['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '1.57']


def generate_launch_description():
    # 1. Connect Base -> Lidar
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_tf',
        arguments=LIDAR_ARGS + ['--frame-id', 'base_link', '--child-frame-id', 'laser']
    )

    # 2. Connect Base -> IMU
    imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='imu_tf',
        arguments=IMU_ARGS + ['--frame-id', 'base_link', '--child-frame-id', 'imu_link']
    )

    # 3. RVIZ
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2'
    )

    # 4. Play Bag (THE FIX IS HERE)
    # We added '--topics /scan /imu/data'
    # This strips out the old /tf data so your new transforms are the only truth.
    play_bag = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', BAG_FILE,
            '--clock', '--loop',
            '--topics', '/scan', '/imu/data'
        ],
        output='screen'
    )

    return LaunchDescription([
        lidar_tf,
        imu_tf,
        rviz,
        play_bag
    ])