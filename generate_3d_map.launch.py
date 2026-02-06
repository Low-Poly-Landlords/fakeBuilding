import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

# --- PATHS ---
BAG_FILE = "/mnt/c/Users/Daniel/PycharmProjects/fakeBuilding/scan_imu_20260203_145509_0.mcap"
CONFIG_DIR = os.path.expanduser('~/carto_ws/config')
CONFIG_FILE = 'my_3d_lidar.lua'

# --- CALIBRATION (VERIFIED) ---
# Lidar: Vertical Ring -> Pitch 90 (or Roll 90 depending on Lidar model)
# You found '--roll 1.57' gave the Vertical Ring.
LIDAR_ARGS = ['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '1.57']

# IMU: Green Arrow Up -> Roll 90
IMU_ARGS = ['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '1.57']

def generate_launch_description():
    
    # 1. Start Cartographer
    cartographer = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        arguments=[
            '-configuration_directory', CONFIG_DIR,
            '-configuration_basename', CONFIG_FILE
        ],
        remappings=[
            ('/imu', '/imu/data'),
            ('/points2', '/scan') # Just in case it looks for points2
        ]
    )

    # 2. Occupancy Grid (Generates the 2D floor plan too)
    occupancy = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='occupancy_node',
        arguments=['-resolution', '0.05']
    )

    # 3. Static Transforms (The Calibration)
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=LIDAR_ARGS + ['--frame-id', 'base_link', '--child-frame-id', 'laser']
    )

    imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=IMU_ARGS + ['--frame-id', 'base_link', '--child-frame-id', 'imu_link']
    )

    # 4. Play Bag (Filtered)
    # We strip the old TF to prevent "Ghosting"
    play_bag = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', BAG_FILE,
            '--clock',
            '--topics', '/scan', '/imu/data'
        ],
        output='screen'
    )
    
    # 5. Rviz (So you can watch it build!)
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2'
    )

    return LaunchDescription([
        lidar_tf, imu_tf, cartographer, occupancy, play_bag, rviz
    ])
