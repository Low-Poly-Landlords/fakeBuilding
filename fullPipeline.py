import os
import sys
import argparse
import numpy as np
import open3d as o3d
import bisect
from scipy.spatial.transform import Rotation as R
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# --- CONFIGURATION ---
# Tuning for the "Solid Block" generation
VOXEL_SIZE = 0.05  # 5cm grid for speed
RANSAC_DIST = 0.05  # 5cm tolerance for wall flatness
PARALLEL_THRESH = 0.9  # 0.9 = Roughly parallel
MERGE_DIST = 0.20  # Merge walls within 20cm
MIN_POINTS = 100  # Ignore tiny clusters
WALL_ITERATIONS = 50  # Max walls to find

# --- CALIBRATION (HARDCODED) ---
# Lidar: Roll 90 deg
LIDAR_FIX = R.from_euler('x', 90, degrees=True).as_matrix()
# IMU: Roll 90 deg
IMU_FIX = R.from_euler('x', 90, degrees=True)


def get_interpolated_imu(target_time, imu_data):
    """Finds the IMU rotation closest to the requested timestamp."""
    times = [x[0] for x in imu_data]
    idx = bisect.bisect_left(times, target_time)
    if idx == 0: return imu_data[0][1]
    if idx == len(times): return imu_data[-1][1]

    before = times[idx - 1]
    after = times[idx]
    if after - target_time < target_time - before:
        return imu_data[idx][1]
    else:
        return imu_data[idx - 1][1]


def laserscan_to_points(msg):
    """Convert 2D scan to 3D points (flat on z=0)."""
    angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
    count = min(len(angles), len(msg.ranges))
    ranges = np.array(msg.ranges[:count])
    angles = angles[:count]
    mask = (ranges > msg.range_min) & (ranges < msg.range_max)
    r = ranges[mask]
    a = angles[mask]
    x = r * np.cos(a)
    y = r * np.sin(a)
    z = np.zeros_like(x)
    return np.column_stack((x, y, z))


def step_1_mcap_to_ply(input_mcap, output_ply):
    print(f"\n[STEP 1/2] Processing MCAP: {input_mcap}")
    reader = make_reader(open(input_mcap, "rb"), decoder_factories=[DecoderFactory()])

    imu_timeline = []
    lidar_msgs = []

    # Read Data
    print("   Reading packets...")
    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic == "/imu/data":
            q = [ros_msg.orientation.x, ros_msg.orientation.y, ros_msg.orientation.z, ros_msg.orientation.w]
            imu_timeline.append((message.log_time, q))
        elif channel.topic == "/scan":
            lidar_msgs.append((message.log_time, ros_msg))

    print(f"   Loaded {len(imu_timeline)} IMU, {len(lidar_msgs)} Scans.")

    global_pcd = o3d.geometry.PointCloud()

    # Fuse Data
    print("   Building Point Cloud...")
    for i, (log_time, scan_msg) in enumerate(lidar_msgs):
        local_points = laserscan_to_points(scan_msg)
        if len(local_points) == 0: continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(local_points)

        # Apply Calibrations
        pcd.rotate(LIDAR_FIX, center=(0, 0, 0))
        raw_quat = get_interpolated_imu(log_time, imu_timeline)
        final_rot = R.from_quat(raw_quat) * IMU_FIX
        pcd.rotate(final_rot.as_matrix(), center=(0, 0, 0))

        global_pcd += pcd

    print(f"   Saving intermediate cloud to {output_ply}...")
    global_pcd = global_pcd.voxel_down_sample(voxel_size=0.03)
    o3d.io.write_point_cloud(output_ply, global_pcd)


def step_2_ply_to_mesh(input_ply, output_obj):
    print(f"\n[STEP 2/2] Generating Solid Mesh: {output_obj}")
    pcd = o3d.io.read_point_cloud(input_ply)
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    detected_planes = []  # List of dicts: {'equation': [], 'points': []}

    print("   Finding and Merging Walls...")
    for i in range(WALL_ITERATIONS):
        if len(pcd.points) < MIN_POINTS: break

        # RANSAC
        plane_model, inliers = pcd.segment_plane(distance_threshold=RANSAC_DIST,
                                                 ransac_n=3, num_iterations=1000)

        # Extract Candidate
        candidate_cloud = pcd.select_by_index(inliers)
        candidate_points = np.asarray(candidate_cloud.points)
        candidate_center = np.mean(candidate_points, axis=0)
        normal_candidate = np.array(plane_model[:3])

        # Merge Logic
        merged = False
        for existing in detected_planes:
            normal_existing = np.array(existing['equation'][:3])

            # Parallel Check
            if abs(np.dot(normal_candidate, normal_existing)) > PARALLEL_THRESH:
                # Distance Check
                dist = abs(np.dot(normal_existing, candidate_center) + existing['equation'][3])
                if dist < MERGE_DIST:
                    existing['points'].extend(candidate_points)
                    merged = True
                    break

        if not merged:
            detected_planes.append({'equation': plane_model, 'points': list(candidate_points)})

        # Remove points to find next wall
        pcd = pcd.select_by_index(inliers, invert=True)

    print(f"   Constructing {len(detected_planes)} solid blocks...")
    final_mesh = o3d.geometry.TriangleMesh()

    for plane in detected_planes:
        wall_pcd = o3d.geometry.PointCloud()
        wall_pcd.points = o3d.utility.Vector3dVector(plane['points'])

        # Create Solid Box (OBB)
        obb = wall_pcd.get_oriented_bounding_box()
        mesh_box = o3d.geometry.TriangleMesh.create_box(
            width=obb.extent[0], height=obb.extent[1], depth=obb.extent[2])

        mesh_box.translate(-mesh_box.get_center())
        mesh_box.rotate(obb.R, center=[0, 0, 0])
        mesh_box.translate(obb.center)
        mesh_box.compute_vertex_normals()

        # Color: Grey for floors (flat), Blue for walls (tall)
        if obb.extent[2] < obb.extent[0] and obb.extent[2] < obb.extent[1]:
            mesh_box.paint_uniform_color([0.7, 0.7, 0.7])
        else:
            mesh_box.paint_uniform_color([0.2, 0.5, 0.8])

        final_mesh += mesh_box

    o3d.io.write_triangle_mesh(output_obj, final_mesh)
    print("   Done!")
    return final_mesh


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <filename.mcap>")
        return

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return

    # Define filenames
    base_name = os.path.splitext(input_file)[0]
    ply_file = base_name + "_cloud.ply"
    obj_file = base_name + "_final.obj"

    # RUN PIPELINE
    step_1_mcap_to_ply(input_file, ply_file)
    final_mesh = step_2_ply_to_mesh(ply_file, obj_file)

    # VISUALIZE
    print("\nOpening Viewer...")
    o3d.visualization.draw_geometries([final_mesh], window_name="Final Result")


if __name__ == "__main__":
    main()