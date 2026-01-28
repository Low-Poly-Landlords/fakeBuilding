#run this 3rd. It takes the lidar sim and turns it into a 3d model. this is simulating
#what our final lidar to model program will look like.

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mcap.reader import make_reader
from rosbags.typesys import get_typestore, Stores
from scipy.spatial.transform import Rotation as R

INPUT_FILENAME = "hotel_scan.mcap"
OUTPUT_FILENAME = "hotel_reconstruction.ply"

# --- CONFIG ---
VOXEL_SIZE = 0.05  # 5cm resolution (Lower = Higher Detail, Slower)
POISSON_DEPTH = 9  # Reconstruction detail level (8-10 is standard)


def main():
    print(f"Reconstructing 3D Model from {INPUT_FILENAME}...")
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    global_pcd = o3d.geometry.PointCloud()
    pose_map = {}

    # --- STEP 1: LOAD DATA ---
    print("Reading MCAP stream (this may take a minute)...")

    # We need to buffer poses first because TF messages might arrive slightly
    # before or after the LiDAR messages they correspond to.

    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)

        # Pass 1: Read Transforms (TF)
        # We build a lookup table of Timestamp -> Matrix
        for schema, channel, message in reader.iter_messages(topics=["/tf"]):
            msg = typestore.deserialize_cdr(message.data, "tf2_msgs/msg/TFMessage")
            for t in msg.transforms:
                # Extract Translation
                trans = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
                # Extract Rotation (Quaternion)
                quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]

                # Create 4x4 Matrix
                rot_mat = R.from_quat(quat).as_matrix()
                pose_mat = np.identity(4)
                pose_mat[:3, :3] = rot_mat
                pose_mat[:3, 3] = trans

                # Store by log time
                pose_map[message.log_time] = pose_mat

    print(f"Loaded {len(pose_map)} poses. Stitching clouds...")

    # Pass 2: Read LiDAR and Transform
    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)

        frame_count = 0
        for schema, channel, message in reader.iter_messages(topics=["/lidar/points"]):
            # Find closest pose
            # (In a real robot, we'd interpolate. Here, nearest neighbor is fine)
            if not pose_map: break

            # Find key in pose_map closest to message.log_time
            # This is slow O(N), but fine for our scale.
            # Optimization: Poses are sorted, could use bisect.
            closest_ts = min(pose_map.keys(), key=lambda k: abs(k - message.log_time))

            # If time diff is too large (>0.1s), skip (sync issue)
            if abs(closest_ts - message.log_time) > 1e8:
                continue

            pose = pose_map[closest_ts]

            # Deserialize PointCloud2
            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/PointCloud2")
            raw_data = np.frombuffer(msg.data, dtype=np.uint8)
            xyz_points = raw_data.view(np.float32).reshape((-1, 3))

            # Convert to Open3D
            curr_pcd = o3d.geometry.PointCloud()
            curr_pcd.points = o3d.utility.Vector3dVector(xyz_points)

            # Transform Local -> Global
            curr_pcd.transform(pose)

            # Add to global map
            global_pcd += curr_pcd

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Stitched {frame_count} frames...", end="\r")

    print(f"\nCloud assembled: {len(global_pcd.points)} points.")

    # --- STEP 2: CLEANUP ---
    print("Downsampling and estimating normals...")
    # 1. Voxel Downsample (Merges points in same 5cm cube)
    # This massively reduces memory usage and creates a uniform grid
    global_pcd = global_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"Downsampled to {len(global_pcd.points)} points.")

    # 2. Estimate Normals (Crucial for Poisson)
    global_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30))

    # 3. Orient Normals
    # We assume the "camera" was roughly at (0,0,0) originally, but for a whole building
    # that's tricky. Open3D's tangent plane consistency is usually best for buildings.
    global_pcd.orient_normals_consistent_tangent_plane(k=20)

    # --- STEP 3: RECONSTRUCTION ---
    print(f"Running Poisson Reconstruction (Depth={POISSON_DEPTH})...")
    # This math creates a watertight surface from the points
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(global_pcd, depth=POISSON_DEPTH)

    # --- STEP 4: POST-PROCESSING ---
    print("Cleaning artifacts...")
    # Poisson creates a "bubble" around the data. We remove low-density ghost geometry.
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Color the mesh based on Height (Z)
    print("Coloring mesh...")
    vertices = np.asarray(mesh.vertices)
    z_values = vertices[:, 2]
    z_min, z_max = np.min(z_values), np.max(z_values)
    z_norm = (z_values - z_min) / (z_max - z_min + 1e-6)

    colormap = plt.get_cmap("jet")
    colors = colormap(z_norm)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()

    # --- SAVE AND SHOW ---
    o3d.io.write_triangle_mesh(OUTPUT_FILENAME, mesh)
    print(f"Saved solid mesh to {OUTPUT_FILENAME}")

    print("Opening Viewer...")
    o3d.visualization.draw_geometries([mesh], window_name="Reconstructed Building", mesh_show_back_face=True)


if __name__ == "__main__":
    main()