import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mcap.reader import make_reader
from rosbags.typesys import get_typestore, Stores
from scipy.spatial.transform import Rotation as R

INPUT_FILENAME = "hotel_scan_2d_wobble.mcap"
OUTPUT_FILENAME = "hotel_reconstruction_2d.ply"

# --- CONFIG ---
# We use a slightly finer voxel size because 2D scans can be very dense in stripes
VOXEL_SIZE = 0.04
POISSON_DEPTH = 9


def main():
    print(f"Reconstructing from 2D Vertical Scan: {INPUT_FILENAME}...")
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    global_pcd = o3d.geometry.PointCloud()
    pose_map = {}

    # --- STEP 1: LOAD POSES ---
    print("Reading TF data...")
    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=["/tf"]):
            msg = typestore.deserialize_cdr(message.data, "tf2_msgs/msg/TFMessage")
            for t in msg.transforms:
                trans = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
                quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]

                rot_mat = R.from_quat(quat).as_matrix()
                pose_mat = np.identity(4)
                pose_mat[:3, :3] = rot_mat
                pose_mat[:3, 3] = trans
                pose_map[message.log_time] = pose_mat

    print(f"Loaded {len(pose_map)} pose transforms.")

    # --- STEP 2: STITCH STRIPS ---
    print("Stitching vertical scan strips...")

    pts_buffer = []

    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        count = 0

        for schema, channel, message in reader.iter_messages(topics=["/lidar/points"]):
            # 1. Sync Time
            if not pose_map: break
            # Find closest pose (Simple nearest neighbor)
            # Since 2D scan is high freq (40hz), nearest is usually very close
            closest_ts = min(pose_map.keys(), key=lambda k: abs(k - message.log_time))

            # 2. Get Pose Matrix
            pose = pose_map[closest_ts]

            # 3. Decode Points
            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/PointCloud2")
            raw_data = np.frombuffer(msg.data, dtype=np.uint8)
            # The data is just float32 x,y,z
            xyz = raw_data.view(np.float32).reshape((-1, 3))

            # 4. Transform Manually (Faster than creating O3D objects per line)
            # Apply Rotation
            # P_global = R * P_local + T
            # pose[:3, :3] is Rotation, pose[:3, 3] is Translation

            # Transpose xyz for matrix math: [3, N]
            xyz_T = xyz.T
            rotated = np.dot(pose[:3, :3], xyz_T)
            translated = rotated + pose[:3, 3:4]  # Add translation vector

            # Append to huge buffer
            pts_buffer.append(translated.T)

            count += 1
            if count % 1000 == 0:
                print(f"Stitched {count} strips...", end="\r")

    # Stack all points into one massive array
    all_points = np.vstack(pts_buffer)
    print(f"\nTotal Points: {len(all_points)}")

    # Move to Open3D
    global_pcd.points = o3d.utility.Vector3dVector(all_points)

    # --- STEP 3: PROCESS & MESH ---
    print("Downsampling (merging stripes)...")
    global_pcd = global_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"Reduced to {len(global_pcd.points)} unique points.")

    print("Estimating Normals...")
    global_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 3, max_nn=40))
    global_pcd.orient_normals_consistent_tangent_plane(k=20)

    print("Running Surface Reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(global_pcd, depth=POISSON_DEPTH)

    # Cleanup
    print("Cleaning artifacts...")
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Color
    vertices = np.asarray(mesh.vertices)
    z_values = vertices[:, 2]
    z_norm = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 1e-6)
    colormap = plt.get_cmap("turbo")  # "Turbo" is great for distinct layers
    colors = colormap(z_norm)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()

    # --- SAVE ---
    o3d.io.write_triangle_mesh(OUTPUT_FILENAME, mesh)
    print(f"Saved reconstruction to {OUTPUT_FILENAME}")

    o3d.visualization.draw_geometries([mesh], window_name="2D Scan Reconstruction", mesh_show_back_face=True)


if __name__ == "__main__":
    main()