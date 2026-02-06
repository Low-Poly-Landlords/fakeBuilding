import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mcap.reader import make_reader
from rosbags.typesys import get_typestore, Stores
from scipy.interpolate import interp1d

# --- CONFIG ---
INPUT_FILENAME = "C:/Robot_Scan_Project/scan_imu_20260203_145509_0.mcap"
OUTPUT_FILENAME = "quilted_room.ply"

# TUNING: Adjust this if the walls look "broken" or overlapping.
# 1.0 = Trust IMU exactly.
# 1.1 = Spin more. 0.9 = Spin less.
YAW_MULTIPLIER = 1.0

# Distance threshold: Don't connect points if they are too far apart (e.g. background to foreground jump)
MESH_MAX_EDGE_LEN = 0.5


def main():
    print(f"Quilt Meshing: {INPUT_FILENAME}")
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # --- STEP 1: IMU ROTATION ---
    print("Reading IMU...")
    imu_timestamps = []
    imu_yaws = []
    current_yaw = 0.0
    last_time = None

    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=["/imu/data"]):
            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/Imu")
            t = message.log_time / 1e9
            if last_time is None:
                last_time = t
                imu_timestamps.append(t)
                imu_yaws.append(0.0)
                continue

            dt = t - last_time
            # Use Z-axis (Yaw)
            current_yaw += (msg.angular_velocity.z) * dt

            imu_timestamps.append(t)
            imu_yaws.append(current_yaw)
            last_time = t

    # Apply the Multiplier Correction
    imu_yaws = np.array(imu_yaws) * YAW_MULTIPLIER
    yaw_interpolator = interp1d(imu_timestamps, imu_yaws, kind='linear', fill_value="extrapolate")

    # --- STEP 2: LOAD & STRUCTURE SCANS ---
    print("Reading Scans & Building Grid...")

    # We need a list of columns, where each column is the XYZ points of one laser scan
    scan_columns = []

    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        count = 0

        for schema, channel, message in reader.iter_messages(topics=["/scan"]):
            t_scan = message.log_time / 1e9
            scan_yaw = yaw_interpolator(t_scan)

            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/LaserScan")

            # Reconstruct angles
            angles = np.arange(msg.angle_min, msg.angle_min + len(msg.ranges) * msg.angle_increment,
                               msg.angle_increment)
            if len(angles) > len(msg.ranges): angles = angles[:len(msg.ranges)]
            ranges = np.array(msg.ranges)

            # We keep ALL points (even invalid ones) initially to preserve the "Grid" structure
            # Invalid points will be marked with NaN

            # Vertical Logic
            local_x = ranges * np.cos(angles)  # Forward
            local_z = ranges * np.sin(angles)  # Up

            # Apply Rotation
            world_x = local_x * np.cos(scan_yaw)
            world_y = local_x * np.sin(scan_yaw)
            world_z = local_z

            # Mask out invalid ranges (too close/far)
            invalid_mask = (ranges < 0.2) | (ranges > 10.0)
            world_x[invalid_mask] = np.nan
            world_y[invalid_mask] = np.nan
            world_z[invalid_mask] = np.nan

            # Store column as (N, 3) array
            col = np.stack([world_x, world_y, world_z], axis=1)
            scan_columns.append(col)

            count += 1
            if count % 200 == 0: print(f"Processed {count} scans...", end="\r")

    if not scan_columns:
        print("No scans found.")
        return

    # --- STEP 3: STITCH TRIANGLES (The "Quilt") ---
    print(f"\nStitching {len(scan_columns)} scan strips...")

    vertices = []
    triangles = []

    # Flatten vertices for Open3D but keep track of indices
    # We need to map (Col_Index, Row_Index) -> Vertex_Index

    # To save memory/time, we will build the mesh directly.
    # Since all scans have the same number of points (N), the grid is regular.

    rows = len(scan_columns[0])
    cols = len(scan_columns)

    # Convert list of arrays to one big (Cols, Rows, 3) array
    # Careful: scans might have different lengths due to dropouts?
    # Usually LaserScan is fixed size. Let's assume fixed size.
    # If sizes differ, we truncate to minimum.
    min_rows = min(len(c) for c in scan_columns)
    grid = np.array([c[:min_rows] for c in scan_columns])  # Shape: (Cols, Rows, 3)

    # Reshape to (Num_Verts, 3)
    all_verts = grid.reshape(-1, 3)

    # Generate faces
    # For every square in the grid:
    # (r, c) ----- (r, c+1)
    #   |             |
    # (r+1,c) ---- (r+1,c+1)

    print("Generating faces...")
    for c in range(cols - 1):
        for r in range(min_rows - 1):
            # Indices in the flat array
            idx_tl = c * min_rows + r  # Top-Left
            idx_bl = c * min_rows + (r + 1)  # Bottom-Left
            idx_tr = (c + 1) * min_rows + r  # Top-Right
            idx_br = (c + 1) * min_rows + (r + 1)  # Bottom-Right

            # Get actual coordinates to check validity
            v_tl = all_verts[idx_tl]
            v_tr = all_verts[idx_tr]
            v_bl = all_verts[idx_bl]
            v_br = all_verts[idx_br]

            # Check for NaNs (Invalid points)
            if np.isnan(v_tl).any() or np.isnan(v_tr).any() or np.isnan(v_bl).any():
                pass  # Skip Triangle 1
            else:
                # Check edge lengths (don't connect foreground to background)
                d1 = np.linalg.norm(v_tl - v_tr)
                d2 = np.linalg.norm(v_tl - v_bl)
                d3 = np.linalg.norm(v_tr - v_bl)
                if d1 < MESH_MAX_EDGE_LEN and d2 < MESH_MAX_EDGE_LEN and d3 < MESH_MAX_EDGE_LEN:
                    triangles.append([idx_tl, idx_bl, idx_tr])  # Triangle 1

            if np.isnan(v_tr).any() or np.isnan(v_bl).any() or np.isnan(v_br).any():
                pass  # Skip Triangle 2
            else:
                d1 = np.linalg.norm(v_tr - v_br)
                d2 = np.linalg.norm(v_bl - v_br)
                d3 = np.linalg.norm(v_tr - v_bl)
                if d1 < MESH_MAX_EDGE_LEN and d2 < MESH_MAX_EDGE_LEN and d3 < MESH_MAX_EDGE_LEN:
                    triangles.append([idx_tr, idx_bl, idx_br])  # Triangle 2

        if c % 50 == 0: print(f"Meshed column {c}/{cols}...", end="\r")

    # --- CLEANUP & SAVE ---
    print("\nFinalizing Mesh...")
    mesh = o3d.geometry.TriangleMesh()

    # Open3D doesn't like NaNs in vertices, even if unused.
    # We filter them out now.

    valid_mask = ~np.isnan(all_verts).any(axis=1)

    # We need to remap triangle indices because we are deleting vertices
    # Old_Index -> New_Index map
    new_indices = np.cumsum(valid_mask) - 1

    # Filter vertices
    clean_verts = all_verts[valid_mask]
    mesh.vertices = o3d.utility.Vector3dVector(clean_verts)

    # Filter triangles that point to invalid vertices (should be none, but safe to check)
    clean_triangles = []
    for t in triangles:
        if valid_mask[t[0]] and valid_mask[t[1]] and valid_mask[t[2]]:
            clean_triangles.append([new_indices[t[0]], new_indices[t[1]], new_indices[t[2]]])

    mesh.triangles = o3d.utility.Vector3dIntVector(np.array(clean_triangles))

    mesh.compute_vertex_normals()

    # Color by Height
    colors = plt.get_cmap("turbo")((clean_verts[:, 2] - np.min(clean_verts[:, 2])) / 2.5)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_triangle_mesh(OUTPUT_FILENAME, mesh)
    print(f"Saved mesh to {OUTPUT_FILENAME}")

    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


if __name__ == "__main__":
    main()