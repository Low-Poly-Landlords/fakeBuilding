import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mcap.reader import make_reader
from rosbags.typesys import get_typestore, Stores
from scipy.interpolate import interp1d

# --- CONFIG ---
INPUT_FILENAME = "C:/Robot_Scan_Project/scan_imu_20260203_145509_0.mcap"
OUTPUT_FILENAME = "extruded_room.ply"
YAW_MULTIPLIER = 1.0

# CRITICAL TUNING
# How tall is the room in the image?
# If the room looks "squashed" flat, INCREASE this (e.g., 3.0, 5.0).
# If the room looks stretched like spaghetti, DECREASE this (e.g., 1.0).
HEIGHT_SCALE = 3.0


def main():
    print(f"Extruding Panorama to 3D: {INPUT_FILENAME}")
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # 1. READ IMU (Yaw Only)
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
            if last_time is None: last_time = t; imu_timestamps.append(t); imu_yaws.append(0.0); continue
            current_yaw += msg.angular_velocity.z * (t - last_time)
            imu_timestamps.append(t);
            imu_yaws.append(current_yaw);
            last_time = t

    yaw_interp = interp1d(imu_timestamps, np.array(imu_yaws) * YAW_MULTIPLIER, kind='linear', fill_value="extrapolate")

    # 2. BUILD THE GRID (Exactly like the Good Image)
    print("Building Grid...")
    DEG_PER_PIXEL = 0.2
    HEIGHT_ROWS = 400

    min_yaw = np.min(imu_yaws) * YAW_MULTIPLIER
    max_yaw = np.max(imu_yaws) * YAW_MULTIPLIER
    width_pixels = int(np.degrees(max_yaw - min_yaw) / DEG_PER_PIXEL) + 10

    # We store the DEPTH (Radius) in this grid
    grid_depth = np.full((width_pixels, HEIGHT_ROWS), np.nan)

    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        count = 0
        min_angle, max_angle = 0, 0

        for schema, channel, message in reader.iter_messages(topics=["/scan"]):
            t_scan = message.log_time / 1e9
            scan_yaw = yaw_interp(t_scan)
            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/LaserScan")

            if count == 0: min_angle, max_angle = msg.angle_min, msg.angle_max

            x_idx = int(np.degrees(scan_yaw - min_yaw) / DEG_PER_PIXEL)
            if x_idx < 0 or x_idx >= width_pixels: continue

            ranges = np.array(msg.ranges)
            angles = np.arange(msg.angle_min, msg.angle_min + len(ranges) * msg.angle_increment, msg.angle_increment)[
                :len(ranges)]

            valid = (ranges > 0.5) & (ranges < 15.0)
            r = ranges[valid]
            theta = angles[valid]

            # Map Vertical Angle directly to Row Index (0 to HEIGHT_ROWS)
            # This forces the "Columns" to stay vertical
            y_indices = ((theta - min_angle) / (max_angle - min_angle) * (HEIGHT_ROWS - 1)).astype(int)

            # Z-Buffer fill
            valid_y = (y_indices >= 0) & (y_indices < HEIGHT_ROWS)
            # We want the MINIMUM depth to hit the wall, not the background
            # Note: Logic slightly inverted from image visualization, but safe for mesh
            current_vals = grid_depth[x_idx, y_indices[valid_y]]

            # If NaN, take new value. If not NaN, take min.
            new_vals = r[valid_y]
            update_mask = np.isnan(current_vals) | (new_vals < current_vals)

            # Update grid
            # This is tricky with numpy indexing, doing simple loop for safety/clarity
            for y, val in zip(y_indices[valid_y], new_vals):
                current = grid_depth[x_idx, y]
                if np.isnan(current) or val < current:
                    grid_depth[x_idx, y] = val

            count += 1
            if count % 1000 == 0: print(f"Processing {count}...", end="\r")

    # 3. EXTRUDE TO 3D (The "Dumb" Transformation)
    print("\nExtruding...")
    vertices = []
    triangles = []
    index_grid = np.full((width_pixels, HEIGHT_ROWS), -1, dtype=int)
    current_vert_idx = 0

    for x in range(width_pixels):
        for y in range(HEIGHT_ROWS):
            depth = grid_depth[x, y]
            if np.isnan(depth): continue

            # ANGLE (Yaw) comes from X
            yaw = np.radians(x * DEG_PER_PIXEL) + min_yaw

            # HEIGHT (Z) comes strictly from Y (Row Index)
            # We map 0..HEIGHT_ROWS to -Scale..+Scale
            # This creates perfect horizontal layers
            world_z = (y / HEIGHT_ROWS) * HEIGHT_SCALE

            # RADIUS comes from Depth
            # world_x = depth * cos(yaw)
            # world_y = depth * sin(yaw)
            world_x = depth * np.cos(yaw)
            world_y = depth * np.sin(yaw)

            vertices.append([world_x, world_y, world_z])
            index_grid[x, y] = current_vert_idx
            current_vert_idx += 1

    # Stitching
    for x in range(width_pixels - 1):
        for y in range(HEIGHT_ROWS - 1):
            idx_tl = index_grid[x, y];
            idx_bl = index_grid[x, y + 1]
            idx_tr = index_grid[x + 1, y];
            idx_br = index_grid[x + 1, y + 1]

            if idx_tl == -1 or idx_bl == -1 or idx_tr == -1 or idx_br == -1: continue

            # Check for gaps (don't stitch foreground to background)
            d1 = grid_depth[x, y]
            d2 = grid_depth[x + 1, y + 1]
            if abs(d1 - d2) < 0.3:
                triangles.append([idx_tl, idx_bl, idx_tr])
                triangles.append([idx_tr, idx_bl, idx_br])

    # 4. SAVE
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    mesh.compute_vertex_normals()

    # Color by Z (Blue Floor, Red Ceiling)
    pts = np.asarray(mesh.vertices)
    z = pts[:, 2]
    z_norm = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-6)
    colors = plt.get_cmap("turbo")(z_norm)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_triangle_mesh(OUTPUT_FILENAME, mesh)
    print(f"\nSaved to {OUTPUT_FILENAME}")

    print("If the room looks 'Squashed', increase HEIGHT_SCALE in the script.")
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


if __name__ == "__main__":
    main()