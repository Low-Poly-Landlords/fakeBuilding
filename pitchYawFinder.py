import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mcap.reader import make_reader
from rosbags.typesys import get_typestore, Stores
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import time

# --- CONFIG ---
INPUT_FILENAME = "C:/Robot_Scan_Project/scan_imu_20260203_145509_0.mcap"
OUTPUT_FILENAME = "extreme_optimized_room.ply"

# --- THE EXTREME SEARCH RANGES ---
# Pitch: Test every angle from looking straight down to straight up
PITCH_RANGE = range(-90, 95, 5)

# Yaw: Test major spin speed errors (from 80% to 120% speed)
YAW_RANGE = np.arange(0.80, 1.20, 0.02)

# Offset: Test if the sensor moved in a circle (radius adjustment)
OFFSET_RANGE = np.arange(-1.0, 1.5, 0.5)


def main():
    start_time = time.time()
    print(f"Extreme Optimization: {INPUT_FILENAME}")
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # --- 1. DATA LOADING (Fast Mode) ---
    print("Loading Sensor Data...")

    # IMU Reading
    rot_x, rot_y, rot_z = [], [], []
    timestamps = []
    ix, iy, iz = 0.0, 0.0, 0.0
    last_time = None

    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=["/imu/data"]):
            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/Imu")
            t = message.log_time / 1e9

            if last_time is None:
                last_time = t;
                timestamps.append(t);
                rot_x.append(0);
                rot_y.append(0);
                rot_z.append(0);
                continue

            dt = t - last_time
            ix += msg.angular_velocity.x * dt
            iy += msg.angular_velocity.y * dt
            iz += msg.angular_velocity.z * dt

            timestamps.append(t);
            rot_x.append(ix);
            rot_y.append(iy);
            rot_z.append(iz);
            last_time = t

    # Axis Detection
    total_x, total_y, total_z = abs(ix), abs(iy), abs(iz)
    best_idx = np.argmax([total_x, total_y, total_z])
    raw_yaws = [rot_x, rot_y, rot_z][best_idx]
    print(f"Detected Spin Axis: {['X', 'Y', 'Z'][best_idx]}")

    # Laser Reading (Downsampled for Search Speed)
    laser_data = []
    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        count = 0
        for schema, channel, message in reader.iter_messages(topics=["/scan"]):
            if count % 20 != 0: count += 1; continue  # Skip 95% of frames for search
            t_scan = message.log_time / 1e9
            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/LaserScan")
            ranges = np.array(msg.ranges)
            angles = np.arange(msg.angle_min, msg.angle_min + len(ranges) * msg.angle_increment, msg.angle_increment)[
                :len(ranges)]
            valid = (ranges > 0.5) & (ranges < 10.0)

            # Very aggressive downsampling for the grid search (speed is key)
            r = ranges[valid][::15]
            theta = angles[valid][::15]
            t_array = np.full(len(r), t_scan)
            laser_data.append(np.column_stack((t_array, r, theta)))
            count += 1

    all_data = np.vstack(laser_data)
    base_yaw_interp = interp1d(timestamps, raw_yaws, kind='linear', fill_value="extrapolate")
    point_base_yaws = base_yaw_interp(all_data[:, 0])

    print(f"Data Loaded. Using {len(all_data)} points for optimization.")

    # --- 2. SCORING FUNCTION ---
    def get_score(yaw_mult, pitch_offset, center_offset):
        # 1. Yaw Transform
        yaws = point_base_yaws * yaw_mult

        # 2. Geometry Transform
        r = all_data[:, 1]
        theta = all_data[:, 2]
        pitch_rad = np.radians(pitch_offset)

        orig_y = r * np.cos(theta)
        orig_z = r * np.sin(theta)

        local_d = orig_y * np.cos(pitch_rad) - orig_z * np.sin(pitch_rad)
        local_d += center_offset  # Apply Radius Fix

        # 3. Projection
        world_x = local_d * np.cos(yaws)
        world_y = local_d * np.sin(yaws)

        # 4. Density Check (Histogram)
        # 60x60 grid covering 20m x 20m area
        hist, _, _ = np.histogram2d(world_x, world_y, bins=60, range=[[-10, 10], [-10, 10]])

        # Score = Sum of Squares (Higher = Sharper Walls)
        return np.sum(hist ** 2)

    # --- 3. EXTREME GRID SEARCH ---
    total_combinations = len(PITCH_RANGE) * len(YAW_RANGE) * len(OFFSET_RANGE)
    print(f"\n--- STARTING EXTREME SEARCH ---")
    print(f"Testing {total_combinations} combinations...")

    best_score = -1
    best_params = (1.0, 0.0, 0.0)  # Yaw, Pitch, Offset
    checked = 0

    # Iterate Pitch (Most likely culprit)
    for pitch in PITCH_RANGE:
        # Iterate Offset (Next likely)
        for offset in OFFSET_RANGE:
            # Iterate Yaw
            for yaw in YAW_RANGE:
                score = get_score(yaw, pitch, offset)

                if score > best_score:
                    best_score = score
                    best_params = (yaw, pitch, offset)
                    print(f" [New Best] Score: {int(score)} | Pitch: {pitch}° | Yaw: {yaw:.2f} | Offset: {offset:.1f}m")

                checked += 1
                if checked % 1000 == 0:
                    print(f"   Checked {checked}/{total_combinations}...", end="\r")

    print(f"\n\nGRID WINNER: Pitch={best_params[1]}, Yaw={best_params[0]:.2f}, Offset={best_params[2]}")

    # --- 4. FINE TUNING (The Polishing Step) ---
    print("Fine-tuning with mathematical solver...")

    def optimizer_func(p):
        return -get_score(p[0], p[1], p[2])

    # Start form the Grid Winner
    res = minimize(
        optimizer_func,
        best_params,
        method='Nelder-Mead',
        bounds=[(0.5, 1.5), (-180, 180), (-5, 5)],
        options={'maxiter': 100}
    )

    final_yaw, final_pitch, final_offset = res.x
    print(f"FINAL RESULT: Pitch={final_pitch:.2f}°, Yaw={final_yaw:.4f}, Offset={final_offset:.3f}m")

    # --- 5. HIGH RES RENDER ---
    print("\nGenerating Final High-Res Mesh...")

    # Reload FULL Data (Better Resolution)
    laser_data_full = []
    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        count = 0
        for schema, channel, message in reader.iter_messages(topics=["/scan"]):
            if count % 2 != 0: count += 1; continue  # Keep 50% of points for mesh
            t_scan = message.log_time / 1e9
            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/LaserScan")
            ranges = np.array(msg.ranges)
            angles = np.arange(msg.angle_min, msg.angle_min + len(ranges) * msg.angle_increment, msg.angle_increment)[
                :len(ranges)]
            valid = (ranges > 0.5) & (ranges < 10.0)

            r = ranges[valid]
            theta = angles[valid]
            t_array = np.full(len(r), t_scan)
            laser_data_full.append(np.column_stack((t_array, r, theta)))
            count += 1

    all_data_f = np.vstack(laser_data_full)
    point_base_yaws_f = base_yaw_interp(all_data_f[:, 0])

    # Apply Best Parameters
    yaws = point_base_yaws_f * final_yaw
    r = all_data_f[:, 1]
    theta = all_data_f[:, 2]
    pitch_rad = np.radians(final_pitch)

    orig_y = r * np.cos(theta)
    orig_z = r * np.sin(theta)

    local_d = orig_y * np.cos(pitch_rad) - orig_z * np.sin(pitch_rad)
    local_h = orig_y * np.sin(pitch_rad) + orig_z * np.cos(pitch_rad)
    local_d += final_offset  # Apply Offset

    world_x = local_d * np.cos(yaws)
    world_y = local_d * np.sin(yaws)
    world_z = local_h

    pts = np.column_stack((world_x, world_y, world_z))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Color by Z
    colors = plt.get_cmap("turbo")((world_z - world_z.min()) / (world_z.max() - world_z.min() + 1e-6))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.estimate_normals()

    o3d.io.write_point_cloud(OUTPUT_FILENAME, pcd)
    print(f"Saved to {OUTPUT_FILENAME}")
    print(f"Time Elapsed: {time.time() - start_time:.1f} seconds")

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()