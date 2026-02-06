import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mcap.reader import make_reader
from rosbags.typesys import get_typestore, Stores
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# --- CONFIG ---
INPUT_FILENAME = "C:/Robot_Scan_Project/scan_imu_20260203_145509_0.mcap"
OUTPUT_FILENAME = "arm_solved_room.ply"


def main():
    print(f"Solving for Arm Distance: {INPUT_FILENAME}")
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # --- 1. LOAD & SYNC DATA ---
    print("Loading and syncing IMU + Lidar...")

    # Load IMU
    timestamps_imu = []
    yaws_imu = []
    current_yaw = 0.0
    last_time = None

    # We load all axes to auto-detect the spin axis
    rot_accum = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    history = {'x': [], 'y': [], 'z': []}

    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=["/imu/data"]):
            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/Imu")
            t = message.log_time / 1e9

            if last_time is None:
                last_time = t
                timestamps_imu.append(t)
                for k in rot_accum: history[k].append(0.0)
                continue

            dt = t - last_time
            rot_accum['x'] += msg.angular_velocity.x * dt
            rot_accum['y'] += msg.angular_velocity.y * dt
            rot_accum['z'] += msg.angular_velocity.z * dt

            timestamps_imu.append(t)
            for k in rot_accum: history[k].append(rot_accum[k])
            last_time = t

    # Detect Axis
    ranges = {k: max(v) - min(v) for k, v in history.items()}
    best_axis = max(ranges, key=ranges.get)
    print(f"Detected Spin Axis: {best_axis.upper()}")
    raw_yaws = np.array(history[best_axis])

    # Interpolator: Get precise Yaw for any time T
    yaw_interpolator = interp1d(timestamps_imu, raw_yaws, kind='linear', fill_value="extrapolate")

    # Load Lidar
    lidar_points = []  # List of (Time, Range, Theta)

    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        count = 0
        for schema, channel, message in reader.iter_messages(topics=["/scan"]):
            # Downsample for speed (optimization needs to be fast)
            if count % 10 != 0: count += 1; continue

            t = message.log_time / 1e9
            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/LaserScan")

            ranges = np.array(msg.ranges)
            angles = np.arange(msg.angle_min, msg.angle_min + len(ranges) * msg.angle_increment, msg.angle_increment)[
                :len(ranges)]
            valid = (ranges > 0.5) & (ranges < 10.0)

            r = ranges[valid][::5]
            th = angles[valid][::5]

            # Save raw data
            # We don't transform yet. We store the ingredients.
            t_col = np.full(len(r), t)
            lidar_points.append(np.column_stack((t_col, r, th)))
            count += 1

    all_data = np.vstack(lidar_points)

    # Pre-calculate Yaws for all laser points
    point_yaws = yaw_interpolator(all_data[:, 0])
    point_ranges = all_data[:, 1]
    point_thetas = all_data[:, 2]

    print(f"Loaded {len(point_ranges)} points. Ready to optimize.")

    # --- 2. THE OPTIMIZER ---

    # This function builds the 3D cloud given a specific Arm Radius
    def build_cloud(arm_radius):
        # 1. Calculate Horizontal Distance from Sensor
        # Assuming vertical line scan (common for side-mounted lidars)
        # horizontal_dist = r * cos(theta)
        # vertical_dist   = r * sin(theta)

        # NOTE: If your lidar is flat, swap cos/sin.
        # Based on your previous "bowtie" images, this mapping seems correct for vertical.
        d_sensor = point_ranges * np.cos(point_thetas)
        z_world = point_ranges * np.sin(point_thetas)

        # 2. Add Arm Radius
        # Total distance from Body Pivot
        d_total = d_sensor + arm_radius

        # 3. Apply Yaw Rotation
        x_world = d_total * np.cos(point_yaws)
        y_world = d_total * np.sin(point_yaws)

        return np.column_stack((x_world, y_world, z_world))

    # The Cost Function: "How blurry is the room?"
    # We take a random subset of points and find their nearest neighbors.
    # If the Arm Radius is correct, the "Left Swing" points will be very close to "Right Swing" points.
    def cost_function(arm_radius):
        pts = build_cloud(arm_radius)

        # Sample 1000 random points to test
        sample_indices = np.random.choice(len(pts), 1000, replace=False)
        sample_pts = pts[sample_indices]

        # Create a KDTree of the FULL cloud
        # (This is fast enough for 100k points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        total_dist = 0
        for p in sample_pts:
            # Find 2 nearest neighbors (1 is itself, so we check the 2nd)
            _, idx, dist_sq = kdtree.search_knn_vector_3d(p, 2)
            # Add distance to closest neighbor
            total_dist += np.sqrt(dist_sq[1])

        avg_dist = total_dist / 1000.0
        print(f"Testing Radius: {arm_radius:.3f}m -> Avg Neighbor Dist: {avg_dist:.4f}")
        return avg_dist

    print("\n--- STARTING OPTIMIZATION ---")
    print("Searching for the radius that makes the walls sharpest...")

    # Search between -0.5m and +1.5m
    res = minimize_scalar(cost_function, bounds=(-0.5, 1.5), method='bounded')

    best_radius = res.x
    print("\n----------------------------------")
    print(f"OPTIMAL ARM RADIUS FOUND: {best_radius:.4f} meters")
    print("----------------------------------")

    # --- 3. SAVE FINAL MESH ---
    final_pts = build_cloud(best_radius)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pts)

    # Color by Z for visibility
    z = final_pts[:, 2]
    colors = plt.get_cmap("turbo")((z - z.min()) / (z.max() - z.min()))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.estimate_normals()
    o3d.io.write_point_cloud(OUTPUT_FILENAME, pcd)
    print(f"Saved optimized cloud to {OUTPUT_FILENAME}")

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()