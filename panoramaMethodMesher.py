import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mcap.reader import make_reader
from rosbags.typesys import get_typestore, Stores
from scipy.interpolate import interp1d

# --- CONFIG ---
INPUT_FILENAME = "C:/Robot_Scan_Project/scan_imu_20260203_145509_0.mcap"
OUTPUT_FILENAME = "universal_room.ply"

# If the room is "Inside Out" (curving the wrong way), flip this to -1.0
SPIN_DIRECTION = 1.0


def main():
    print(f"Universal Spinner: {INPUT_FILENAME}")
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # --- 1. DETECT THE SPIN AXIS ---
    print("Analyzing Gyroscope to find the spin...")

    rot_x, rot_y, rot_z = 0.0, 0.0, 0.0
    timestamps = []
    # We store all 3 accumulations to plot/use later
    hist_x, hist_y, hist_z = [], [], []

    last_time = None

    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=["/imu/data"]):
            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/Imu")
            t = message.log_time / 1e9

            if last_time is None:
                last_time = t
                timestamps.append(t)
                hist_x.append(0);
                hist_y.append(0);
                hist_z.append(0)
                continue

            dt = t - last_time

            # Accumulate rotation on all 3 axes
            rot_x += msg.angular_velocity.x * dt
            rot_y += msg.angular_velocity.y * dt
            rot_z += msg.angular_velocity.z * dt

            hist_x.append(rot_x)
            hist_y.append(rot_y)
            hist_z.append(rot_z)
            timestamps.append(t)
            last_time = t

    # FIND THE WINNER
    mag_x = abs(rot_x)
    mag_y = abs(rot_y)
    mag_z = abs(rot_z)

    print(f"\n--- GYRO STATS (Total Rotation) ---")
    print(f"Axis X: {np.degrees(rot_x):.1f} deg")
    print(f"Axis Y: {np.degrees(rot_y):.1f} deg")
    print(f"Axis Z: {np.degrees(rot_z):.1f} deg")

    # Select the axis with maximum rotation
    best_axis = np.argmax([mag_x, mag_y, mag_z])
    axis_names = ['X', 'Y', 'Z']
    print(f"WINNER: Rotation detected on {axis_names[best_axis]} Axis!")

    if best_axis == 0:
        chosen_yaw = np.array(hist_x)
    elif best_axis == 1:
        chosen_yaw = np.array(hist_y)
    else:
        chosen_yaw = np.array(hist_z)

    # Apply direction fix
    chosen_yaw *= SPIN_DIRECTION

    # Interpolator
    yaw_interp = interp1d(timestamps, chosen_yaw, kind='linear', fill_value="extrapolate")

    # --- 2. BUILD ROOM USING DOMINANT AXIS ---
    print(f"Unrolling room using {axis_names[best_axis]} rotation...")

    points = []
    colors = []  # Color by time (Blue->Red) to trace the spin

    # PRE-CALC COLOR MAP
    time_start = timestamps[0]
    time_duration = timestamps[-1] - timestamps[0]

    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        count = 0
        for schema, channel, message in reader.iter_messages(topics=["/scan"]):
            # Downsample for speed
            if count % 10 != 0:
                count += 1
                continue

            t_scan = message.log_time / 1e9

            # Get the rotation from our WINNING axis
            scan_angle = yaw_interp(t_scan)

            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/LaserScan")
            ranges = np.array(msg.ranges)
            angles = np.arange(msg.angle_min, msg.angle_min + len(ranges) * msg.angle_increment, msg.angle_increment)[
                :len(ranges)]

            valid = (ranges > 0.5) & (ranges < 10.0)
            r = ranges[valid][::5]  # Decimate points
            theta = angles[valid][::5]

            # CONVERT TO 3D
            # If the spin is on a weird axis, the "Up" vector might be different too.
            # We assume "Standard Vertical" mounting relative to the rotation.
            # (i.e. if you spin around Y, the laser line is perpendicular to Y)

            # Laser Profile
            local_h = r * np.sin(theta)  # Height along the slice
            local_d = r * np.cos(theta)  # Distance from center

            # Rotate this profile around the Vertical (Z)
            # We treat the "Winning Axis" as the vertical axis for the reconstruction
            world_x = local_d * np.cos(scan_angle)
            world_y = local_d * np.sin(scan_angle)
            world_z = local_h

            # Stack
            pts = np.stack([world_x, world_y, world_z], axis=1)
            points.append(pts)

            # Color
            progress = (t_scan - time_start) / time_duration
            c = plt.get_cmap("jet")(progress)[:3]
            colors.append(np.tile(c, (len(pts), 1)))

            count += 1
            if count % 500 == 0: print(f"Processing {count}...", end="\r")

    # --- 3. VISUALIZE ---
    print("\nMerging...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors))

    print("DONE. Opening Visualizer.")
    print("If the room is INSIDE OUT, change SPIN_DIRECTION to -1.0")
    print(f"Detected Spin Axis: {axis_names[best_axis]}")

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()