import numpy as np
import matplotlib.pyplot as plt
from mcap.reader import make_reader
from rosbags.typesys import get_typestore, Stores
from scipy.interpolate import interp1d

# --- CONFIG ---
INPUT_FILENAME = "C:/Robot_Scan_Project/scan_imu_20260203_145509_0.mcap"

# Use the best guess you have so far (e.g., 1.0)
INITIAL_MULTIPLIER = 1.0


def main():
    print(f"Generating Panorama from: {INPUT_FILENAME}")
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # --- 1. GET RAW IMU DATA ---
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
            # Integrate Z-axis (Yaw)
            current_yaw += (msg.angular_velocity.z) * dt

            imu_timestamps.append(t)
            imu_yaws.append(current_yaw)
            last_time = t

    imu_yaws = np.array(imu_yaws)

    # --- 2. INTERACTIVE PLOT FUNCTION ---
    def generate_image(multiplier):
        # Create interpolator with new multiplier
        adjusted_yaws = imu_yaws * multiplier
        yaw_interp = interp1d(imu_timestamps, adjusted_yaws, kind='linear', fill_value="extrapolate")

        # We will bucket points into a 2D grid
        # Width = 360 degrees (or total rotation range)
        # Height = Meters

        # Estimate range to set image size
        total_rot = np.ptp(adjusted_yaws)
        min_rot = np.min(adjusted_yaws)

        # Resolution: 0.2 degrees per pixel horizontal, 2cm per pixel vertical
        width_pixels = int(np.degrees(total_rot) / 0.2) + 100
        if width_pixels > 5000: width_pixels = 5000  # Cap size
        height_pixels = 500  # 5 meters / 1cm

        image_depth = np.full((height_pixels, width_pixels), np.nan)

        print(f"Building image for Multiplier {multiplier}...")

        with open(INPUT_FILENAME, "rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages(topics=["/scan"]):
                t_scan = message.log_time / 1e9

                # Get Yaw for this column
                scan_yaw = yaw_interp(t_scan)

                # Map Yaw to Image X
                # (scan_yaw - min_rot) / (total_rot) * width
                x_idx = int(np.degrees(scan_yaw - min_rot) / 0.2)
                if x_idx < 0 or x_idx >= width_pixels: continue

                msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/LaserScan")
                ranges = np.array(msg.ranges)
                angles = np.arange(msg.angle_min, msg.angle_min + len(ranges) * msg.angle_increment,
                                   msg.angle_increment)[:len(ranges)]

                # Map Vertical Angle to Image Y
                # Assuming simple vertical mount:
                # Top of scan = High Y, Bottom = Low Y
                # scan angle usually goes -Pi/2 to +Pi/2 or similar
                # We map this 1-to-1 to pixels

                # Filter valid
                valid = (ranges > 0.2) & (ranges < 10.0)
                r = ranges[valid]
                theta = angles[valid]

                # Convert polar angle to pixel height
                # Assuming theta is roughly -1.5 to +1.5 radians
                # Map to 0..height_pixels
                y_indices = ((theta - msg.angle_min) / (msg.angle_max - msg.angle_min) * (height_pixels - 1)).astype(
                    int)

                # Set pixels (Depth)
                # We only keep the CLOSEST depth per pixel to handle overlap
                current_vals = image_depth[y_indices, x_idx]
                mask = np.isnan(current_vals) | (r < current_vals)
                image_depth[y_indices[mask], x_idx] = r[mask]

        return image_depth

    # --- 3. GENERATE INITIAL IMAGE ---
    img = generate_image(INITIAL_MULTIPLIER)

    plt.figure(figsize=(12, 6))
    plt.imshow(img, cmap='magma_r', aspect='auto')
    plt.colorbar(label='Depth (meters)')
    plt.title(f"Lidar Panorama (Multiplier: {INITIAL_MULTIPLIER})\nLook for straight vertical structures.")
    plt.xlabel("Rotation Angle (Yaw)")
    plt.ylabel("Vertical Scan Angle")
    plt.tight_layout()
    plt.show()

    print("\nDIAGNOSIS GUIDE:")
    print("1. Do you see clear, straight vertical lines? (Doorframes, Corners)")
    print("   -> If YES: Your multiplier is correct.")
    print("   -> If lines are 'Sawtooth' or 'Wavy': Your IMU sync is bad.")
    print("   -> If lines are 'Slanted' ( / or \\ ): You need to adjust the YAW_MULTIPLIER.")
    print("      - Slanted /// means increase multiplier.")
    print("      - Slanted \\\\\\ means decrease multiplier.")


if __name__ == "__main__":
    main()