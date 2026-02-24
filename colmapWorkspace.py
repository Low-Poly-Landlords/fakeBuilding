import os
import cv2
import numpy as np
import bisect
from scipy.spatial.transform import Rotation as R
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# --- CONFIGURATION ---
INPUT_FILE = "newestScan.mcap"
WORKSPACE_DIR = "colmap_workspace"

# The Tuned Hardware Values
CAM_OFFSET = np.array([0.0, 0.0, 0.05])
CAM_ROLL = -17.0
CAM_PITCH = 0
CAM_YAW = 0.0

# Calculated Intrinsics for Pi HQ Camera + 6mm Lens @ 640x480
IMG_W = 640
IMG_H = 480
FX = 611.0
FY = 611.0
CX = 320.0
CY = 240.0

IMU_FIX = R.from_euler('x', 90, degrees=True)
BASE_ROBOT_TO_CAM = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
USER_CAM_FIX = R.from_euler('xyz', [CAM_ROLL, CAM_PITCH, CAM_YAW], degrees=True).as_matrix()
FINAL_ROBOT_TO_CAM = USER_CAM_FIX @ BASE_ROBOT_TO_CAM


def get_interpolated_pose(target_time, pose_data):
    times = [x[0] for x in pose_data]
    idx = bisect.bisect_left(times, target_time)
    if idx == 0: return pose_data[0][1]
    if idx == len(times): return pose_data[-1][1]
    before = times[idx - 1]
    after = times[idx]
    if after - target_time < target_time - before:
        return pose_data[idx][1]
    return pose_data[idx - 1][1]


def setup_workspace():
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)

    img_dir = os.path.join(WORKSPACE_DIR, "images")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    sparse_dir = os.path.join(WORKSPACE_DIR, "sparse")
    if not os.path.exists(sparse_dir):
        os.makedirs(sparse_dir)

    return img_dir, sparse_dir


def main():
    print(f"Reading {INPUT_FILE}...")
    reader = make_reader(open(INPUT_FILE, "rb"), decoder_factories=[DecoderFactory()])

    imu_data = []
    images = []

    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic == "/imu/data":
            q = [ros_msg.orientation.x, ros_msg.orientation.y, ros_msg.orientation.z, ros_msg.orientation.w]
            imu_data.append((message.log_time, q))
        elif channel.topic == "/camera/image_raw":
            try:
                np_arr = np.frombuffer(ros_msg.data, dtype=np.uint8)
                img = np_arr.reshape((IMG_H, IMG_W, 3))
                images.append((message.log_time, img))
            except:
                pass

    print(f"Found {len(images)} images.")

    img_dir, sparse_dir = setup_workspace()

    # 1. Write cameras.txt
    # Format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    cam_file = open(os.path.join(sparse_dir, "cameras.txt"), "w")
    cam_file.write(f"1 PINHOLE {IMG_W} {IMG_H} {FX} {FY} {CX} {CY}\n")
    cam_file.close()

    # 2. Write images.txt and save JPEGs
    img_file = open(os.path.join(sparse_dir, "images.txt"), "w")

    # We only need about 1 image every second or so to avoid blowing up the engine
    skip_rate = max(1, len(images) // 200)  # Aim for ~200 highly synced images

    image_id = 1
    print("Exporting images and calculating poses...")

    for i in range(0, len(images), skip_rate):
        t, img_bgr = images[i]

        # Blur check: skip blurry frames
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 100.0:
            continue

        # Get Pose
        raw_quat = get_interpolated_pose(t, imu_data)
        r_robot_to_world = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()

        # We need World -> Camera (COLMAP Standard)
        # 1. Camera to Robot
        r_cam_to_robot = FINAL_ROBOT_TO_CAM.T
        t_cam_to_robot = CAM_OFFSET

        # 2. Camera to World
        r_cam_to_world = r_robot_to_world @ r_cam_to_robot
        t_cam_to_world = r_robot_to_world @ t_cam_to_robot

        # 3. World to Camera (Inverse)
        r_world_to_cam = r_cam_to_world.T
        t_world_to_cam = -r_world_to_cam @ t_cam_to_world

        # Convert to Quaternion (W, X, Y, Z for COLMAP)
        q = R.from_matrix(r_world_to_cam).as_quat()
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]
        tx, ty, tz = t_world_to_cam

        # Save Image
        img_name = f"frame_{image_id:04d}.jpg"
        cv2.imwrite(os.path.join(img_dir, img_name), img_bgr)

        # Write to images.txt
        # Format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        img_file.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img_name}\n")
        img_file.write("\n")  # COLMAP needs an empty line for 2D points

        image_id += 1

    img_file.close()

    # 3. Create empty points3D.txt (Required by standard, but we provide our own mesh)
    open(os.path.join(sparse_dir, "points3D.txt"), "w").close()

    print(f"\nSuccess! Exported {image_id - 1} crisp images to '{WORKSPACE_DIR}'.")


if __name__ == "__main__":
    main()