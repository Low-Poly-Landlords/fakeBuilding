import torch
import numpy as np
import math
import json
import time
from mcap.writer import Writer
from rosbags.typesys import get_typestore, Stores
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
INPUT_LAYOUT = "hotel_layout.json"
OUTPUT_FILENAME = "hotel_scan_2d_wobble.mcap"

# 2D SENSOR SETTINGS
FOV_MIN, FOV_MAX = -60.0, 60.0  # Vertical Field of View (slices floor to ceiling)
NUM_RAYS = 1024  # High resolution vertical line
MAX_RANGE = 15.0  # Typical handheld lidar range

# MOTION SETTINGS
WALK_SPEED = 0.8  # Walk slower to ensure density
FPS = 40  # High FPS because 2D lidars spin fast (e.g. 40Hz)
SWEEP_SPEED = 2.0  # How fast they wave it side-to-side (oscillations per sec)
SWEEP_ANGLE = 45.0  # Degrees to sweep left/right

# GPU SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Simulating 2D Vertical Slice on: {device.type.upper()}")


# --- MATH HELPERS ---
def gpu_raycast(rays_o, rays_d, box_min, box_max):
    # Reshape for broadcast: Rays [N, 1, 3] vs Boxes [1, M, 3]
    ro, rd = rays_o.unsqueeze(1), rays_d.unsqueeze(1)
    bmin, bmax = box_min.unsqueeze(0), box_max.unsqueeze(0)

    # Intersection logic
    inv_d = 1.0 / (rd + 1e-6)
    t0 = (bmin - ro) * inv_d
    t1 = (bmax - ro) * inv_d

    t_min = torch.min(t0, t1)
    t_max = torch.max(t0, t1)

    t_enter = torch.max(t_min, dim=2).values
    t_exit = torch.min(t_max, dim=2).values

    hit_mask = (t_exit >= t_enter) & (t_exit > 0)

    distances = torch.where(hit_mask, t_enter, torch.tensor(float('inf'), device=device))
    closest_dists, _ = torch.min(distances, dim=1)

    return closest_dists


# --- PATH GENERATION (Same methodical route) ---
FLOOR_HEIGHT = 3.5
ROOM_WIDTH, ROOM_DEPTH = 5.0, 6.0
ROOMS_PER_SIDE = 4


def generate_waypoints():
    waypoints = []
    z = 1.5
    waypoints.extend([(0, -8, z), (0, 0, z), (5, 0, z), (8.5, 0, z), (0, 0, z), (-1.5, 0, z)])
    for f in range(1, 5):
        z = (f * FLOOR_HEIGHT) + 1.5
        waypoints.append((-1.5, 0, z))
        for r in range(ROOMS_PER_SIDE):
            x = r * ROOM_WIDTH + (ROOM_WIDTH / 2)
            waypoints.extend([(x, 0, z), (x, 3.5, z), (x, 0, z), (x, -3.5, z), (x, 0, z)])
        end_x = ROOMS_PER_SIDE * ROOM_WIDTH
        waypoints.extend([(end_x, 0, z), (end_x + 1.5, 0, z), (end_x, 0, z), (-1.5, 0, z)])
    return waypoints


def interpolate_path(waypoints):
    full_path = []
    total_time = 0.0

    for i in range(len(waypoints) - 1):
        p1 = np.array(waypoints[i])
        p2 = np.array(waypoints[i + 1])
        dist = np.linalg.norm(p2 - p1)

        # Teleport (Elevator)
        if dist > 5.0 and abs(p1[2] - p2[2]) > 1.0:
            continue

        num_steps = max(1, int((dist / WALK_SPEED) * FPS))
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        walk_yaw = math.atan2(dy, dx)

        for s in range(num_steps):
            t = s / num_steps
            pos = p1 + (p2 - p1) * t

            # THE WOBBLE LOGIC
            # We calculate a time-based sweep
            # sweep_yaw = sin(time) * 45 degrees
            current_time = total_time + (s / FPS)
            sweep_offset = math.sin(current_time * SWEEP_SPEED) * math.radians(SWEEP_ANGLE)

            final_yaw = walk_yaw + sweep_offset

            full_path.append((*pos, final_yaw))

        total_time += (num_steps / FPS)

    return full_path


def main():
    # 1. LOAD LAYOUT
    with open(INPUT_LAYOUT, 'r') as f:
        layout_data = json.load(f)

    b_mins, b_maxs = [], []
    for b in layout_data:
        b_mins.append([b['min_x'], b['min_y'], b['min_z']])
        b_maxs.append([b['max_x'], b['max_y'], b['max_z']])

    box_min_t = torch.tensor(b_mins, dtype=torch.float32, device=device)
    box_max_t = torch.tensor(b_maxs, dtype=torch.float32, device=device)

    path = interpolate_path(generate_waypoints())
    print(f"Path length: {len(path)} frames.")

    # 2. GENERATE 2D VERTICAL SLICE
    # A vertical line means Azimuth = 0, Elevation varies
    elevs = torch.linspace(math.radians(FOV_MIN), math.radians(FOV_MAX), NUM_RAYS, device=device)

    # In sensor frame (x=forward), a vertical slice is in the X-Z plane
    # x = cos(elev)
    # y = 0
    # z = sin(elev)

    base_x = torch.cos(elevs)
    base_y = torch.zeros_like(elevs)
    base_z = torch.sin(elevs)

    # [NUM_RAYS, 3]
    base_rays = torch.stack([base_x, base_y, base_z], dim=1)

    # 3. ROS SETUP
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    TFMessage = typestore.types['tf2_msgs/msg/TFMessage']
    TransformStamped = typestore.types['geometry_msgs/msg/TransformStamped']
    Header = typestore.types['std_msgs/msg/Header']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']
    QuaternionMsg = typestore.types['geometry_msgs/msg/Quaternion']
    PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
    PointField = typestore.types['sensor_msgs/msg/PointField']

    print(f"Starting 2D Wobble Simulation...")
    start_time = time.time()

    with open(OUTPUT_FILENAME, "wb") as f:
        writer = Writer(f)
        writer.start()

        # Schemas
        tf_schema = """geometry_msgs/TransformStamped[] transforms\n================================================================================\nMSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\ngeometry_msgs/Transform transform\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n================================================================================\nMSG: geometry_msgs/Transform\ngeometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation\n================================================================================\nMSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n================================================================================\nMSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w"""
        pc_schema = """std_msgs/Header header\nuint32 height\nuint32 width\nsensor_msgs/PointField[] fields\nbool is_bigendian\nuint32 point_step\nuint32 row_step\nuint8[] data\nbool is_dense\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n================================================================================\nMSG: sensor_msgs/PointField\nuint8 INT8=1\nuint8 UINT8=2\nuint8 INT16=3\nuint8 UINT16=4\nuint8 INT32=5\nuint8 UINT32=6\nuint8 FLOAT32=7\nuint8 FLOAT64=8\nstring name\nuint32 offset\nuint8 datatype\nuint32 count"""

        tf_sid = writer.register_schema(name="tf2_msgs/msg/TFMessage", encoding="ros2msg", data=tf_schema.encode())
        pc_sid = writer.register_schema(name="sensor_msgs/msg/PointCloud2", encoding="ros2msg", data=pc_schema.encode())
        tf_cid = writer.register_channel(topic="/tf", message_encoding="cdr", schema_id=tf_sid)
        pc_cid = writer.register_channel(topic="/lidar/points", message_encoding="cdr", schema_id=pc_sid)

        for i, (rx, ry, rz, yaw) in enumerate(path):
            ts_ns = 1000000000 + int(i * (1e9 / FPS))

            # --- A. ROTATE RAYS (Yaw Sweep) ---
            # We apply the Wobble Yaw here
            c, s = math.cos(yaw), math.sin(yaw)

            rot_x = base_rays[:, 0] * c - base_rays[:, 1] * s
            rot_y = base_rays[:, 0] * s + base_rays[:, 1] * c
            rot_z = base_rays[:, 2]  # No pitch/roll change, z stays z

            current_rays_d = torch.stack([rot_x, rot_y, rot_z], dim=1)
            current_rays_o = torch.tensor([rx, ry, rz], device=device).expand_as(current_rays_d)

            # --- B. GPU RAYCAST ---
            closest_dists = gpu_raycast(current_rays_o, current_rays_d, box_min_t, box_max_t)

            # --- C. FILTER ---
            valid_mask = closest_dists < MAX_RANGE
            valid_dists = closest_dists[valid_mask]

            # IMPORTANT: Reconstruct points in Sensor Frame (no rotation)
            # The PC2 message expects points relative to the sensor center.
            valid_base = base_rays[valid_mask]
            points_xyz = valid_base * valid_dists.unsqueeze(1)
            points_cpu = points_xyz.cpu().numpy().astype(np.float32)

            # --- D. WRITE ---
            # TF
            q = R.from_euler('z', yaw).as_quat()
            tf_msg = TFMessage(transforms=[TransformStamped(
                header=Header(stamp=Time(sec=ts_ns // 10 ** 9, nanosec=ts_ns % 10 ** 9), frame_id="map"),
                child_frame_id="base_link",
                transform=typestore.types['geometry_msgs/msg/Transform'](
                    translation=Vector3(x=rx, y=ry, z=rz - 1.5),
                    rotation=QuaternionMsg(x=q[0], y=q[1], z=q[2], w=q[3])
                )
            )])
            writer.add_message(tf_cid, ts_ns, typestore.serialize_cdr(tf_msg, "tf2_msgs/msg/TFMessage"), ts_ns)

            # PointCloud
            pc_msg = PointCloud2(
                header=Header(stamp=Time(sec=ts_ns // 10 ** 9, nanosec=ts_ns % 10 ** 9), frame_id="base_link"),
                height=1, width=len(points_cpu),
                fields=[
                    PointField(name="x", offset=0, datatype=7, count=1),
                    PointField(name="y", offset=4, datatype=7, count=1),
                    PointField(name="z", offset=8, datatype=7, count=1)
                ],
                is_bigendian=False, point_step=12, row_step=12 * len(points_cpu),
                data=points_cpu.view(np.uint8).flatten(),  # FIXED FOR ROSBAGS
                is_dense=True
            )
            writer.add_message(pc_cid, ts_ns, typestore.serialize_cdr(pc_msg, "sensor_msgs/msg/PointCloud2"), ts_ns)

            if i % 100 == 0:
                print(f"Simulating: {i}/{len(path)} frames", end='\r')

        writer.finish()
        print(f"\nSaved {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()