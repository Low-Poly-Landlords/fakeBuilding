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
OUTPUT_FILENAME = "hotel_scan.mcap"

# Sensor Settings (High Quality)
FOV_MIN, FOV_MAX = -45.0, 45.0
RINGS = 32  # Back to High Quality
POINTS_PER_RING = 512  # Back to High Quality
WALK_SPEED = 1.5
FPS = 10

# Building Constants
FLOOR_HEIGHT = 3.5
ROOM_WIDTH = 5.0
ROOM_DEPTH = 6.0
ROOMS_PER_SIDE = 4

# Setup Device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running computation on: {device.type.upper()}")


def gpu_raycast(rays_o, rays_d, box_min, box_max):
    """
    rays_o: [N_rays, 3]
    rays_d: [N_rays, 3]
    box_min: [M_boxes, 3]
    box_max: [M_boxes, 3]

    Returns: [N_rays] distances
    """
    # 1. RESHAPE FOR BROADCASTING
    # We want to compare every Ray (N) against every Box (M)
    # Rays shape: [N, 1, 3]
    # Boxes shape: [1, M, 3]
    ro = rays_o.unsqueeze(1)
    rd = rays_d.unsqueeze(1)
    bmin = box_min.unsqueeze(0)
    bmax = box_max.unsqueeze(0)

    # 2. SLAB METHOD (Parallelized)
    # Avoid div by zero
    inv_d = 1.0 / (rd + 1e-6)

    t0 = (bmin - ro) * inv_d
    t1 = (bmax - ro) * inv_d

    t_min = torch.min(t0, t1)
    t_max = torch.max(t0, t1)

    # Largest entry time across X,Y,Z
    t_enter = torch.max(t_min, dim=2).values
    # Smallest exit time across X,Y,Z
    t_exit = torch.min(t_max, dim=2).values

    # 3. MASK HITS
    # Hit if: (t_exit >= t_enter) AND (t_exit > 0)
    hit_mask = (t_exit >= t_enter) & (t_exit > 0)

    # 4. FIND CLOSEST
    # Set misses to infinity
    # Shape: [N_rays, M_boxes]
    distances = torch.where(hit_mask, t_enter, torch.tensor(float('inf'), device=device))

    # For each ray, find the SMALLEST distance across all boxes
    # Shape: [N_rays]
    closest_dists, _ = torch.min(distances, dim=1)

    return closest_dists


# --- PATH GENERATION (Same as before) ---
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
    for i in range(len(waypoints) - 1):
        p1 = np.array(waypoints[i])
        p2 = np.array(waypoints[i + 1])
        dist = np.linalg.norm(p2 - p1)
        if dist > 5.0 and abs(p1[2] - p2[2]) > 1.0:
            full_path.append((*p2, 0))
            continue
        num_steps = max(1, int((dist / WALK_SPEED) * FPS))
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        yaw = math.atan2(dy, dx)
        for s in range(num_steps):
            t = s / num_steps
            pos = p1 + (p2 - p1) * t
            look_yaw = yaw + math.sin(t * 5) * 0.5
            full_path.append((*pos, look_yaw))
    return full_path


def main():
    # 1. LOAD LAYOUT TO GPU
    print(f"Loading Layout to {device}...")
    with open(INPUT_LAYOUT, 'r') as f:
        layout_data = json.load(f)

    b_mins = []
    b_maxs = []
    for b in layout_data:
        b_mins.append([b['min_x'], b['min_y'], b['min_z']])
        b_maxs.append([b['max_x'], b['max_y'], b['max_z']])

    # Create Tensors [M_boxes, 3]
    box_min_t = torch.tensor(b_mins, dtype=torch.float32, device=device)
    box_max_t = torch.tensor(b_maxs, dtype=torch.float32, device=device)

    path = interpolate_path(generate_waypoints())
    print(f"Path length: {len(path)} frames.")

    # 2. PRECOMPUTE RAYS (Spherical -> Cartesian)
    elevs = torch.linspace(math.radians(FOV_MIN), math.radians(FOV_MAX), RINGS, device=device)
    azis = torch.linspace(0, 2 * math.pi, POINTS_PER_RING, device=device)
    e_grid, a_grid = torch.meshgrid(elevs, azis, indexing='ij')

    # Base Rays (x=forward)
    base_x = torch.cos(e_grid) * torch.cos(a_grid)
    base_y = torch.cos(e_grid) * torch.sin(a_grid)
    base_z = torch.sin(e_grid)

    # Flatten: [N_rays, 3]
    base_rays = torch.stack([base_x.flatten(), base_y.flatten(), base_z.flatten()], dim=1)

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

    print(f"Starting GPU Simulation ({RINGS * POINTS_PER_RING} rays/frame)...")
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

            # --- A. ROTATE RAYS (ON GPU) ---
            c, s = math.cos(yaw), math.sin(yaw)

            # Rotate X and Y components
            rot_x = base_rays[:, 0] * c - base_rays[:, 1] * s
            rot_y = base_rays[:, 0] * s + base_rays[:, 1] * c
            rot_z = base_rays[:, 2]

            current_rays_d = torch.stack([rot_x, rot_y, rot_z], dim=1)

            # Create Origins [N, 3]
            current_rays_o = torch.tensor([rx, ry, rz], device=device).expand_as(current_rays_d)

            # --- B. GPU RAYCASTING ---
            closest_dists = gpu_raycast(current_rays_o, current_rays_d, box_min_t, box_max_t)

            # --- C. FILTER & DOWNLOAD ---
            # Mask valid hits (Range < 20m)
            valid_mask = closest_dists < 20.0

            # Filter
            valid_dists = closest_dists[valid_mask]

            # Calculate Local Point Cloud (Sensor Frame)
            # We use the UN-ROTATED base rays because LiDAR data is relative to the sensor
            valid_base = base_rays[valid_mask]

            points_xyz = valid_base * valid_dists.unsqueeze(1)

            # Move to CPU for writing (The inevitable bottleneck)
            points_cpu = points_xyz.cpu().numpy().astype(np.float32)

            # --- D. WRITE MESSAGES ---
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
                data=points_cpu.view(np.uint8).flatten(), is_dense=True
            )
            writer.add_message(pc_cid, ts_ns, typestore.serialize_cdr(pc_msg, "sensor_msgs/msg/PointCloud2"), ts_ns)

            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"GPU Sim: {i}/{len(path)} frames | {elapsed:.1f}s elapsed", end='\r')

        writer.finish()
        print(f"\nDone! Saved {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
