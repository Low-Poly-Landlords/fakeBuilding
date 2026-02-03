#run this second, it simulates a person walking through the building
#doing a lidar scan. This will take a while to run (at least 2 hours).

import numpy as np
import math
import json
from mcap.writer import Writer
from rosbags.typesys import get_typestore, Stores
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
INPUT_LAYOUT = "hotel_layout.json"
OUTPUT_FILENAME = "hotel_scan.mcap"

# Sensor Settings
FOV_MIN, FOV_MAX = -45.0, 45.0  # Wide view to see ceiling/floor
RINGS = 32  # Vertical resolution
POINTS_PER_RING = 360  # Horizontal resolution
WALK_SPEED = 1.0  # Meters per second
FPS = 10  # Scans per second

# Building Constants (Must match generator)
FLOOR_HEIGHT = 3.5
ROOM_WIDTH = 5.0
ROOM_DEPTH = 6.0
HALLWAY_WIDTH = 2.0
ROOMS_PER_SIDE = 4


# --- MATH HELPERS ---
class Box:
    def __init__(self, data):
        self.name = data['name']
        self.min_x = data['min_x']
        self.max_x = data['max_x']
        self.min_y = data['min_y']
        self.max_y = data['max_y']
        self.min_z = data['min_z']
        self.max_z = data['max_z']

    def intersect(self, ro, rd):
        # Optimized Slab Method
        # ro: Ray Origin (x,y,z), rd: Ray Direction (x,y,z)
        with np.errstate(divide='ignore'):
            t1 = (self.min_x - ro[0]) / rd[0]
            t2 = (self.max_x - ro[0]) / rd[0]
            t3 = (self.min_y - ro[1]) / rd[1]
            t4 = (self.max_y - ro[1]) / rd[1]
            t5 = (self.min_z - ro[2]) / rd[2]
            t6 = (self.max_z - ro[2]) / rd[2]

        tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
        tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))

        if tmax < 0 or tmin > tmax:
            return float('inf')
        return tmin


def generate_waypoints():
    """ Creates a methodical path through the building """
    waypoints = []

    # 1. FLOOR 1: LOBBY
    z = 1.5  # Eye height
    # Start outside -> Walk in -> Reception -> Office -> Elevator
    waypoints.extend([
        (0, -6, z),  # Outside
        (0, 0, z),  # Center Lobby
        (5, 0, z),  # Near Desk
        (6.5, 0, z),  # Door to Office
        (8.5, 0, z),  # Inside Office
        (6.5, 0, z),  # Exit Office
        (0, 0, z),  # Back to center
        (-1.5, 0, z)  # At Elevator
    ])

    # 2. FLOORS 2-5
    for f in range(1, 5):  # Floor index 1 to 4
        z = (f * FLOOR_HEIGHT) + 1.5

        # Teleport to Elevator Lobby
        waypoints.append((-1.5, 0, z))

        # Walk down Hallway visiting rooms
        for r in range(ROOMS_PER_SIDE):
            x_center = r * ROOM_WIDTH + (ROOM_WIDTH / 2)

            # Walk to outside room
            waypoints.append((x_center, 0, z))

            # Visit Top Room (Y > 0)
            waypoints.append((x_center, 1.5, z))  # Doorway
            waypoints.append((x_center, 3.5, z))  # Inside
            waypoints.append((x_center, 1.5, z))  # Exit

            # Back to Hall
            waypoints.append((x_center, 0, z))

            # Visit Bot Room (Y < 0)
            waypoints.append((x_center, -1.5, z))  # Doorway
            waypoints.append((x_center, -3.5, z))  # Inside
            waypoints.append((x_center, -1.5, z))  # Exit

            # Back to Hall
            waypoints.append((x_center, 0, z))

        # Visit Linen Closet at end
        end_x = (ROOMS_PER_SIDE * ROOM_WIDTH)
        waypoints.append((end_x, 0, z))
        waypoints.append((end_x + 1.5, 0, z))  # Inside closet
        waypoints.append((end_x, 0, z))

        # Walk back to Elevator (Fast return)
        waypoints.append((-1.5, 0, z))

    return waypoints


def interpolate_path(waypoints):
    """ Expands sparse waypoints into per-frame poses """
    full_path = []

    for i in range(len(waypoints) - 1):
        p1 = np.array(waypoints[i])
        p2 = np.array(waypoints[i + 1])

        dist = np.linalg.norm(p2 - p1)

        # If distance is huge (elevator teleport), just jump
        if dist > 5.0 and abs(p1[2] - p2[2]) > 1.0:
            full_path.append((*p2, 0))  # Yaw doesn't matter on teleport
            continue

        # Calculate steps
        duration = dist / WALK_SPEED
        num_steps = int(duration * FPS)
        if num_steps < 1: num_steps = 1

        # Calculate Yaw (Look direction)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        yaw = math.atan2(dy, dx)

        for s in range(num_steps):
            t = s / num_steps
            pos = p1 + (p2 - p1) * t

            # Add some "head bobble" or looking around
            look_yaw = yaw + math.sin(t * 10) * 0.3

            full_path.append((*pos, look_yaw))

    return full_path


# --- MAIN ---
def main():
    print(f"Loading {INPUT_LAYOUT}...")
    with open(INPUT_LAYOUT, 'r') as f:
        data = json.load(f)

    boxes = [Box(b) for b in data]
    print(f"Loaded {len(boxes)} geometry objects.")

    print("Planning Path...")
    waypoints = generate_waypoints()
    path = interpolate_path(waypoints)
    print(f"Generated path with {len(path)} frames (~{len(path) / FPS / 60:.1f} minutes sim time)")

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    TFMessage = typestore.types['tf2_msgs/msg/TFMessage']
    TransformStamped = typestore.types['geometry_msgs/msg/TransformStamped']
    Header = typestore.types['std_msgs/msg/Header']
    Time = typestore.types['builtin_interfaces/msg/Time']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']
    QuaternionMsg = typestore.types['geometry_msgs/msg/Quaternion']
    PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
    PointField = typestore.types['sensor_msgs/msg/PointField']

    # Precompute Ray Angles
    elevations = np.linspace(math.radians(FOV_MIN), math.radians(FOV_MAX), RINGS)
    azimuths = np.linspace(0, 2 * math.pi, POINTS_PER_RING)

    # Precompute sin/cos for optimization
    sin_el = np.sin(elevations)
    cos_el = np.cos(elevations)

    print(f"Starting Simulation -> {OUTPUT_FILENAME}")

    with open(OUTPUT_FILENAME, "wb") as f:
        writer = Writer(f)
        writer.start()

        # Schemas (Simplified registration for brevity, assume standard ROS2)
        tf_schema = """geometry_msgs/TransformStamped[] transforms\n================================================================================\nMSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\ngeometry_msgs/Transform transform\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n================================================================================\nMSG: geometry_msgs/Transform\ngeometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation\n================================================================================\nMSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n================================================================================\nMSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w"""
        pc_schema = """std_msgs/Header header\nuint32 height\nuint32 width\nsensor_msgs/PointField[] fields\nbool is_bigendian\nuint32 point_step\nuint32 row_step\nuint8[] data\nbool is_dense\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n================================================================================\nMSG: sensor_msgs/PointField\nuint8 INT8=1\nuint8 UINT8=2\nuint8 INT16=3\nuint8 UINT16=4\nuint8 INT32=5\nuint8 UINT32=6\nuint8 FLOAT32=7\nuint8 FLOAT64=8\nstring name\nuint32 offset\nuint8 datatype\nuint32 count"""

        tf_sid = writer.register_schema(name="tf2_msgs/msg/TFMessage", encoding="ros2msg", data=tf_schema.encode())
        pc_sid = writer.register_schema(name="sensor_msgs/msg/PointCloud2", encoding="ros2msg", data=pc_schema.encode())
        tf_cid = writer.register_channel(topic="/tf", message_encoding="cdr", schema_id=tf_sid)
        pc_cid = writer.register_channel(topic="/lidar/points", message_encoding="cdr", schema_id=pc_sid)

        start_ns = 1000000000

        for i, (rx, ry, rz, yaw) in enumerate(path):
            ts = start_ns + int(i * (1e9 / FPS))

            # 1. TF MSG
            q = R.from_euler('z', yaw).as_quat()
            tf_msg = TFMessage(transforms=[TransformStamped(
                header=Header(stamp=Time(sec=ts // 10 ** 9, nanosec=ts % 10 ** 9), frame_id="map"),
                child_frame_id="base_link",
                transform=typestore.types['geometry_msgs/msg/Transform'](
                    translation=Vector3(x=rx, y=ry, z=rz - 1.5),  # Base_link is on floor, cam is at eye level
                    rotation=QuaternionMsg(x=q[0], y=q[1], z=q[2], w=q[3])
                )
            )])
            writer.add_message(tf_cid, ts, typestore.serialize_cdr(tf_msg, "tf2_msgs/msg/TFMessage"), ts)

            # 2. LIDAR RAYCAST
            points = []

            # OPTIMIZATION: Only check boxes near current Z level
            relevant_boxes = [b for b in boxes if b.min_z < rz + 1.0 and b.max_z > rz - 3.0]

            for r_idx in range(RINGS):
                # Calculate Z component of ray direction
                dz = sin_el[r_idx]
                r_cos_el = cos_el[r_idx]

                for a_idx in range(POINTS_PER_RING):
                    angle = yaw + azimuths[a_idx]
                    dx = math.cos(angle) * r_cos_el
                    dy = math.sin(angle) * r_cos_el

                    min_dist = 20.0  # Max Range

                    # Intersect Geometry
                    for box in relevant_boxes:
                        d = box.intersect([rx, ry, rz], [dx, dy, dz])
                        if d < min_dist:
                            min_dist = d

                    if min_dist < 19.0:
                        # Add noise
                        dist = min_dist + np.random.normal(0, 0.01)

                        # Convert to Local Frame
                        # Hit Point World
                        hx, hy, hz = rx + dist * dx, ry + dist * dy, rz + dist * dz
                        # World -> Body Translation
                        tx, ty, tz = hx - rx, hy - ry, hz - (rz - 1.5)  # relative to base_link
                        # World -> Body Rotation
                        c, s = math.cos(-yaw), math.sin(-yaw)
                        lx = tx * c - ty * s
                        ly = tx * s + ty * c
                        lz = tz

                        points.append([lx, ly, lz])

            # 3. WRITE PC MSG
            np_pts = np.array(points, dtype=np.float32)
            pc_msg = PointCloud2(
                header=Header(stamp=Time(sec=ts // 10 ** 9, nanosec=ts % 10 ** 9), frame_id="base_link"),
                height=1, width=len(points),
                fields=[
                    PointField(name="x", offset=0, datatype=7, count=1),
                    PointField(name="y", offset=4, datatype=7, count=1),
                    PointField(name="z", offset=8, datatype=7, count=1)
                ],
                is_bigendian=False, point_step=12, row_step=12 * len(points),
                data=np_pts.view(np.uint8).flatten(), is_dense=True
            )
            writer.add_message(pc_cid, ts, typestore.serialize_cdr(pc_msg, "sensor_msgs/msg/PointCloud2"), ts)

            print(f"Simulating: {i}/{len(path)} frames | Floor {int(rz / 3.5) + 1}", end='\r')

        writer.finish()
        print(f"\nCompleted! Saved {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()