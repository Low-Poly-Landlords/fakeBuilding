import glob
import os
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

files = glob.glob("*.mcap")
if not files:
    print("No .mcap files found in workspace.")
    raise SystemExit(1)

files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
target = files[0]
print(f"Inspecting newest .mcap: {target}\n")

try:
    reader = make_reader(open(target, "rb"), decoder_factories=[DecoderFactory()])
except Exception as e:
    print("Failed to open/read with DecoderFactory:", e)
    raise

topics = {}
for schema, channel, message, ros_msg in reader.iter_decoded_messages():
    t = channel.topic
    if t not in topics:
        topics[t] = {
            "type": schema.name,
            "count": 0,
            "encoding": getattr(ros_msg, "encoding", "N/A"),
            "width": getattr(ros_msg, "width", 0),
            "height": getattr(ros_msg, "height", 0)
        }
    topics[t]["count"] += 1

print("--- TOPICS ---")
for topic, info in sorted(topics.items()):
    print(f"Topic: {topic}")
    print(f"  Type:     {info['type']}")
    print(f"  Count:    {info['count']}")
    print(f"  Encoding: {info['encoding']}")
    print(f"  Size:     {info['width']}x{info['height']}")
    print("-")
