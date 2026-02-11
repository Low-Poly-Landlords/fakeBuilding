from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

INPUT_FILE = "scan_imu_20260203_145509_0.mcap"


def main():
    print(f"Inspecting {INPUT_FILE}...")
    reader = make_reader(open(INPUT_FILE, "rb"), decoder_factories=[DecoderFactory()])

    topics = {}

    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic not in topics:
            topics[channel.topic] = {
                "type": schema.name,
                "count": 0,
                "encoding": getattr(ros_msg, "encoding", "N/A"),
                "width": getattr(ros_msg, "width", 0),
                "height": getattr(ros_msg, "height", 0)
            }
        topics[channel.topic]["count"] += 1

    print("\n--- RESULTS ---")
    for topic, info in topics.items():
        print(f"Topic: {topic}")
        print(f"  Type:     {info['type']}")
        print(f"  Count:    {info['count']}")
        print(f"  Encoding: {info['encoding']}")
        print(f"  Size:     {info['width']}x{info['height']}")
        print("-" * 20)


if __name__ == "__main__":
    main()