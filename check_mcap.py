import sys
from mcap.reader import make_reader

# USAGE: python check_mcap.py your_file.mcap

if len(sys.argv) < 2:
    print("Please provide the path to your .mcap file.")
    sys.exit()

path = sys.argv[1]

print(f"--- Inspecting: {path} ---")
with open(path, "rb") as f:
    reader = make_reader(f)
    print(f"Profile: {reader.get_header().profile}")

    # Count messages per topic
    topic_counts = {}
    topic_types = {}

    for schema, channel, message in reader.iter_messages():
        if channel.topic not in topic_counts:
            topic_counts[channel.topic] = 0
            topic_types[channel.topic] = schema.name
        topic_counts[channel.topic] += 1

    print("\nTOPICS FOUND:")
    print(f"{'TOPIC NAME':<30} | {'MSG TYPE':<30} | {'COUNT'}")
    print("-" * 75)

    for topic, count in topic_counts.items():
        msg_type = topic_types[topic]
        print(f"{topic:<30} | {msg_type:<30} | {count}")