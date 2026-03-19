from mcap.reader import make_reader
from mcap_zstd_helper import iter_decoded_messages_with_zstd
from mcap_ros2.decoder import DecoderFactory

INPUT = "raw_scan_0.mcap"

reader = make_reader(open(INPUT, "rb"), decoder_factories=[DecoderFactory()])

counts = {}
failed = 0
for schema, channel, message, ros_msg in iter_decoded_messages_with_zstd(reader):
    t = channel.topic
    counts[t] = counts.get(t, 0) + 1

print("Decoded message counts (per topic):")
for k, v in counts.items():
    print(f"  {k}: {v}")
print("Done.")
