from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# REPLACE with your filename
path = "scan_imu_20260203_145509_0.mcap"

with open(path, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic == '/scan':
            print(f"Your Lidar Frame ID is: '{ros_msg.header.frame_id}'")
            break