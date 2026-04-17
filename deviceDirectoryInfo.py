from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

INPUT_FILE = "BIG SCAN/deluxe2_20260410_151732/raw_scan/raw_scan_0.mcap"


def main():
    print(f"Inspecting {INPUT_FILE}...")

    # Open the reader without the automatic decoder factory
    reader = make_reader(open(INPUT_FILE, "rb"))
    decoder_factory = DecoderFactory()

    topics = {}

    # iter_messages() does NOT decode the heavy payload, so it won't crash on bad UTF-8
    # and it runs significantly faster.
    for schema, channel, message in reader.iter_messages():
        if channel.topic not in topics:
            topics[channel.topic] = {
                "type": schema.name,
                "count": 0,
                "encoding": "N/A",
                "width": 0,
                "height": 0,
                "decoded_once": False
            }

        topics[channel.topic]["count"] += 1

        # Only try to decode until we successfully grab the metadata once per topic
        if not topics[channel.topic]["decoded_once"]:
            try:
                # Manually decode the message
                decoder = decoder_factory.decoder_for(schema.name, schema)
                ros_msg = decoder(message.data)

                # Extract the required info
                topics[channel.topic]["encoding"] = getattr(ros_msg, "encoding", "N/A")
                topics[channel.topic]["width"] = getattr(ros_msg, "width", 0)
                topics[channel.topic]["height"] = getattr(ros_msg, "height", 0)

                # Flag as successful so we don't waste CPU decoding this topic again
                topics[channel.topic]["decoded_once"] = True

            except UnicodeDecodeError:
                # If this specific message is corrupt, safely ignore it and let the loop
                # try again on the next message in this topic.
                pass
            except Exception:
                # Catch any other random serialization errors
                pass

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