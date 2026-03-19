import os
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

TARGET = "raw_scan_0.mcap"
MAX_SAMPLE = 512

if not os.path.exists(TARGET):
    print("Target file not found:", TARGET)
    raise SystemExit(1)

reader = make_reader(open(TARGET, "rb"), decoder_factories=[DecoderFactory()])

print(f"Scanning {TARGET} for decode failures...")
count = 0
for schema, channel, message in reader.iter_messages():
    count += 1
    try:
        # locate or build decoder for this channel
        decoder = reader._decoders.get(message.channel_id)
        if decoder is None:
            for factory in reader._decoder_factories:
                decoder = factory.decoder_for(channel.message_encoding, schema)
                if decoder is not None:
                    reader._decoders[message.channel_id] = decoder
                    break
        if decoder is None:
            # no decoder available; skip
            continue
        # attempt decode
        _ = decoder(message.data)
    except UnicodeDecodeError as e:
        print("\n=== UnicodeDecodeError caught ===")
        print("Topic:", channel.topic)
        print("Schema:", getattr(schema, 'name', None))
        print("Channel ID:", message.channel_id)
        print("Message log_time:", message.log_time)
        print("Message size:", len(message.data) if message.data else 0)
        print("Decoder factory message_encoding:", channel.message_encoding)
        sample = (message.data or b"")[:MAX_SAMPLE]
        print("Sample hex:", sample.hex())
        try:
            print("Sample decoded (errors=replace):", sample.decode('utf-8', errors='replace'))
        except Exception:
            pass
        print("Exception:", e)
        break
    except Exception as e:
        # other decoding errors: print and continue
        print(f"Other decode error at msg {count} topic {channel.topic}: {e}")
        # optionally continue

print("Done scan.")
