from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

import zstandard as zstd


def iter_decoded_messages_with_zstd(reader):
    """Yield (schema, channel, message, decoded_message).

    Tries the reader's decoders first; on UnicodeDecodeError attempts zstd decompression
    of the message bytes and retries the decoder.
    """
    # reuse reader's decoder cache when possible
    for schema, channel, message in reader.iter_messages():
        decoder = reader._decoders.get(message.channel_id)
        if decoder is None:
            for factory in reader._decoder_factories:
                decoder = factory.decoder_for(channel.message_encoding, schema)
                if decoder is not None:
                    reader._decoders[message.channel_id] = decoder
                    break
        if decoder is None:
            # no decoder available
            continue

        data = bytes(message.data or b"")

        # try normal decode first
        try:
            decoded = decoder(data)
            yield schema, channel, message, decoded
            continue
        except UnicodeDecodeError:
            pass
        except Exception:
            # other decode error; we'll still try decompress
            pass

        # try zstd decompression
        try:
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(data)
            decoded = decoder(decompressed)
            yield schema, channel, message, decoded
            continue
        except Exception:
            # failed to decompress or decode; skip
            continue
