import os
import sys
import glob
from collections import defaultdict
from statistics import mean
from mcap.reader import make_reader

TARGETS = ["raw_scan_0.mcap", "enlabopenroom-001.mcap"]

MAGICS = {
    'jpeg': b"\xff\xd8\xff",
    'png': b"\x89PNG"
}

PRINTABLE_THRESHOLD = 0.9


def analyze_file(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    stats = {}
    stats['path'] = path
    stats['topics'] = {}

    MAX_MESSAGES_PER_TOPIC = 200
    SAMPLE_BYTES = 2048

    with open(path, 'rb') as f:
        reader = make_reader(f)
        counts_per_topic = defaultdict(int)
        for schema, channel, message in reader.iter_messages():
            t = channel.topic
            # limit work per topic
            if counts_per_topic[t] >= MAX_MESSAGES_PER_TOPIC:
                continue
            counts_per_topic[t] += 1

            entry = stats['topics'].setdefault(t, {
                'schema': schema.name,
                'count': 0,
                'sizes': [],
                'first_bytes': None,
                'has_jpeg': False,
                'has_png': False,
                'printable_ratios': []
            })

            entry['count'] += 1
            data = message.data or b''
            sz = len(data)
            entry['sizes'].append(sz)
            if entry['first_bytes'] is None:
                entry['first_bytes'] = data[:64]

            sample = data[:SAMPLE_BYTES]
            # quick magic checks on sample
            if MAGICS['jpeg'] in sample:
                entry['has_jpeg'] = True
            if MAGICS['png'] in sample:
                entry['has_png'] = True

            # printable ratio on sample
            if len(sample) > 0:
                printable = sum(1 for b in sample if 32 <= b <= 126)
                entry['printable_ratios'].append(printable / len(sample))

    # finalize aggregates
    for t, entry in stats['topics'].items():
        entry['avg_size'] = mean(entry['sizes']) if entry['sizes'] else 0
        entry['min_size'] = min(entry['sizes']) if entry['sizes'] else 0
        entry['max_size'] = max(entry['sizes']) if entry['sizes'] else 0
        entry['sample_hex'] = entry['first_bytes'].hex() if entry['first_bytes'] is not None else ''
        entry['avg_printable'] = mean(entry['printable_ratios']) if entry['printable_ratios'] else 0.0
    return stats


def print_report(stats):
    if stats is None:
        return
    print(f"\n--- Report for {stats['path']} ---")
    topics = stats['topics']
    for t, info in sorted(topics.items()):
        print(f"Topic: {t}")
        print(f"  Schema:    {info['schema']}")
        print(f"  Count:     {info['count']}")
        print(f"  Size (avg/min/max): {info['avg_size']:.1f}/{info['min_size']}/{info['max_size']}")
        print(f"  Avg printable bytes: {info['avg_printable']:.3f}")
        print(f"  Contains JPEG bytes: {info['has_jpeg']}")
        print(f"  Contains PNG bytes:  {info['has_png']}")
        print(f"  First bytes (hex): {info['sample_hex'][:96]}{'...' if len(info['sample_hex'])>96 else ''}")
        print("-")


def compare_stats(a, b):
    topics_a = set(a['topics'].keys())
    topics_b = set(b['topics'].keys())

    only_a = topics_a - topics_b
    only_b = topics_b - topics_a
    common = topics_a & topics_b

    print(f"\n=== Comparison: {a['path']}  vs  {b['path']} ===")
    if only_a:
        print("Topics only in first file:")
        for t in sorted(only_a):
            print(f"  {t} (schema={a['topics'][t]['schema']})")
    if only_b:
        print("Topics only in second file:")
        for t in sorted(only_b):
            print(f"  {t} (schema={b['topics'][t]['schema']})")

    print("\nCommon topics with differences:")
    for t in sorted(common):
        ia = a['topics'][t]
        ib = b['topics'][t]
        diffs = []
        if ia['schema'] != ib['schema']:
            diffs.append(f"schema: {ia['schema']} != {ib['schema']}")
        if abs(ia['avg_size'] - ib['avg_size']) / (max(1, ia['avg_size'], ib['avg_size'])) > 0.5:
            diffs.append(f"avg_size: {ia['avg_size']:.0f} vs {ib['avg_size']:.0f}")
        if ia['has_jpeg'] != ib['has_jpeg']:
            diffs.append(f"has_jpeg: {ia['has_jpeg']} vs {ib['has_jpeg']}")
        if ia['has_png'] != ib['has_png']:
            diffs.append(f"has_png: {ia['has_png']} vs {ib['has_png']}")
        if abs(ia['avg_printable'] - ib['avg_printable']) > 0.2:
            diffs.append(f"printable_ratio: {ia['avg_printable']:.2f} vs {ib['avg_printable']:.2f}")

        if diffs:
            print(f"  Topic: {t}")
            for d in diffs:
                print(f"    - {d}")


if __name__ == '__main__':
    found = {}
    for t in TARGETS:
        if not os.path.exists(t):
            print(f"Warning: target not found: {t}")
            continue
        found[t] = analyze_file(t)
        print_report(found[t])

    if len(found) == 2:
        a, b = found[TARGETS[0]], found[TARGETS[1]]
        compare_stats(a, b)
    else:
        print("Need both files for comparison.\nFinished per-file reports.")
