"""Verify the downloaded Inkling checkpoint is complete: every shard referenced by
the index exists on disk and its safetensors header is parseable."""
import json, os, struct, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/Users/david/llm/Inkling-src"
idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
bad = []
for s in shards:
    p = os.path.join(SRC, s)
    if not os.path.exists(p):
        bad.append((s, "missing")); continue
    try:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n).decode())
            # sanity: file is at least header + declared data
            maxend = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
            if os.path.getsize(p) < 8 + n + maxend:
                bad.append((s, "truncated"))
    except Exception as e:
        bad.append((s, f"header:{e}"))

# also ensure no leftover .incomplete
inc_dir = os.path.join(SRC, ".cache", "huggingface", "download")
inc = [f for f in (os.listdir(inc_dir) if os.path.isdir(inc_dir) else []) if f.endswith(".incomplete")]

print(f"shards referenced: {len(shards)} | bad: {len(bad)} | incomplete: {len(inc)}")
for s, why in bad[:20]:
    print("  BAD:", s, why)
if bad or inc:
    sys.exit(1)
print("INTEGRITY OK")
