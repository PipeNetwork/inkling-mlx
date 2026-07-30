"""Verify a bf16 build is an exact passthrough of the source checkpoint.

`--bits None` conversion applies only name-mapping, layout transforms and a cast that
is a no-op for a bf16 source — so every output tensor must be *bit-identical* to the
transformed input. That is a stronger check than perplexity, and it is the only
practical one for a build larger than the machine's RAM.

    python scripts/verify_bf16_passthrough.py <SRC> <BUILD> [n_source_shards]
"""
import json, os, sys
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inkling_mlx.convert import map_name, transform

SRC = sys.argv[1]
DST = sys.argv[2]
NSH = int(sys.argv[3]) if len(sys.argv) > 3 else 4

src_idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
dst_idx = json.load(open(os.path.join(DST, "model.safetensors.index.json")))["weight_map"]

# group source tensors by shard; check a strided sample of shards end to end
shards = sorted(set(src_idx.values()))
pick = shards[:: max(1, len(shards) // NSH)][:NSH]

checked = mismatch = dropped = 0
for sh in pick:
    names = [n for n, s in src_idx.items() if s == sh]
    src_t = mx.load(os.path.join(SRC, sh))
    cache = {}
    for n in names:
        outs = map_name(n)
        if not outs:                       # model.mtp.* is intentionally dropped
            dropped += 1
            assert not any(k.startswith(n) for k in dst_idx), f"{n} should be dropped"
            continue
        for out_name, kind in outs:
            assert out_name in dst_idx, f"missing in build: {out_name}"
            f = dst_idx[out_name]
            if f not in cache:
                cache = {f: mx.load(os.path.join(DST, f))}   # one shard resident
            got = cache[f][out_name]
            want = transform(src_t[n], kind)
            if got.dtype != want.dtype or got.shape != want.shape or not mx.array_equal(got, want):
                mismatch += 1
                print(f"  MISMATCH {out_name}: got {got.shape}/{got.dtype} want {want.shape}/{want.dtype}")
            checked += 1
    del src_t, cache
    print(f"[verify] {sh}: {checked} tensors checked, {mismatch} mismatched", flush=True)

print(f"\n[verify] shards sampled: {len(pick)}/{len(shards)}")
print(f"[verify] tensors bit-identical: {checked - mismatch}/{checked} | mtp dropped: {dropped}")
print("BF16 PASSTHROUGH OK" if mismatch == 0 else "BF16 PASSTHROUGH FAILED")
sys.exit(1 if mismatch else 0)
