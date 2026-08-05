"""Convert a slice of the canonical reacher h5 into a lance dataset that is
BYTE-COMPATIBLE with the on-policy recordings.

`build_dyna_mix.py` concatenates the expert and on-policy sides, so they must
agree on both the Arrow schema and the ENCODING of the pixel blobs. The recorder
(SWM_RECORD_PATH) writes:

    episode_idx int32
    step_idx    int32
    pixels      binary                       (JPEG blob, PIL, quality 95)
    action      fixed_size_list<float>[2]

TWO BUGS LIVED HERE, both of which produced a dataset that looked correct and
failed much later:

1. TYPES. A first version used list<uint8> / list<float> / int64 and the mix
   build died with
     LanceError(Arrow): C Data interface error: The datatype "List(UInt8)"
     expects 2 buffers, but requested 2
   so the types below are copied from the recorder rather than chosen.

2. ENCODING. The second version got the types right but wrote RAW HxWx3 uint8
   (`frame.tobytes()`, 150528 B/frame) while the recorder writes JPEG
   (~8.6 kB/frame). pa.binary() accepts both, so the mix built cleanly and the
   failure only surfaced deep inside the dataloader, hours later, as
     PIL.UnidentifiedImageError: cannot identify image file <BytesIO>
   which killed the Dyna fine-tune. Frames are now encoded with the library's
   OWN `_encode_frame`, so the bytes match by construction rather than by
   guessing the quality setting.

   Matching the encoder matters beyond mere decodability: the mix is 50/50, so
   if the two halves carried different compression artifacts the world model
   could learn to tell expert frames from on-policy frames -- a spurious cue
   that would quietly corrupt the entire Dyna comparison.

EPISODE RANGE MATTERS: the expert side must exclude the evaluation pool
8000:10000 or the fine-tune is train-on-test -- the exact defect found in the
round-1 mix (400k leaked rows, 10.1%).
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import hdf5plugin  # noqa: F401
import lance
import numpy as np
import pyarrow as pa

sys.path.insert(0, "/workspace/swm_cem")
from stable_worldmodel.data.formats.lance import (  # noqa: E402
    _DEFAULT_JPEG_QUALITY,
    _encode_frame,
)

p = argparse.ArgumentParser()
p.add_argument("--h5", default="/workspace/datasets_canon/lewm-reacher/reacher.h5")
p.add_argument("--out", required=True)
p.add_argument("--ep-lo", type=int, default=0)
p.add_argument("--ep-hi", type=int, default=1500)
p.add_argument("--batch", type=int, default=1000)
p.add_argument("--workers", type=int, default=16)
a = p.parse_args()

f = h5py.File(a.h5, "r")
ep = f["ep_idx"][:]
keep = np.nonzero((ep >= a.ep_lo) & (ep < a.ep_hi))[0]
assert len(keep), "empty episode range"
lo, hi = int(keep[0]), int(keep[-1]) + 1
n = hi - lo
leak = ((ep[lo:hi] >= 8000) & (ep[lo:hi] < 10000)).sum()
print(f"episodes [{a.ep_lo},{a.ep_hi}) -> rows [{lo},{hi}) = {n}", flush=True)
print(f"eval-pool rows included: {leak} (must be 0)", flush=True)
assert leak == 0, "expert side would leak the evaluation pool"
print(f"encoding JPEG q={_DEFAULT_JPEG_QUALITY} with {a.workers} workers", flush=True)

schema = pa.schema([
    pa.field("episode_idx", pa.int32()),
    pa.field("step_idx", pa.int32()),
    pa.field("pixels", pa.binary()),
    pa.field("action", pa.list_(pa.float32(), 2)),
])


def _enc(frame):
    return _encode_frame(frame, _DEFAULT_JPEG_QUALITY)


def batches(pool):
    done = 0
    for s in range(lo, hi, a.batch):
        e = min(s + a.batch, hi)
        px = np.ascontiguousarray(f["pixels"][s:e], dtype=np.uint8)
        act = np.ascontiguousarray(f["action"][s:e], dtype=np.float32)
        blobs = list(pool.map(_enc, list(px), chunksize=32))
        tb = pa.table({
            "episode_idx": pa.array(f["ep_idx"][s:e].astype(np.int32)),
            "step_idx": pa.array(f["step_idx"][s:e].astype(np.int32)),
            "pixels": pa.array(blobs, type=pa.binary()),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(act.reshape(-1), type=pa.float32()), 2),
        }, schema=schema)
        done += e - s
        if done % 50000 < a.batch:
            mb = np.mean([len(b) for b in blobs])
            print(f"  {done}/{n} rows  (mean blob {mb:.0f} B)", flush=True)
        for b in tb.to_batches():
            yield b


with ProcessPoolExecutor(max_workers=a.workers) as pool:
    lance.write_dataset(batches(pool), a.out, schema=schema, mode="create",
                        max_rows_per_file=25_000)

ds = lance.dataset(a.out)
e2 = ds.to_table(columns=["episode_idx"]).column(0).to_numpy()
blob = ds.to_table(columns=["pixels"], limit=1).column(0)[0].as_py()
print(f"wrote {len(e2)} rows, {len(np.unique(e2))} episodes, monotone="
      f"{bool((np.diff(e2) >= 0).all())} -> {a.out}", flush=True)
print(f"first blob: len={len(blob)} head={blob[:4]!r}", flush=True)
assert blob[:2] == b"\xff\xd8", "not a JPEG -- the loader's decoder will reject this"
for fld in ds.schema:
    print(f"   {fld.name} {fld.type}", flush=True)
print("H5_TO_LANCE_DONE", flush=True)
