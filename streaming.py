"""Streaming version of my classical pipeline (example.ipynb), for the whole session.

Same chain as the notebook, chunk by chunk, so peak RAM stays around 2 GB whatever the session length:
    rigid motion correction (CaImAn, one template for all chunks) -> x2 downsample
    -> spatial high-pass (frame - GaussianBlur sigma 4.5) -> lens mask
    -> 5-frame box filter -> temporal top-hat (opening over 299 frames, ~10 s) = act
    -> calm-period noise map -> rise detector (act step > k * noise, 5 consecutive steps, 2x2 px).

Chunk edges: each AVI file is a chunk. The temporal filters see CONTEXT frames of the neighbouring
chunks on both sides (overlap-save), so the result does not depend on where the files were cut.
The noise map is estimated per chunk (it follows slow drifts; the notebook used one chunk anyway).

Outputs (memory-mapped .npy files, open them with np.load(path, mmap_mode='r')):
    act.npy    float32 (T, 300, 300)  activation movie, T = all raw frames (8935 here)
    rise.npy   bool    (T, 300, 300)  rise[t] = the step from frame t-1 to frame t is part of a detected rise
    noise.npy  float32 (n_chunks, 300, 300)
Frame index = raw frame index = row of the authors' C in the NWB = index into the raw timestamps.
The camera start-up frames (0-3) are left at act = 0, rise = False.
"""
from pathlib import Path
import time

import cv2
import numpy as np
from scipy import ndimage

N_STARTUP = 4      # black / partial frames at the start of the first file
BOX = 5            # temporal box filter, frames
OPEN = 299         # temporal opening, frames (~10 s)
REF = 61           # smoothing used to find calm periods, frames (~2 s)
# a temporal-filter output frame depends on input frames up to BOX//2 + 2*(OPEN//2) = 300 frames away;
# rise needs act 4 frames around each frame (diff + 5-step erosion)
MARGIN = 4
CONTEXT = BOX // 2 + 2 * (OPEN // 2) + MARGIN + 2
STRIP = 50         # rows per strip for the temporal filters (bounds RAM)


def fov_mask(shape):
    """Inscribed circle of the frame: the GRIN lens image circle; nothing outside it."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return np.hypot(yy - (shape[0] - 1) / 2, xx - (shape[1] - 1) / 2) <= shape[0] / 2


def highpass(frames, fov):
    """Spatial high-pass, frame by frame: frame - GaussianBlur((45, 45), 4.5); zero outside the lens."""
    out = np.empty(frames.shape, np.float32)
    for i, f in enumerate(frames):
        out[i] = f - cv2.GaussianBlur(f, (45, 45), 4.5)
    out[:, ~fov] = 0
    return out


def temporal(high):
    """Box filter + temporal top-hat, in row strips to keep the temporaries small."""
    act = np.empty(high.shape, np.float32)
    for r in range(0, high.shape[1], STRIP):
        hs = ndimage.uniform_filter1d(high[:, r:r + STRIP], BOX, axis=0)
        act[:, r:r + STRIP] = hs - ndimage.grey_opening(hs, size=(OPEN, 1, 1))
    return act


def noise_map(act):
    """Per-pixel noise from calm periods: std of (act - 2 s mean) where the 2 s mean is in its lowest 20%."""
    noise = np.empty(act.shape[1:], np.float32)
    for r in range(0, act.shape[1], STRIP):
        a = act[:, r:r + STRIP]
        ref = ndimage.uniform_filter1d(a, REF, axis=0)
        calm = ref <= np.percentile(ref, 20, axis=0)
        noise[r:r + STRIP] = np.nanstd(np.where(calm, a - ref, np.nan), axis=0)
    return noise


def rise_detector(act, noise, k=0.7, run=5, patch=2):
    """rise[j] for the step act[j] -> act[j+1]: above k * noise for `run` consecutive steps over a patch x patch block."""
    rising = np.diff(act, axis=0) > k * noise
    return ndimage.binary_erosion(rising, structure=np.ones((run, patch, patch), bool))


def registered_chunks(files, max_shifts=(35, 35), gSig_filt=(8, 8), keep_mmap=False):
    """Yield each file motion-corrected (rigid, NoRMCorre) and downsampled x2, as float32 (n, 300, 300).

    Same settings as the notebook. The first file sets the template, the others are registered to it,
    so all chunks share one frame of reference. Each chunk's 1.44 GB memmap is deleted after reading.
    """
    import caiman as cm
    from caiman.motion_correction import MotionCorrect
    template = None
    for f in files:
        mc = MotionCorrect([str(f)], max_shifts=max_shifts, gSig_filt=gSig_filt, pw_rigid=False, border_nan='copy')
        mc.motion_correct(template=template, save_movie=True)
        if template is None:
            template = mc.total_template_rig
        Yr, dims, T = cm.load_memmap(mc.mmap_file[0])
        size = (dims[1] // 2, dims[0] // 2)
        out = np.empty((T, size[1], size[0]), np.float32)
        for t in range(T):   # frame t is column t of Yr, contiguous in this F-order memmap
            out[t] = cv2.resize(np.asarray(Yr[:, t]).reshape(dims, order='F'), size, interpolation=cv2.INTER_AREA)
        del Yr
        if not keep_mmap:
            Path(mc.mmap_file[0]).unlink()
        yield out


def stream(chunks, out_dir, n_frames, shape=(300, 300), k=0.7, run=5, patch=2, log=print):
    """Run the chain on an iterable of chunks (float32 movies, already registered and downsampled).

    n_frames: total number of frames over all chunks (to size the output files).
    Returns the activation map (max of act over time), the onset-count map and the chunk start frames.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    act_out = np.lib.format.open_memmap(out_dir / "act.npy", "w+", np.float32, (n_frames, *shape))
    rise_out = np.lib.format.open_memmap(out_dir / "rise.npy", "w+", bool, (n_frames, *shape))
    act_out[:N_STARTUP] = 0
    rise_out[:N_STARTUP + 1] = False
    fov = fov_mask(shape)
    peak = np.zeros(shape, np.float32)
    onsets = np.zeros(shape, np.int32)
    noises, starts = [], []

    it = iter(chunks)
    nxt = next(it)
    nxt = highpass(nxt[N_STARTUP:], fov)        # session starts after the start-up frames
    cur, start, prev_tail = None, N_STARTUP, np.zeros((0, *shape), np.float32)
    last_rise = np.zeros(shape, bool)           # rise at the last frame of the previous chunk (for onsets)
    i = 0
    while nxt is not None:
        t0 = time.time()
        cur = nxt
        try:
            nxt = highpass(next(it), fov)
        except StopIteration:
            nxt = None
        n = len(cur)
        ahead = nxt[:CONTEXT] if nxt is not None else np.zeros((0, *shape), np.float32)
        ext = np.concatenate([prev_tail, cur, ahead])        # this chunk with context on both sides
        act_ext = temporal(ext)
        del ext
        c0 = len(prev_tail)                                   # index of this chunk's first frame in ext
        act = act_ext[c0:c0 + n]
        noise = noise_map(act)

        # rise for this chunk's frames, computed on act with a few frames of margin on both sides
        a0 = max(c0 - MARGIN, 0)
        a1 = min(c0 + n + MARGIN, len(act_ext))
        steps = rise_detector(act_ext[a0:a1], noise, k, run, patch)   # steps[j]: frame a0+j -> a0+j+1
        j0 = c0 - a0 - 1                                     # step that ends on this chunk's first frame
        rise = np.zeros((n, *shape), bool)
        if j0 >= 0:
            rise[:] = steps[j0:j0 + n]
        else:                                                # very first frame of the session: no step into it
            rise[1:] = steps[:n - 1]

        act_out[start:start + n] = act
        rise_out[start:start + n] = rise
        peak = np.maximum(peak, act.max(0))
        onsets += (rise[0] & ~last_rise) + (rise[1:] & ~rise[:-1]).sum(0)
        last_rise = rise[-1].copy()
        noises.append(noise)
        starts.append(start)
        log(f"chunk {i}: frames {start}-{start + n - 1}, {rise.sum()} rise voxels, {time.time() - t0:.0f} s")

        prev_tail = cur[-CONTEXT:]
        start += n
        i += 1
        del act_ext, act, rise, steps, cur

    act_out.flush(); rise_out.flush()
    del act_out, rise_out
    noises = np.array(noises)
    np.save(out_dir / "noise.npy", noises)
    return peak, onsets, noises, np.array(starts)


def run_session(files, out_dir="out/session", **kw):
    """Whole session from the raw AVI files: motion correction + chain. Writes act.npy, rise.npy, noise.npy."""
    import av
    n_frames = 0
    for f in files:
        with av.open(str(f)) as c:
            n_frames += c.streams.video[0].frames
    return stream(registered_chunks(files), out_dir, n_frames, **kw)
