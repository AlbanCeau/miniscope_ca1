"""Full-session results from the streamed pipeline (streaming.py), compared with the authors' published extraction.

Boilerplate written with an AI assistant (Claude Code). The method is the one developed in the notebook:
activation movie act, rise detector, zones = groups of rise events with overlapping footprints, zone traces = mean
of act over the zone. The reference is the authors' CNMF-E extraction in the session NWB (accepted components only,
footprints 600x600 block-summed to our 300x300, traces C), aligned by frame index (C row k = raw frame k).

Everything streams over the session in chunks, so RAM stays around 1-2 GB.
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.csgraph import connected_components

from streaming import N_STARTUP, fov_mask

FPS = 29.78
CHUNK = 1000


# ---------------------------------------------------------------- authors' extraction (NWB)

def load_authors(nwb_path, shape=(300, 300), thr=0.2, which="accepted"):
    """Accepted (or "rejected") components: footprints (pixels x K, sparse, downsampled to `shape`), masks (footprint > thr * max),
    traces C (K x T_C), centres (row, col) in our pixels."""
    import pynwb
    io = pynwb.NWBHDF5IO(str(nwb_path), "r")
    nwb = io.read()
    ps = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
    accepted = np.asarray(ps["accepted"][:], bool)
    acc = np.flatnonzero(accepted if which == "accepted" else ~accepted)
    H, W = shape
    cols = []
    for i in acc:                                   # one 600x600 mask at a time
        m = np.asarray(ps["image_mask"][i], np.float32)
        f = m.reshape(H, m.shape[0] // H, W, m.shape[1] // W).sum((1, 3))
        cols.append(sp.csc_matrix(f.reshape(-1, 1)))
    A = sp.hstack(cols).tocsc()
    Ad = A.toarray()
    masks = sp.csc_matrix((Ad > thr * Ad.max(0)).astype(np.float32))
    C = np.asarray(nwb.processing["ophys"]["Fluorescence"]["C"].data[:, acc]).T
    centres = np.c_[ps["center_y"][acc], ps["center_x"][acc]] * H / 600 - 0.25   # block centre of a 2x2 sum
    io.close()
    return dict(A=A, masks=masks, C=C, centres=centres, idx=acc, shape=shape)


def grid_offset(peak, A, shape, max_shift=3):
    """Integer shift (dy, dx) of the authors' footprints that best overlaps my activation map (their motion
    correction used a different template from mine). Brute force over +-max_shift px."""
    a = np.asarray(A.sum(1)).reshape(shape)
    p = np.where(fov_mask(shape), peak, 0)
    score = {(dy, dx): float((np.roll(a, (dy, dx), axis=(0, 1)) * p).sum())
             for dy in range(-max_shift, max_shift + 1) for dx in range(-max_shift, max_shift + 1)}
    return max(score, key=score.get)


def shift_footprints(au, dy, dx):
    """Move the authors' footprints by (dy, dx) pixels onto my grid."""
    H, W = au["shape"]
    def sh(M):
        d = M.toarray().reshape(H, W, -1)
        d = np.roll(d, (dy, dx), axis=(0, 1))
        return sp.csc_matrix(d.reshape(H * W, -1))
    au = dict(au)
    au["A"], au["masks"] = sh(au["A"]), sh(au["masks"])
    au["centres"] = au["centres"] + [dy, dx]
    return au


# ---------------------------------------------------------------- my results, streamed from disk

def peak_map(act):
    peak = np.zeros(act.shape[1:], np.float32)
    for t0 in range(0, len(act), CHUNK):
        peak = np.maximum(peak, act[t0:t0 + CHUNK].max(0))
    return peak


def label_sparse(rise):
    """Same as ndimage.label(rise) (face connectivity, same label order), but only touches the active voxels:
    rise is ~0.02% full, so this never builds the dense (T, H, W) label array (3.2 GB of int32 for a session)."""
    T, H, W = rise.shape
    t, y, x = np.nonzero(rise)                      # C order, so the keys below are sorted
    key = (t * H + y) * W + x
    n = len(key)
    rows, cols = [], []
    for off, ok in ((1, x < W - 1), (W, y < H - 1), (H * W, t < T - 1)):   # +x, +y, +t neighbours
        j = np.searchsorted(key, key + off)
        hit = ok & (j < n)
        hit[hit] = key[j[hit]] == key[hit] + off
        rows.append(np.flatnonzero(hit)); cols.append(j[hit])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    G = sp.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(n, n))
    n_ev, lab = connected_components(G, directed=False)   # numbered by first voxel in raster order, like ndimage.label
    return t, y, x, lab, n_ev


def session_events(rise):
    """3-D rise events over the whole session. Returns footprints F (events x pixels) and each event's first frame."""
    T, H, W = rise.shape
    t, y, x, lab, n_ev = label_sparse(rise)
    t_first = t[np.unique(lab, return_index=True)[1]]           # voxels are in time order
    F = sp.csr_matrix((np.ones(len(lab)), (lab, y * W + x)), shape=(n_ev, H * W))
    F.data[:] = 1
    return F, t_first


def group_zones(F, shape, iou=0.5, min_events=2, min_area=9):
    """Zones = connected groups of events whose footprints overlap (IoU > iou); keep zones seen >= min_events times
    and covering >= min_area px (3x3 ~ 10 um: a soma core; smaller ones are mostly repeated noise blobs).
    Returns the zone map (0 = none, k = zone k-1), events per zone, zone of each event."""
    n_ev = F.shape[0]
    area = F.getnnz(1)
    inter = (F @ F.T).tocoo()                                   # pixels shared by each pair of events
    ratio = inter.data / (area[inter.row] + area[inter.col] - inter.data)
    G = sp.csr_matrix((ratio, (inter.row, inter.col)), shape=(n_ev, n_ev)) > iou   # keeps only the True entries
    n_z, ev_zone = connected_components(G, directed=False)
    n_events = np.bincount(ev_zone)
    counts = sp.csr_matrix((np.ones(n_ev), (ev_zone, np.arange(n_ev)))) @ F   # (zone, pixel): events covering it
    best = counts.argmax(0).A1
    zmap = np.where((counts.getnnz(0) > 0) & (n_events[best] >= min_events), best + 1, 0).reshape(shape)
    area = np.bincount(zmap.ravel(), minlength=n_z + 1)
    zmap[area[zmap] < min_area] = 0
    return zmap, n_events, ev_zone


def zone_traces(act, zmap, ids):
    """Mean of act over each zone, streamed over the session: (T, zones)."""
    px = np.flatnonzero(zmap)
    z = np.searchsorted(ids, zmap.ravel()[px])                  # row of each pixel's zone
    Zm = sp.csr_matrix(((1 / np.bincount(z)[z]).astype(np.float32), (z, px)), shape=(len(ids), zmap.size))
    out = np.zeros((len(act), len(ids)), np.float32)
    for t0 in range(0, len(act), CHUNK):
        out[t0:t0 + CHUNK] = np.asarray(act[t0:t0 + CHUNK]).reshape(-1, zmap.size) @ Zm.T
    return out


# ---------------------------------------------------------------- comparisons

def containment(zmap, ids, masks, fov, frac=0.8, n_null=200, cells=1, seed=0):
    """Space: a zone is found if >= frac of its pixels lie inside a single reference mask (one cell); a zone
    straddling two cells or partly outside every cell is a miss. Precision = zones found / zones; recall = reference
    cells with at least one zone found inside them / cells.
    Chance: each zone's shape is rotated by a random angle and moved to a random position drawn from the local
    density of reference cells (stacked masks blurred by a Gaussian of sigma = `cells` median cell diameters), inside
    the lens circle, so random zones land where cells are as often as mine do, in any orientation.
    Returns the placement probability map as `density`."""
    H, W = zmap.shape
    M = sp.csr_matrix(masks, dtype=np.float32)                  # (pixels, cells)
    K = M.shape[1]
    diam = 2 * np.sqrt(np.median(M.getnnz(0)) / np.pi)        # median cell diameter, px
    density = ndimage.gaussian_filter(np.asarray(M.sum(1)).reshape(H, W), cells * diam) * fov
    p = (density / density.sum()).ravel()
    shapes = []                                                 # each zone's pixels relative to its centre
    for i in ids:
        yy, xx = np.nonzero(zmap == i)
        cy, cx = int(round(yy.mean())), int(round(xx.mean()))
        shapes.append((yy - cy, xx - cx, cy, cx))

    def rotate(dy, dx, theta):
        """Zone (pixel offsets dy, dx from its centre) rotated by theta, nearest-neighbour like ndimage.rotate(order=0):
        each output pixel takes the value at its inverse-rotated position, so the area is kept to within a few %."""
        R = int(np.ceil(np.hypot(dy, dx).max())) + 1
        m = np.zeros((2 * R + 1, 2 * R + 1), bool)
        m[dy + R, dx + R] = True                                # the zone in a small square around its centre
        gy, gx = np.mgrid[-R:R + 1, -R:R + 1]
        c, s = np.cos(theta), np.sin(theta)
        sy, sx = np.round(c * gy + s * gx).astype(int), np.round(-s * gy + c * gx).astype(int)
        keep = m[np.clip(sy + R, 0, 2 * R), np.clip(sx + R, 0, 2 * R)] & (abs(sy) <= R) & (abs(sx) <= R)
        return gy[keep], gx[keep]

    def score(centres, angles):
        cell = np.full(len(shapes), -1)                         # cell containing each zone, -1 = miss
        for z, ((dy, dx, _, _), (cy, cx), a) in enumerate(zip(shapes, centres, angles)):
            ry, rx = rotate(dy, dx, a) if a else (dy, dx)
            yy, xx = ry + cy, rx + cx
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)   # pixels off the frame count as outside
            cnt = np.asarray(M[yy[ok] * W + xx[ok]].sum(0)).ravel()
            if cnt.max() >= frac * len(yy):
                cell[z] = cnt.argmax()
        return cell

    cell = score([(cy, cx) for _, _, cy, cx in shapes], np.zeros(len(shapes)))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        c = score(np.c_[np.unravel_index(rng.choice(H * W, len(shapes), p=p), (H, W))], rng.uniform(0, 2 * np.pi, len(shapes)))
        null.append(((c >= 0).mean(), len(np.unique(c[c >= 0])) / K))
    null = np.array(null)
    n_found, n_cells = (cell >= 0).sum(), len(np.unique(cell[cell >= 0]))
    return dict(cell=cell, n_zones=len(shapes), n_found=n_found, n_ref=K, n_cells_found=n_cells,
                prec=n_found / len(shapes), rec=n_cells / K, prec_chance=null[:, 0].mean(), rec_chance=null[:, 1].mean(),
                diam=diam, cells=cells, sigma=cells * diam, density=p.reshape(H, W), null=null)   # null: (n_null, 2) draws


def print_containment(c, ref="CaImAn"):
    n, f, K, k = c["n_zones"], c["n_found"], c["n_ref"], c["n_cells_found"]
    print(f"space, zones >= 80% inside a single {ref} cell (chance: zones rotated and moved at random, weighted by the density of "
          f"{ref} cells, Gaussian sigma {c['sigma']:.0f} px = {c['cells']} cell diameter{'s' if c['cells'] != 1 else ''}):")
    print(f"  inside one cell:           {f}/{n} zones ({f / n:.0%}, chance {c['prec_chance']:.0%})")
    print(f"  straddling or outside:     {n - f}/{n} zones ({(n - f) / n:.0%})")
    print(f"  {ref} cells with no zone: {K - k}/{K} ({(K - k) / K:.0%}); with one: {k}/{K} ({k / K:.0%}, chance {c['rec_chance']:.0%})")


# ---------------------------------------------------------------- figures

def _contours(ax, masks, shape, idx=None, **kw):
    M = masks.toarray().reshape(*shape, -1)
    for k in (range(M.shape[2]) if idx is None else idx):
        cs = ax.contour(M[..., k], [0.5], **kw)
    return cs


def _crop(ax, masks, shape, pad=10):
    ys, xs = np.nonzero(np.asarray(masks.sum(1)).reshape(shape))
    ax.set_xlim(xs.min() - pad, xs.max() + pad); ax.set_ylim(ys.max() + pad, ys.min() - pad); ax.set_axis_off()


def chance_table(c, ref="CaImAn"):
    """My detector vs chance, as in the notebook: zones inside a single reference cell, and reference cells with a zone."""
    import pandas as pd
    n, f, K, k, pc, rc = c["n_zones"], c["n_found"], c["n_ref"], c["n_cells_found"], c["prec_chance"], c["rec_chance"]
    return pd.DataFrame({
        "my pipeline": [f"{f}/{n} ({f / n:.0%})", f"{n - f}/{n} ({(n - f) / n:.0%})",
                        f"{k}/{K} ({k / K:.0%})", f"{K - k}/{K} ({(K - k) / K:.0%})"],
        "chance":      [f"{pc * n:.1f}/{n} ({pc:.0%})", f"{(1 - pc) * n:.1f}/{n} ({1 - pc:.0%})",
                        f"{rc * K:.1f}/{K} ({rc:.0%})", f"{(1 - rc) * K:.1f}/{K} ({1 - rc:.0%})"],
    }, index=[f"zones >= 80% inside a single {ref} cell", "zones straddling two cells or outside",
              f"{ref} cells with one of my zones inside", f"{ref} cells with none of my zones"])


def placement_figure(zmap, au, sc, title=""):
    """The chance level's placement map: stacked reference masks, and the placement probability built from them."""
    import matplotlib.pyplot as plt
    shape = au["shape"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im0 = ax[0].imshow(np.asarray(au["masks"].sum(1)).reshape(shape), cmap='magma', interpolation='nearest')
    fig.colorbar(im0, ax=ax[0], fraction=0.046, label="authors' cells covering the pixel")
    ax[0].set_title(f"{title}stacked masks of the authors' accepted cells")
    im1 = ax[1].imshow(sc["density"], cmap='magma')
    fig.colorbar(im1, ax=ax[1], fraction=0.046, label="placement probability per pixel")
    ax[1].set_title(f"placement probability, sigma {sc['sigma']:.0f} px")
    for a in ax:
        _contours(a, au["masks"], shape, colors='w', linewidths=0.4, alpha=0.7)
        a.contour(zmap > 0, [0.5], colors='red', linewidths=0.8)
        a.set_axis_off()
    fig.tight_layout()
    return fig


def traces_figure(peak, zmap, ids, n_events, traces, ev_zone, t_first, au, cell, r, ts, n_top=8, title=""):
    """Same display as the notebook's chunk: top, the n_top most active zones inside a single authors' cell (one colour
    each) and that cell's contour (same colour); bottom, my zone trace (colour) vs the cell's trace C (grey)."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import ListedColormap
    shape = au["shape"]
    found = np.flatnonzero(cell >= 0)
    top = found[np.argsort(-n_events[ids[found] - 1])[:n_top]]
    col = ['#ff3b3b', '#ff9f1c', '#00c9a7', '#3ddc4b', '#00d5ff', '#4d7cff', '#c86bff', '#ff5ec8'][:len(top)]
    halo = [pe.withStroke(linewidth=3, foreground='black')]

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 18), height_ratios=[1, 0.9])
    ax.imshow(peak ** .5, cmap='Greys_r', vmin=0, vmax=(peak ** .5).max() / 0.6)   # dimmed, so the colours stand out
    ax.imshow(np.ma.masked_equal(zmap > 0, 0), cmap='Greys', vmin=0, vmax=2, alpha=0.6, interpolation='nearest')
    _contours(ax, au["masks"], shape, colors='w', linewidths=0.5, alpha=0.5)
    for i, z in enumerate(top):
        ax.imshow(np.ma.masked_equal(zmap == ids[z], 0), cmap=ListedColormap([col[i]]), interpolation='nearest')
        _contours(ax, au["masks"], shape, idx=[cell[z]], colors=[col[i]], linewidths=2).set(path_effects=halo)
        y, x = ndimage.center_of_mass(zmap == ids[z])
        ax.text(x + 5, y - 5, str(ids[z]), color=col[i], fontsize=13, fontweight='bold', path_effects=halo)
    _crop(ax, au["masks"], shape)
    ax.set_title(f"{title}my {len(ids)} zones (grey; {len(top)} most active found, in colour) and the authors' "
                 f"{au['masks'].shape[1]} accepted cells (white; containing cell in colour)")

    T = len(traces)
    t = ts[:T] if len(ts) >= T else np.arange(T) / FPS
    for i, z in enumerate(top):
        c = au["C"][cell[z]]
        bx.plot(t[:len(c)], c / c.max() + 1.3 * i, color='0.6', lw=0.6)
        bx.plot(t, traces[:, z] / traces[N_STARTUP:, z].max() + 1.3 * i, color=col[i], lw=0.7)
        on = t_first[ev_zone == ids[z] - 1]                     # streamed rise[t] is already the step t-1 -> t
        bx.plot(t[on], np.full(len(on), 1.3 * i - 0.15), '|', color=col[i], ms=6)
    bx.set_yticks(1.3 * np.arange(len(top)),
                  [f"zone {ids[z]}\n(cell {au['idx'][cell[z]]}, r {r[z]:.2f})" for z in top], fontsize=8)
    for lab, c in zip(bx.get_yticklabels(), col):
        lab.set_color(c)
    bx.set_xlabel("time (s)"); bx.set_xlim(t[0], t[-1])
    bx.set_title("my zone trace from act (colour), trace C of the authors' cell containing it (grey), my event onsets (ticks)",
                 fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- everything

def compare_with_authors(session_dir, nwb_path, iou=0.5, min_events=2, min_area=9, title=""):
    """Whole session vs the authors' accepted cells, in space (containment against chance) and in the traces."""
    session_dir = Path(session_dir)
    act = np.load(session_dir / "act.npy", mmap_mode="r")
    rise = np.load(session_dir / "rise.npy", mmap_mode="r")
    shape = act.shape[1:]
    fov = fov_mask(shape)

    peak = peak_map(act)
    au = load_authors(nwb_path, shape)
    dy, dx = grid_offset(peak, au["A"], shape)
    print(f"authors' extraction: {au['masks'].shape[1]} accepted cells; grid offset to my map: dy={dy}, dx={dx} px")
    if (dy, dx) != (0, 0):
        au = shift_footprints(au, dy, dx)

    F, t_first = session_events(rise)
    zmap, n_events, ev_zone = group_zones(F, shape, iou, min_events, min_area)
    ids = np.unique(zmap[zmap > 0])
    traces = zone_traces(act, zmap, ids)
    print(f"{F.shape[0]} rise events -> {len(ids)} zones seen at least {min_events} times, >= {min_area} px "
          f"(median area {np.median(np.bincount(zmap.ravel())[ids]):.0f} px)")

    # space: zones inside a single accepted cell, against chance
    sc = containment(zmap, ids, au["masks"], fov)
    print_containment(sc, "authors'")

    # where the misses go: zones >= 80% inside a single component the authors rejected
    rej = load_authors(nwb_path, shape, which="rejected")
    if (dy, dx) != (0, 0):
        rej = shift_footprints(rej, dy, dx)
    Zb = sp.csr_matrix((zmap == ids[:, None, None]).reshape(len(ids), -1).astype(np.float32))
    in_rej = (Zb @ rej["masks"]).toarray().max(1) / np.asarray(Zb.sum(1)).ravel() >= 0.8
    miss = sc["cell"] < 0
    print(f"  of the {miss.sum()} zones not inside an accepted cell, {(miss & in_rej).sum()} are >= 80% inside a single "
          f"component the authors rejected ({rej['masks'].shape[1]} rejected components)")

    # traces: each found zone vs the authors' cell containing it
    cell = sc["cell"]
    found = np.flatnonzero(cell >= 0)
    r = np.full(len(ids), np.nan)
    r[found] = [np.corrcoef(traces[N_STARTUP:au["C"].shape[1], z], au["C"][cell[z], N_STARTUP:])[0, 1] for z in found]
    print(f"{len(found)} zones inside a single authors' cell: trace correlation median {np.nanmedian(r):.2f} "
          f"(range {np.nanmin(r):.2f} to {np.nanmax(r):.2f})")

    import pynwb
    with pynwb.NWBHDF5IO(str(Path(nwb_path).with_name(Path(nwb_path).name.replace("_behavior+image+ophys", "_ophys"))), "r") as io:
        ts = np.asarray(io.read().acquisition["OnePhotonSeries"].timestamps[:])
    figs = [placement_figure(zmap, au, sc, title),
            traces_figure(peak, zmap, ids, n_events, traces, ev_zone, t_first, au, cell, r, ts, title=title)]
    return dict(peak=peak, zmap=zmap, ids=ids, n_events=n_events, traces=traces, ev_zone=ev_zone, t_first=t_first,
                authors=au, containment=sc, cell=cell, trace_corr=r, inside_rejected=in_rej, table=chance_table(sc, "authors'"),
                offset=(dy, dx), figs=figs)
