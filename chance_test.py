"""Chance level of the zone-in-cell test with many draws, on the full sessions, against the authors' CNMF-E cells.

Loads my zones (from the streamed rise.npy) and the authors' accepted footprints (NWB), then runs
fullsession.containment with n_null draws: each zone rotated at random and placed according to the local density of
the authors' cells (Gaussian, sigma = 1 cell diameter). Prints the null distribution and a one-sided p-value
(share of draws at or above my value, with the +1 correction), and saves a histogram per session.
Written with an AI assistant (Claude Code). Usage, from the repo root: python chance_test.py [n_null],
or python chance_test.py chunk [n_null] for chunk 0 (33 s) against a fresh CaImAn run, as in the notebook
(about 3 min for 10,000 draws on both sessions). Needs data/ (download.py) and out/session, out/session_S1 (streaming).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fullsession
from streaming import fov_mask

# CaImAn's temporary memmaps go inside the repo (out/caiman_temp), not ~/caiman_data/temp
os.makedirs("out/caiman_temp", exist_ok=True)
os.environ.setdefault("CAIMAN_TEMP", os.path.abspath("out/caiman_temp"))

SESSIONS = {
    "S3": ("out/session", "data/sub-F06/sub-F06_ses-CPFE-F06-S3_behavior+image+ophys.nwb"),
    "S1": ("out/session_S1", "data/sub-F06/sub-F06_ses-CPFE-F06-S1_behavior+image+ophys.nwb"),
}


def p_value(null, observed):
    """One-sided: probability that chance does at least as well as observed (+1 so it is never exactly 0)."""
    return ((null >= observed).sum() + 1) / (len(null) + 1)


def check_inputs():
    """The inputs are not in the repo: the data comes from DANDI, the streamed results from the notebooks."""
    missing = [f for d, nwb in SESSIONS.values() for f in (Path(d) / "rise.npy", Path(nwb)) if not f.exists()]
    if missing:
        sys.exit("missing inputs (run from the repo root):\n  " + "\n  ".join(map(str, missing)) +
                 "\n- NWB files: python3 download.py \"sub-F06/sub-F06_ses-CPFE-F06-S3*\" (and S1*)"
                 "\n- out/session*: the whole-session streaming cell of minimal_pipeline.ipynb (S3),"
                 "\n  or streaming.run_session(files, out_dir) on the session's AVI files")


def run(name, session_dir, nwb, n_null, ax):
    rise = np.load(Path(session_dir) / "rise.npy", mmap_mode="r")
    shape = rise.shape[1:]
    au = fullsession.load_authors(nwb, shape)
    zmap, _, _ = fullsession.group_zones(fullsession.session_events(rise)[0], shape)
    ids = np.unique(zmap[zmap > 0])
    t0 = time.perf_counter()
    c = fullsession.containment(zmap, ids, au["masks"], fov_mask(shape), n_null=n_null)
    dt = time.perf_counter() - t0
    print(f"\n{name}: {c['n_zones']} zones, {c['n_ref']} authors' cells, {n_null} draws in {dt:.0f} s "
          f"(sigma {c['sigma']:.0f} px = 1 cell diameter)")
    for j, (label, obs, n) in enumerate([("zones inside one cell", c["prec"], c["n_zones"]),
                                         ("cells with a zone    ", c["rec"], c["n_ref"])]):
        null = c["null"][:, j]
        q = np.percentile(null, [50, 95, 99, 99.9])
        print(f"  {label}: mine {obs:.1%} ({obs * n:.0f}/{n}); chance mean {null.mean():.1%} +- {null.std():.1%}, "
              f"median {q[0]:.1%}, 95% {q[1]:.1%}, 99% {q[2]:.1%}, 99.9% {q[3]:.1%}, max {null.max():.1%}; "
              f"z = {(obs - null.mean()) / null.std():.1f}, p = {p_value(null, obs):.1e}")
        ax[j].hist(null * 100, bins=np.arange(0, 101, 1), color="0.6")
        ax[j].axvline(obs * 100, color="red", lw=2, label=f"mine, {obs:.0%}")
        ax[j].set_title(f"{name}: {label.strip()}")
        ax[j].set_xlabel("%"); ax[j].legend(frameon=False)


def chunk0(n_null):
    """Chunk 0 of S3 (33 s) against a fresh CaImAn CNMF-E run, as in minimal_pipeline.ipynb: motion correction, x2
    downsampling, my chain and zones, CNMF-E with the authors' parameters (~2 min). Reports my value, the chance
    mean and the best of n_null draws for each row of the notebook's table."""
    import caiman as cm
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf import cnmf, params
    from scipy import ndimage
    import streaming
    fname = sorted(Path("data/sub-F06/sub-F06_ses-CPFE-F06-S3_ophys").glob("*.avi"))[0]
    mc = MotionCorrect([str(fname)], max_shifts=(35, 35), gSig_filt=(8, 8), pw_rigid=False, border_nan='copy')
    mc.motion_correct(save_movie=True)
    mms = np.asarray(cm.load(mc.mmap_file[0]).resize(0.5, 0.5, 1))[4:]
    dims = mms.shape[1:]
    fov = fov_mask(dims)
    act = streaming.temporal(streaming.highpass(mms, fov))         # the notebook's chain: high-pass, box filter, top-hat
    rise = streaming.rise_detector(act, streaming.noise_map(act))  # calm-period noise, rise > 0.7 noise, 5 frames, 2x2 px
    del mms
    zmap, _, _ = fullsession.group_zones(fullsession.session_events(rise)[0], dims, min_area=1)   # notebook: no min area
    ids = np.unique(zmap[zmap > 0])

    fname_new = cm.save_memmap([mc.mmap_file[0]], base_name='chance_cnmfe_', order='C', resize_fact=(0.5, 0.5, 1), remove_init=4)
    Yr, dims_c, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims_c), order='F')
    opts = params.CNMFParams(params_dict={
        'fr': 29.78, 'decay_time': 1.0, 'p': 1, 'method_init': 'corr_pnr', 'K': None,
        'gSig': (2, 2), 'gSiz': (9, 9), 'center_psf': True, 'min_corr': 0.9, 'min_pnr': 10,
        'ring_size_factor': 1.5, 'ssub_B': 1, 'nb': 0, 'nb_patch': 0, 'low_rank_background': None,
        'rf': 20, 'stride': 12, 'only_init': True, 'ssub': 1, 'tsub': 1, 'merge_thr': 0.8,
        'normalize_init': False, 'update_background_components': True,
        'method_deconvolution': 'oasis', 'del_duplicates': True, 'border_pix': 0})
    cnm = cnmf.CNMF(n_processes=1, params=opts, dview=None)
    np.random.seed(0)
    cnm.fit(images)
    com_px = np.clip(np.round(cm.base.rois.com(cnm.estimates.A, *dims)).astype(int), 0, np.array(dims) - 1)
    cnm.estimates.select_components(idx_components=np.flatnonzero(fov[com_px[:, 0], com_px[:, 1]]))
    del images, Yr
    Path(fname_new).unlink()
    A = cnm.estimates.A.toarray(); K = A.shape[1]
    masks = (A > 0.2 * A.max(0)).reshape(*dims, K, order='F').reshape(-1, K).astype(np.float32)

    c = fullsession.containment(zmap, ids, masks, fov, n_null=n_null)
    n, f, k = c["n_zones"], c["n_found"], c["n_cells_found"]
    best_f = int(round(c["null"][:, 0].max() * n)); best_k = int(round(c["null"][:, 1].max() * K))
    mean_f, mean_k = c["null"][:, 0].mean(), c["null"][:, 1].mean()
    print(f"\nchunk 0 (33 s): {n} zones, {K} CaImAn cells, {n_null} draws")
    rows = [("zones >= 80% inside a single CaImAn cell", f, n, mean_f * n, best_f),
            ("zones straddling two cells or outside", n - f, n, (1 - mean_f) * n, n - best_f),
            ("CaImAn cells with one of my zones inside", k, K, mean_k * K, best_k),
            ("CaImAn cells with none of my zones", K - k, K, (1 - mean_k) * K, K - best_k)]
    for label, mine, tot, mean, best in rows:
        print(f"  {label:42s} mine {mine}/{tot} ({mine / tot:.0%})   chance {mean:.1f}/{tot} ({mean / tot:.0%})   "
              f"best of {n_null}: {best}/{tot} ({best / tot:.0%})")
    for j, obs in ((0, c["prec"]), (1, c["rec"])):
        null = c["null"][:, j]
        print(f"  {'zones inside' if j == 0 else 'cells found'}: z = {(obs - null.mean()) / null.std():.1f}, p = {p_value(null, obs):.1e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chunk":
        chunk0(int(sys.argv[2]) if len(sys.argv) > 2 else 10_000)
        sys.exit()
    n_null = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    check_inputs()
    fig, axes = plt.subplots(len(SESSIONS), 2, figsize=(11, 3.5 * len(SESSIONS)))
    for ax, (name, (d, nwb)) in zip(axes, SESSIONS.items()):
        run(name, d, nwb, n_null, ax)
    fig.tight_layout()
    Path("out").mkdir(exist_ok=True)
    fig.savefig("out/chance_test.png", dpi=110)
    print("\nhistograms: out/chance_test.png")
