# miniscope_ca1

Sandbox for running CaImAn (CNMF-E) on raw one-photon miniscope data, and looking at neuron–neuron correlations before and after background handling. Started 2026-11-19 for the ESPCI C4 (Vetere lab) application.

## Data

DANDI 002047, "Hippocampal CA1 one-photon calcium imaging and pose-tracked behavior in wild-type and 5xFAD mice across four behavioral paradigms" (Plusnin, Rogozhnikova, Sotskov, Pospelov, Ivanova, Savelev et al.). Licence CC-BY-4.0. https://dandiarchive.org/dandiset/002047

- Dorsal CA1, GCaMP6s, GRIN lens, UCLA Miniscope V4.4, ~30 fps.
- 47 mice, 201 sessions: open field variants (OF, MOF, EOF) and a contextual fear task, context pre-exposure facilitation effect (CPFE).
- Each session ships every processing stage in NWB: **raw miniscope movies** (AVI chunks), source-extraction parameters, spatial footprints and traces of retained and rejected components, curation decisions, calcium events, cross-session matches, plus behaviour video and DeepLabCut poses.

The published footprints and traces are a reference to compare a fresh CNMF-E run against.

## Session pulled by default

`sub-F06/sub-F06_ses-CPFE-F06-S3` (fear task, session 3), ~1.2 GB:

- `sub-F06_ses-CPFE-F06-S3_ophys/*.avi` — raw miniscope movie, 9 chunks
- `sub-F06_ses-CPFE-F06-S3_behavior+image+ophys.nwb` — processed results and behaviour
- behaviour video (.wmv / .mp4)

`python3 download.py` fetches it into `data/` (git-ignored). Another session: `python3 download.py "sub-F06/sub-F06_ses-CPFE-F06-S1*"`. Check the CPFE session numbering (which day is pre-exposure, shock, test) in the dataset description or paper before interpreting.

## Environment

`uv sync` builds `.venv/` (Python 3.12): CaImAn 1.13.2 (from the GitHub tag; the PyPI `caiman` is an unrelated package), pynwb, JupyterLab. Run `uv run jupyter lab`, or point your editor at `.venv/bin/python`.

## Running it

`example_clean.ipynb` is the notebook. Its outputs are saved, so it reads without running. To rerun it from scratch: `uv sync`, then `uv run jupyter lab` and Run All. The first run downloads the session (~1 GB into `data/`), runs motion correction and CaImAn on the first chunk, and streams the whole session into `out/session` (~3.5 GB of memory-mapped arrays; later runs reuse it). CaImAn writes ~2 GB of temporary files to `~/caiman_data/temp`.

- `streaming.py`: the detection chain over the whole session, chunk by chunk.
- `fullsession.py`: whole-session comparison with the authors' published extraction.
- `chance_test.py`: the chance test with 10,000 random draws (`python chance_test.py chunk` for the first chunk; `python chance_test.py` for full sessions S3 and S1, which needs S1 downloaded and streamed first, as the script explains).
