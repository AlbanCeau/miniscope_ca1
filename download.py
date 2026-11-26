"""Fetch one session of DANDI 002047 (dorsal CA1, 1-photon miniscope, CC-BY-4.0) into data/.

usage: python3 download.py [glob]   default: sub-F06/sub-F06_ses-CPFE-F06-S3*
or from Python: import download; download.download()
Stdlib only. Skips files already present at the right size."""
import json
import sys
import urllib.request
from pathlib import Path

API = "https://api.dandiarchive.org/api"
DANDISET = "002047"
DEFAULT = "sub-F06/sub-F06_ses-CPFE-F06-S3*"
DEST = Path(__file__).resolve().parent / "data"


def download(glob=DEFAULT, dest=DEST):
    url = f"{API}/dandisets/{DANDISET}/versions/draft/assets/?page_size=200&glob={glob}"
    assets = []
    while url:
        page = json.load(urllib.request.urlopen(url))
        assets += page["results"]
        url = page.get("next")
    print(f"{len(assets)} files, {sum(a['size'] for a in assets) / 1e6:.0f} MB")
    for a in assets:
        out = Path(dest) / a["path"]
        if out.exists() and out.stat().st_size == a["size"]:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        with urllib.request.urlopen(f"{API}/assets/{a['asset_id']}/download/") as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(out)
        print(f"  {a['path']} ({a['size'] / 1e6:.0f} MB)")


if __name__ == "__main__":
    download(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
