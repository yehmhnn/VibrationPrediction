"Run preserved offline analyses in a fresh copy; never overwrite submitted evidence."

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["welch", "wavelet", "metadata"])
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = a.work_dir.resolve()
    if work.exists():
        raise FileExistsError("Work directory must not exist")
    if work == root or root in work.parents:
        raise ValueError("Use a work directory outside the submitted repository")
    manifest = root / "evidence/datasets/cwru_multi_record_B/manifest.csv"
    rows = [
        r
        for r in csv.DictReader(manifest.open())
        if r["split"] in ["train", "validation"]
    ]
    assert {r["record_id"] for r in rows} == {
        "105_0",
        "106_1",
        "107_2",
        "118_0",
        "119_1",
        "120_2",
        "130@6_0",
        "131@6_1",
        "132@6_2",
    }
    inputs = []
    if a.mode != "metadata":
        if a.input_dir is None:
            p.error("--input-dir is required for signal analyses")
        for r in rows:
            source = a.input_dir / Path(r["source_path"]).name
            if not source.is_file():
                raise FileNotFoundError(source.name)
            if hashlib.sha256(source.read_bytes()).hexdigest() != r["source_sha256"]:
                raise ValueError("Input hash mismatch: " + r["record_id"])
            inputs.append(source.resolve())
    if a.check_only:
        print(
            (
                "PASS: fresh destination, required metadata and authorized input hashes"
                " checked; nothing executed"
            )
        )
        return
    work.mkdir(parents=True, mode=0o700)
    for name in ["analysis", "evidence", "configs"]:
        shutil.copytree(root / name, work / name)
    (work / "scripts").mkdir()
    shutil.copyfile(root / "scripts/tsflow_data.py", work / "scripts/tsflow_data.py")
    target = work / "analysis/studyII_welch_20261028/inputs"
    target.mkdir(exist_ok=True)
    for source in inputs:
        (target / source.name).symlink_to(source)
    scripts = {
        "welch": [
            "studyII_welch_20261028/analyze.py",
            "studyII_welch_20261028/write_report.py",
        ],
        "wavelet": [
            "studyII_wavelet_denoising_20261028/analyze.py",
            "studyII_wavelet_denoising_20261028/write_report.py",
        ],
        "metadata": ["cwru_dataset_psd_20261029/prepare.py"],
    }
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(work / ".mplcache"),
    }
    for script in scripts[a.mode]:
        subprocess.run(
            [sys.executable, "-B", str(work / "analysis" / script)],
            cwd=work,
            env=env,
            check=True,
        )
    print("Completed in fresh work directory; submitted evidence was not modified")


if __name__ == "__main__":
    main()
