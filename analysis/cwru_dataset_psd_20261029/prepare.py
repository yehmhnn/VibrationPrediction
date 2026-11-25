"Metadata verification and redraw of saved Welch spectra; no raw/test signal access."

import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(__import__("tempfile").gettempdir(), "vibration_analysis_mpl"),
)
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = Path(__file__).resolve().parent
R = O.parents[1]
A = O / "figures"
V = R / "analysis/evidence/VibrationPrediction_Report_Revised"
E = R / "analysis/evidence/VibrationPrediction_Report_TwoStudies_20261016"
W = R / "analysis/studyII_welch_20261028"
sources = {}


def track(p):
    sources[str(p.relative_to(R))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def rows(p):
    return list(csv.DictReader(track(p).open()))


def load(p):
    return json.loads(track(p).read_text())


def write(name, rr):
    with (O / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0]))
        w.writeheader()
        w.writerows(rr)


def table(h, rr):
    return (
        "| "
        + " | ".join(h)
        + " |\n|"
        + "|".join(["---"] * len(h))
        + "|\n"
        + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rr)
        + "\n"
    )


sm = load(R / "evidence/datasets/cwru_105_0_chronological/metadata.json")
sr = rows(R / "evidence/datasets/cwru_105_0_chronological/manifest.csv")
mr = rows(R / "evidence/datasets/cwru_multi_record_B/manifest.csv")
mm = load(R / "evidence/datasets/cwru_multi_record_B/metadata.json")
norm = load(R / "evidence/datasets/cwru_multi_record_B/normalization.json")
N = next(int(r["num_samples"]) for r in mr if r["record_id"] == "105_0")
retained = sum(int(r["num_samples"]) for r in sr)
assert N - retained == 2400
split = []
for r in sr:
    n = int(r["num_samples"])
    stride = 32 if r["split"] == "train" else 96
    split.append(
        dict(
            partition=r["split"],
            start=int(r["source_start"]),
            end_exclusive=int(r["source_end_exclusive"]),
            raw_samples=n,
            duration_s=n / 12000,
            fraction_original=n / N,
            fraction_retained=n / retained,
            L=512,
            H=96,
            stride=stride,
            windows=max(0, (n - 608) // stride + 1),
        )
    )
write("studyI_split.csv", split)
records = []
for r in mr:
    s = r["split"]
    n = int(r["num_samples"])
    stride = 32 if s == "train" else 96
    assert r["record_id"] in mm["split_record_ids"][s]
    records.append(
        dict(
            record_id=r["record_id"],
            filename=Path(r["source_path"]).name,
            fault=r["fault_type"],
            load_hp=int(r["load_id"]),
            rpm=int(r["rpm"]),
            fs_hz=int(r["sampling_rate_hz"]),
            samples=n,
            duration_s=n / 12000,
            stride=stride,
            windows=(n - 608) // stride + 1,
            assignment="reserved" if s == "test" else s,
            verification="manifest metadata; reserved signal not reopened"
            if s == "test"
            else "manifest and prior hash-verified MAT analysis",
        )
    )
write("studyII_recordings.csv", records)
for model in ["tsflow", "timesnet"]:
    run = f"multi_record_B_{model}_objective_v1"
    c = load(R / "configs/study_ii" / run / "config.json")
    res = load(R / "evidence/experiments/study_ii" / run / "result.json")
    assert not res["test_evaluated"]
    assert c["num_origins"]["train"] == sum(
        r["windows"] for r in records if r["assignment"] == "train"
    )
    assert c["num_origins"]["validation"] == sum(
        r["windows"] for r in records if r["assignment"] == "validation"
    )
for p in [
    V / "source_artifacts/scripts/preprocess_cwru_within_record.py",
    V / "source_artifacts/scripts/sequence_data.py",
    R / "scripts/tsflow_data.py",
    E / "evidence/scripts/run_convergence_experiment.py",
]:
    track(p)

for model in ["tsflow", "timesnet"]:
    c = load(R / f"configs/study_i/single_105_0_{model}_objective_v2/config.json")
    assert c["num_origins"] == {"train": 2210, "validation": 242, "test": 242}
prov = load(V / "data/reconstructed/inference_provenance.json")
assert prov["num_origins"] == 242 and prov["eval_stride"] == 96
ps = rows(W / "welch_spectra.csv")
summary = rows(W / "record_summary.csv")
method = load(W / "provenance_and_checks.json")
assert method["source_hashes_all_match"]
assert set(r["record_id"] for r in ps) == {
    r["record_id"] for r in records if r["assignment"] != "reserved"
}
S = {
    rid: (
        np.array([float(v["frequency_hz"]) for v in ps if v["record_id"] == rid]),
        np.array(
            [
                float(v["psd_original_units2_per_hz"])
                for v in ps
                if v["record_id"] == rid
            ]
        ),
    )
    for rid in set(r["record_id"] for r in ps)
}
plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
colors = ["#0072B2", "#009E73", "#D55E00"]
faults = [
    ("inner_race_fault", "Inner race"),
    ("ball_fault", "Ball"),
    ("outer_race_fault", "Outer race (@6)"),
]


def finish(fig, name, note):
    fig.text(0.5, 0.018, note, ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(A / (name + ".pdf"))
    fig.savefig(A / (name + ".png"), dpi=240)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(12.8, 6))
for (fault, label), color in zip(faults, colors):
    r = next(r for r in records if r["fault"] == fault and r["load_hp"] == 0)
    f, p = S[r["record_id"]]
    ax.semilogy(
        f, p, color=color, lw=1.1, label=f"{label}: {r['record_id']}, {r['rpm']} RPM"
    )
ax.axvspan(2000, 4000, color="#bfc7cd", alpha=0.18, zorder=0)
ax.set(
    xlim=(0, 6000),
    xlabel="Frequency (Hz)",
    ylabel="PSD (source acceleration units²/Hz; log scale)",
    title="Fault-type comparison at 0 hp · 12 kHz drive-end recordings",
)
ax.grid(alpha=0.17)
ax.legend(loc="lower left")
finish(
    fig,
    "cwru_fault_psd",
    (
        "Post-report exploratory Welch PSD · Individual recordings, not fault-class "
        "averages · Absolute power; no variance normalization"
    ),
)
fig, axes = plt.subplots(1, 3, figsize=(15, 5.7), sharex=True, sharey=True)
for ax, (fault, label) in zip(axes, faults):
    for r in [
        r for r in records if r["fault"] == fault and r["assignment"] != "reserved"
    ]:
        f, p = S[r["record_id"]]
        l = r["load_hp"]
        ax.semilogy(
            f,
            p,
            color=colors[l],
            lw=0.9,
            ls="--" if l == 1 else "-",
            label=f"{l} hp · {r['rpm']} RPM\n{r['record_id']}",
        )
    ax.axvspan(2000, 4000, color="#bfc7cd", alpha=0.18, zorder=0)
    ax.set(
        xlim=(0, 6000),
        xlabel="Frequency (Hz)",
        xticks=[0, 2000, 4000, 6000],
        title=label,
    )
    ax.grid(alpha=0.17)
    ax.legend(loc="lower left", fontsize=10)
axes[0].set_ylabel("PSD (source acceleration units²/Hz; log scale)")
finish(
    fig,
    "cwru_load_psd",
    (
        "Post-report exploration · 0/1 hp: training; 2 hp: validation · Common log "
        "scale, absolute PSD · No 3 hp signals"
    ),
)
si = table(
    [
        "Partition",
        "Indices [start,end)",
        "Raw samples",
        "Duration (s)",
        "% original",
        "% retained",
        "Stride",
        "Windows",
    ],
    [
        [
            r["partition"],
            f"[{r['start']},{r['end_exclusive']})",
            r["raw_samples"],
            f"{r['duration_s']:.9f}",
            f"{r['fraction_original'] * 100:.9f}",
            f"{r['fraction_retained'] * 100:.0f}",
            r["stride"],
            r["windows"],
        ]
        for r in split
    ],
)
sii = table(
    [
        "MAT filename / ID",
        "Fault",
        "hp",
        "RPM",
        "Hz",
        "Raw samples",
        "Duration (s)",
        "Windows",
        "Role",
    ],
    [
        [
            r["filename"],
            r["fault"].replace("_fault", "").replace("_", " "),
            r["load_hp"],
            r["rpm"],
            r["fs_hz"],
            r["samples"],
            f"{r['duration_s']:.6f}",
            r["windows"],
            r["assignment"],
        ]
        for r in records
    ],
)
report = f"""# Verified CWRU dataset statistics and spectral assets

2026-10-29. No training, inference, raw recording access or reserved-signal access was performed. The figures redraw existing verified PSD values. All statistics distinguish raw samples from windows. Reserved-record statistics are verified against manifest/code metadata, not newly measured from reserved arrays.

## Study I: 105_0 chronological split

Original N={N} samples, duration N/fs={N / 12000:.9f} s at12000Hz (sample support convention N/fs, not (N−1)/fs). Raw count is recorded in the StudyII manifest and was previously checked in the allowed 105_0 MAT analysis. Retained samples={retained}; discarded gap samples=2400 ({2400 / N * 100:.9f}% of original). The retained-data fractions are exactly60/20/20. Relative to the entire original recording, use the distinct percentages below.

{si}
Two gaps: [71319,72519) and [96292,97492), each1200 samples/0.1s. Not random sample splitting. For chronological preprocessing, available=N−2gap; train=int(0.6available); validation=(available−train)//2; test=remainder.

Primary L512/H96 spans42.666667ms context and8ms forecast. Window count=max(0,floor((n−L−H)/stride)+1). Window starts are partition-local0,stride,...; no window crosses a partition or recording boundary. Contexts overlap; nonoverlapping evaluation target blocks do not make errors statistically independent.

The242 common test origins are **all** valid H96 stride96 starts: context start=97492+96j and forecast start=98004+96j, j=0,...,241. Last context start120628; last forecast start121140; final target interval[121140,121236), leaving29 trailing samples. No subsampling was used to create242; the same grid applies to both persisted model outputs. Waveform example61 is a separately chosen visualization, not the grid definition. Full H96 metric uses242×96=23232 target elements.

Historical independently trained horizons use evaluation strideH and have different window counts; the242 claim is specifically H96/fixed-H96 prefixes. From the same count formula:

"""
report += table(
    [
        "H",
        "Training stride32 windows",
        "Validation strideH windows",
        "Test strideH windows",
    ],
    [
        [
            h,
            (71319 - 512 - h) // 32 + 1,
            (23773 - 512 - h) // h + 1,
            (23773 - 512 - h) // h + 1,
        ]
        for h in [24, 48, 72, 96]
    ],
)
report += f"""
Normalization: (x−mean)/std, fitted only to raw chronological training samples[0,71319), float64 mean/population SD(ddof0), stored float32. Mean={sm["train_mean"]}, SD={sm["train_std"]}. It is not the global StudyII normalization. TSFlow adapter uses identity scaling on normalized arrays; TimesNet additionally applies past-context internal normalization/de-normalization as part of its architecture.

Evidence:
- `evidence/datasets/cwru_105_0_chronological/metadata.json` and `manifest.csv`: exact intervals, gap, normalizer, partition counts.
- `{V.relative_to(R)}/source_artifacts/scripts/preprocess_cwru_within_record.py`: chronological_boundaries and mean/std fit.
- `{V.relative_to(R)}/source_artifacts/scripts/sequence_data.py`: ManifestSequenceDataset.window_index, range(0,last_start+1,stride), context/target slicing; `tsflow_data.py` reuses this grid.
- `configs/study_i/single_105_0_{{tsflow,timesnet}}_objective_v2/config.json`: L512,H96,strides32/96,num_origins2210/242/242 (also verified in the previous audit).
- `{V.relative_to(R)}/data/reconstructed/inference_provenance.json`:242 origins, stride96, exact_target_match=true, historical TSFlow100/TimesNet30 checkpoint family. No new historical test evaluation was made.

## Study II: recording-disjoint assignment

{sii}
All listed IDs correspond to same-named MAT files. All have0.007-inch defects; outer-race position is@6. Window counts are computed from each manifest sample count using L512/H96, stride32 for train and96 for validation/reserved. **Reserved window counts are potential loader windows, not evaluated forecasts.** The code and metadata call that split `test`; this report labels it reserved to distinguish assignment from evaluation. Both final original StudyII model result.json files explicitly record test_evaluated=false. Thus the3hp recordings were reserved and not evaluated for those final models; this does not erase any separate historical test inspection under other protocols.

"""
report += table(
    ["Role", "Raw sample total", "Usable windows"],
    [
        [
            s,
            sum(r["samples"] for r in records if r["assignment"] == s),
            sum(r["windows"] for r in records if r["assignment"] == s),
        ]
        for s in ["train", "validation", "reserved"]
    ],
)
report += f"""
StudyII normalization uses six training recordings only, sample-weighted global mean{norm["train_mean"]} and population SD{norm["train_std"]}, from{norm["training_sample_count"]} training samples. Validation and reserved signals do not fit the normalizer. Full records are independently standardized; no cross-record windows are formed.

Evidence:
- `evidence/datasets/cwru_multi_record_B/manifest.csv`: filename,channel,fault,load_id,rpm,sampling_rate_hz,num_samples,source intervals,split. RPM/count derive from MAT checks; fs/load/fault derive from acquisition metadata, not a timestamp field in the MAT.
- Same-directory `metadata.json`: split_record_ids,windowing L512/H96/strides32/96; `normalization.json`: training IDs/sample count, mean/std and uses_validation_or_test=false.
- Server `./scripts/cwru_protocols.py`, read-only inspection: GROUPS exact IDs/counts/RPM and split='train' if load<2 else 'validation' if load==2 else 'test'.
- Server `./scripts/preprocess_cwru_protocol_b.py`, read-only inspection: checks MAT channel/RPM/count against assignments; standardizes full records; reports windows with (n−608)//stride+1. It was inspected, not executed.
- `{E.relative_to(R)}/evidence/scripts/run_convergence_experiment.py`: L512/H96 and loader strides32/96; both `multi_record_B_*_objective_v1/config.json`: matching loader counts; corresponding `result.json`: test_evaluated=false.
- `{W.relative_to(R)}/record_summary.csv` and `provenance_and_checks.json`: prior SHA-256-verified actual measurements for the nine nonreserved MAT files. Reserved counts/RPM were not remeasured in this task; they are metadata-verified.

## hp, RPM, and operating conditions

**hp** means horsepower, the motor-load condition recorded for the test (0hp does not mean zero shaft rotation). **RPM** means revolutions per minute of the motor shaft. Use recording-specific stored values, not the approximate speed associated with a load row. E.g.1hp outer race=1773RPM while1hp inner race/ball=1772RPM;2hp outer race=1750RPM versus1748RPM in the other two. These values are in the manifest and source GROUPS.

Shaft-speed changes can shift rotation-related spectral lines and repetition rates; load can affect bearing excitation and amplitudes. These are possible mechanisms, not a causal finding from this study: load,RPM and acquisition identity change together. Model transfer from0/1hp recordings to2hp is therefore a joint operating-condition/recording change. CWRU describes accelerometer acquisition and recorded speed/load in its [apparatus documentation](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures) and [data-variable guide](https://engineering.case.edu/bearingdatacenter/download-data-file). Physical acceleration calibration in the local artifacts remains unknown; plots retain source numerical units rather than assuming calibrated g.

## Figures and verified observations — post-report exploration

FigureA `assets/cwru_fault_psd.pdf`: individual0hp training recordings105_0(inner race,1797RPM),118_0(ball,1796RPM),130@6_0(outer race,1796RPM). Same nominal load, almost identical but explicitly unequal RPM. This is not an average or a universal fault-class signature.

FigureB `assets/cwru_load_psd.pdf`: one panel/fault, overlaying exactly0/1hp training and2hp validation; actual stored RPM in every legend. Common log-y scale and original-unit PSD preserve absolute power differences; no per-record variance normalization. Both figures span0–6000Hz and lightly shade2–4kHz. No3hp curve is present.

Reused source: `{W.relative_to(R)}/welch_spectra.csv` (psd_original_units2_per_hz), originally computed using fs12000,Hann,nperseg8192,noverlap4096,nfft8192,constant detrend,density scaling,mean averaging,one-sided PSD, no smoothing. Original equivalent `{W.relative_to(R)}/spectral_overview.pdf` was inspected; its data are reused to create the specifically requested layouts, not recomputed or altered. Frequency spacing1.46484375Hz;28 Welch segments/record. Existing band/RMS tables supply the observations below.

### Dataset-slide bullets

- CWRU drive-end vibration is sampled at12kHz;512 past samples predict96 future samples (42.67ms→8ms).
- StudyI uses chronological regions of one105_0 acquisition, separated by100ms gaps.
- StudyII trains on six0/1hp recordings and validates on three2hp recordings across three fault types.
- Three3hp recordings remain reserved; raw sample counts and forecasting-window counts are different quantities.

### Spectral-slide bullets

- Across all nine analyzed recordings, most power lies at2–4kHz: inner race77.4–78.1%, ball90.3–92.8%, outer race97.2–98.0%.
- Peak positions/heights vary with acquisition; similar broad bands do not imply identical spectral shape or forecasting difficulty.
- Outer-race2hp total power is lower than both outer-race training recordings; its high forecasting RMSE is not explained by unusually elevated validation power.

Numerical sources: `{W.relative_to(R)}/band_power.csv` (sum fraction_total for2000–3000 and3000–4000Hz); `record_summary.csv` (integrated Welch power), `dominant_peaks.csv` (peak frequencies). Observations are descriptive; no causal or denoising-improvement claim is made.

### Overleaf inclusion

```latex
\\includegraphics[width=0.98\\linewidth]{{assets/cwru_fault_psd.pdf}}
\\includegraphics[width=0.98\\linewidth]{{assets/cwru_load_psd.pdf}}
```

Use on separate frames and label “Post-report exploratory PSD analysis.” No Beamer slides were generated.

## Outputs and verification

Created: `assets/cwru_fault_psd.pdf`, `assets/cwru_load_psd.pdf`, correspondingPNG previews, plus this analysis directory's `prepare.py`, `REPORT.md`, `studyI_split.csv`, `studyII_recordings.csv`, and `provenance.json`. No original artifact was modified. `prepare.py` reproduces tables and figures solely from metadata and saved PSD CSV values. Source SHA-256 values and plotting versions are in provenance.json. Duration/sample fractions and window counts are derived arithmetically from the cited file fields; reserved signal arrays remain unopened.
"""
(O / "REPORT.md").write_text(report)
(O / "provenance.json").write_text(
    json.dumps(
        dict(
            source_sha256=sources,
            versions=dict(numpy=np.__version__, matplotlib=matplotlib.__version__),
            no_reserved_signal_access=True,
            source_psd_parameters=method["welch"],
            figureA_ids=["105_0", "118_0", "130@6_0"],
            figureB_ids=sorted(S),
            studyI_original_samples=N,
            studyI_retained_samples=retained,
        ),
        indent=2,
    )
    + "\n"
)
print(si)
print(sii)
