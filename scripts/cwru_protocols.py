"Approved acquisition metadata; MAT verifies channel, RPM, count and signal integrity."

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = (
    "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data"
)
PROTOCOL_DATASETS = {
    "single_105_0": ROOT / "datasets/processed/cwru_105_0_chronological",
    "multi_record_B": ROOT / "datasets/processed/cwru_multi_record_B",
}
# IDs, counts and stored RPM were checked in the read-only audit.
GROUPS = [
    (
        "inner_race_fault",
        "IR/007",
        ["105_0", "106_1", "107_2", "108_3"],
        [121265, 121991, 122136, 122917],
        [1797, 1772, 1748, 1721],
    ),
    (
        "ball_fault",
        "B/007",
        ["118_0", "119_1", "120_2", "121_3"],
        [122571, 121410, 121556, 121556],
        [1796, 1772, 1748, 1722],
    ),
    (
        "outer_race_fault",
        "OR/007/@6",
        ["130@6_0", "131@6_1", "132@6_2", "133@6_3"],
        [121991, 122426, 121410, 122571],
        [1796, 1773, 1750, 1725],
    ),
]
PROTOCOL_B = []
for fault, folder, ids, counts, rpms in GROUPS:
    for load, (rid, n, rpm) in enumerate(zip(ids, counts, rpms)):
        numeric = rid.split("@")[0].split("_")[0]
        PROTOCOL_B.append(
            dict(
                record_id=rid,
                source_relative_path=f"12k_Drive_End_Bearing_Fault_Data/{folder}/{rid}.mat",
                source_key=f"X{numeric}_DE_time",
                condition=fault,
                fault_diameter_inch=0.007,
                load_id=load,
                rpm=rpm,
                sampling_rate_hz=12000,
                num_samples=n,
                split="train" if load < 2 else "validation" if load == 2 else "test",
                sensor="drive_end_accelerometer",
                outer_race_position="@6"
                if fault == "outer_race_fault"
                else "not_applicable",
            )
        )
