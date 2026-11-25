import csv
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    results_path = root / "logs" / "evaluation_results.csv"
    report_path = root / "reports" / "report.md"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing experiment results: {results_path}")

    rows = []
    with results_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    report_lines = [
        "# Vibration Prediction Report",
        "",
        "## Summary",
        (
            "This report collects the baseline TimesNet results for the CWRU and Ottawa"
            " datasets."
        ),
        "",
        "## Evaluation Results",
        "",
        "| dataset | model | L | H | MAE | RMSE | run_id |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['dataset']} | {row['model']} | {row['L']} | {row['H']} | {row['MAE']} | {row['RMSE']} | {row['run_id']} |"
        )

    report_lines += [
        "",
        "## Figures",
        "- `figures/spectral_screening.png`: dataset spectral comparison",
        "- `figures/gp_complexity_profile.png`: GP prior complexity",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
