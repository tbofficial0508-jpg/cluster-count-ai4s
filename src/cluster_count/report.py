from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd


def metric_table_row(name: str, metrics: dict[str, float]) -> str:
    return f"| {name} | {metrics['mae']:.2f} | {metrics['rmse']:.2f} | {metrics['mape']:.3f} | {metrics['bias']:.2f} |"


def render_failure_section(title: str, rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return []
    lines = [f"### {title}", "", "| Group | MAE | RMSE | MAPE | Bias |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['group']} | {row['mae']:.2f} | {row['rmse']:.2f} | {row['mape']:.3f} | {row['bias']:.2f} |")
    lines.append("")
    return lines


def build_report(metrics: dict[str, object], predictions: pd.DataFrame) -> str:
    lines = [
        "# Validation Report",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        f"- Images evaluated: {metrics['n_images']}",
        f"- Density-model MAE: {metrics['model']['mae']:.2f}",
        f"- Watershed baseline MAE: {metrics['watershed']['mae']:.2f}",
        "",
        "## Method Comparison",
        "",
        "| Method | MAE | RMSE | MAPE | Bias |",
        "| --- | ---: | ---: | ---: | ---: |",
        metric_table_row("Density / ensemble", metrics["model"]),
        metric_table_row("Watershed baseline", metrics["watershed"]),
        "",
        "## Failure Regimes",
        "",
        "These slices are where a non-expert user should expect the most count risk or the strongest need for overlay review.",
        "",
    ]

    for name, rows in metrics.get("failure_regimes", {}).items():
        lines.extend(render_failure_section(name.replace("_", " ").title(), rows))

    hardest = metrics.get("hardest_examples", [])
    if hardest:
        lines.extend(
            [
                "## Hardest Examples",
                "",
                "| Image | Truth | Pred | Baseline | Abs Error | Confidence | Warnings | Overlay |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in hardest:
            overlay_name = Path(row["overlay_path"]).name
            lines.append(
                f"| {row['image_name']} | {row['true_count']:.1f} | {row['pred_count']:.1f} | {row['baseline_count']:.1f} | "
                f"{row['abs_error']:.1f} | {row['confidence']:.2f} | {row['warnings'] or 'none'} | {overlay_name} |"
            )
        lines.append("")

    low_confidence = predictions[predictions["confidence"] < 0.55]
    lines.extend(
        [
            "## Limitations",
            "",
            "- This is a prototype benchmark and validation framework for robust cell counting under clustering and blur, not a clinically validated system.",
            "- Confidence is heuristic and should be interpreted together with overlays, not as a standalone quality certificate.",
            f"- {len(low_confidence)} image(s) fell below the default confidence threshold of 0.55 in this run.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a markdown validation report from evaluation outputs.")
    parser.add_argument("--metrics-json", type=Path, required=True, help="Path to metrics.json.")
    parser.add_argument("--predictions-csv", type=Path, required=True, help="Path to count.csv or predictions CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output markdown path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.predictions_csv)
    report_text = build_report(metrics, predictions)
    args.output.write_text(report_text, encoding="utf-8")
    print(f"Saved markdown report to {args.output}")


if __name__ == "__main__":
    main()

