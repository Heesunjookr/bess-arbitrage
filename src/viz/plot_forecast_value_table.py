"""
plot_forecast_value_table.py
----------------------------
Render the perfect-forecast value table as a static PNG.

Run:  python -m src.viz.plot_forecast_value_table
"""

import os
from pathlib import Path
from tempfile import gettempdir

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import ROOT, get_logger, load_config

logger = get_logger("plot_forecast_value_table")


def _outdir() -> Path:
    cfg = load_config()
    outdir = ROOT / cfg["paths"]["processed"]["dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _fmt_int(value: int) -> str:
    return f"{value:,.0f}"


def build_table() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            ("2022", 88593, 61732, 0.70, 26862),
            ("2023", 48558, 35151, 0.72, 13407),
            ("2024", 60583, 47317, 0.78, 13266),
            ("2025", 68587, 55523, 0.81, 13065),
            ("2026 H1", 79506, 67439, 0.85, 12067),
        ],
        columns=[
            "Year",
            "Perfect foresight (ceiling)",
            "Persistence D-1 (executable)",
            "Capture (D-1 / PF)",
            "Value of a perfect forecast (PF - D-1)",
        ],
    )
    return df


def plot_forecast_value_table(df: pd.DataFrame, out_path: Path) -> None:
    display = df.copy()
    for col in (
        "Perfect foresight (ceiling)",
        "Persistence D-1 (executable)",
        "Value of a perfect forecast (PF - D-1)",
    ):
        display[col] = display[col].map(_fmt_int)
    display["Capture (D-1 / PF)"] = (display["Capture (D-1 / PF)"] * 100).round(0).astype(int).astype(str) + "%"

    display.columns = [
        "Year",
        "Perfect foresight\n(ceiling)",
        "Persistence D-1\n(executable)",
        "Capture\n(D-1 / PF)",
        "Value of a perfect forecast\n(PF - D-1)",
    ]

    fig, ax = plt.subplots(figsize=(12, 3.35), dpi=200)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(
        "Perfect Forecast Value vs. Executable D-1 Persistence",
        fontsize=17,
        fontweight="bold",
        color="#111827",
        pad=10,
    )

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        colWidths=[0.10, 0.20, 0.22, 0.15, 0.27],
        cellLoc="right",
        loc="center",
        bbox=[0, 0.03, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold", ha="right")
            cell.set_height(0.18)
        else:
            cell.set_facecolor("#f9fafb" if row % 2 else "white")
            cell.set_text_props(color="#111827", ha="right")
            cell.set_height(0.14)
        if col == 0:
            cell.set_text_props(ha="left")

    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    outdir = _outdir()
    plot_forecast_value_table(build_table(), outdir / "forecast_value_table.png")
    logger.info("saved forecast_value_table.png")


if __name__ == "__main__":
    main()
