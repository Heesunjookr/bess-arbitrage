"""Render the forecast-tier capture table as a static PNG.

Run:  python -m src.viz.plot_forecast_tier_table
"""

import os
from pathlib import Path
from tempfile import gettempdir

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ROOT, get_logger, load_config

logger = get_logger("plot_forecast_tier_table")


def _outdir() -> Path:
    cfg = load_config()
    outdir = ROOT / cfg["paths"]["processed"]["dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def plot_forecast_tier_table(out_path: Path) -> None:
    columns = ["Tier", "Information set", "Capture", "Gap recovered"]
    rows = [
        ["persist_d1", "D-1 price curve", "82.3%", "baseline"],
        ["blend", "D-1 + D-7 curves, rolling OLS", "86.4%", "23%"],
        ["gbm", "everything below + calendar, gradient boosting", "88.3%", "34%"],
        ["rl_quad", "residual-load forecast only (RL + RL²)", "92.5%", "58%"],
        ["blend_rl", "D-1 + D-7 + RL + RL², rolling OLS", "94.5%", "69%"],
    ]

    fig, ax = plt.subplots(figsize=(12, 3.55), dpi=200)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(
        "Forecast Tier Performance",
        fontsize=17,
        fontweight="bold",
        color="#111827",
        pad=11,
    )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colWidths=[0.17, 0.51, 0.15, 0.17],
        cellLoc="left",
        loc="center",
        bbox=[0, 0.03, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        cell.set_linewidth(0.8)

        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.17)
        else:
            cell.set_facecolor("#f9fafb" if row % 2 else "white")
            cell.set_text_props(color="#111827")
            cell.set_height(0.145)

        if col >= 2:
            cell.set_text_props(ha="right")

    # Preserve the emphasis from the Markdown source.
    for col in (1,):
        table[(4, col)].set_text_props(weight="bold")
    for col in (2, 3):
        table[(4, col)].set_text_props(weight="bold", ha="right")
        table[(5, col)].set_text_props(weight="bold", ha="right", color="#166534")
    table[(5, 0)].set_text_props(weight="bold", color="#166534")
    table[(5, 1)].set_text_props(weight="bold", color="#166534")
    for col in range(4):
        table[(5, col)].set_facecolor("#dcfce7")

    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    out_path = _outdir() / "forecast_tier_table.png"
    plot_forecast_tier_table(out_path)
    logger.info("saved %s", out_path.name)


if __name__ == "__main__":
    main()
