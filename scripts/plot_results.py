"""Render static result figures from the saved report analysis JSON.

Requires Matplotlib. Does not load the estimator, refit, or modify prediction files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bikeflow-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#B4BDC8", "#E7A340", "#177F8A"]


def main() -> None:
    result = json.loads((ROOT / "reports/full_run_analysis.json").read_text())
    output = ROOT / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    keys = ["zero_baseline", "seasonal_baseline", "model"]
    labels = ["Always zero", "Seasonal mean", "Poisson boosting"]
    for axis, metric in zip(axes, ["mae", "rmse"], strict=True):
        values = [result["overall"][key][metric] for key in keys]
        bars = axis.bar(labels, values, color=COLORS, width=0.62)
        axis.bar_label(bars, fmt="%.4f", padding=5)
        axis.set_ylim(0, max(values) * 1.22)
        axis.set_title(f"{metric.upper()}: lower is better")
        axis.set_ylabel("Trips per station per 30 minutes")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=9)
    fig.suptitle("Full run: 150 stations, April–June 2026", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "92.84% of test windows contain zero departures. MAE and RMSE favor different forecasts.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(output / "model_comparison.png", dpi=180)
    plt.close(fig)

    daily = pd.DataFrame(result["daily_network_series"])
    dates = pd.to_datetime(daily["date"])
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(dates, daily["actual"], color="#172B4D", label="Observed", linewidth=1.8)
    axis.plot(dates, daily["model"], color=COLORS[2], label="Poisson boosting", linewidth=1.6)
    axis.plot(
        dates, daily["seasonal"], color=COLORS[1], label="Seasonal mean", linewidth=1.2, alpha=0.85
    )
    axis.set_title("Daily sums of sequential 30-minute forecasts")
    axis.set_ylabel("Departures across evaluated station-windows")
    axis.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.legend(loc="upper left", ncol=3, frameon=False)
    axis.grid(axis="y", alpha=0.15)
    fig.text(
        0.5,
        0.015,
        "Historical rolling forecasts summed after the day; this is not a day-ahead forecast.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(output / "daily_network_forecasts.png", dpi=180)
    plt.close(fig)

    patterns = result["operational_patterns"]["station_windows"]
    ids = [3014, 3030, 4250, 4387]
    names = ["Union Station West Portal", "Main & 1st", "Severance & Adams", "Row DTLA"]
    fig, axis = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(ids))
    for offset, window, label, color in [
        (-0.18, "07:00-10:00", "Weekdays 7–10 AM", "#177F8A"),
        (0.18, "16:00-19:00", "Weekdays 4–7 PM", "#D87E35"),
    ]:
        changes = [
            -next(
                row["net_loss_per_day"]
                for row in patterns
                if row["year"] == 2026
                and row["station_id"] == station_id
                and row["day_type"] == "Weekday"
                and row["window"] == window
            )
            for station_id in ids
        ]
        bars = axis.barh(y + offset, changes, height=0.32, label=label, color=color)
        axis.bar_label(bars, fmt="%+.2f", padding=4, fontsize=10)
    axis.set_yticks(y, names)
    axis.invert_yaxis()
    axis.axvline(0, color="#637487", linewidth=1)
    axis.set_xlim(-6.2, 5.1)
    axis.set_xlabel("Average customer-trip net change: arrivals minus departures (bikes)")
    axis.set_title("Station flow reversals: April–June 2026")
    axis.grid(axis="x", alpha=0.15)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=10)
    fig.text(
        0.5,
        0.015,
        "Negative = bikes leave; positive = bikes accumulate. "
        "Excludes operator transfers and inventory.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(output / "station_flow_patterns.png", dpi=180)
    plt.close(fig)
    print(f"Saved three figures to {output}")


if __name__ == "__main__":
    main()
