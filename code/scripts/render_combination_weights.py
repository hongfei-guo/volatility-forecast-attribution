#!/usr/bin/env python3
"""Render monthly QLIKE combination weights."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT
REPLICATION = ROOT
EVAL = REPLICATION / "results" / "evaluation"
TAIL = REPLICATION / "results" / "tail"
FIGURES = ROOT / "reproduced" / "figures"

MARKETS = ["SP500", "FTSE100", "DAX"]
MARKET_LABELS = {"SP500": "S\\&P~500", "FTSE100": "FTSE~100", "DAX": "DAX"}
HORIZONS = [1, 5, 10]


def p_text(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def difference_cell(row: pd.Series) -> str:
    return f"{row['mean_loss_difference']:+.4f} [{p_text(float(row['p_value_two_sided']))}]"


def tail_difference_cell(row: pd.Series) -> str:
    return f"{row['mean_loss_difference']:+.4f} [{p_text(float(row['p_value']))}]"


def load_json_array(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array")
    return [str(item) for item in parsed]


def build_weight_figure() -> None:
    source = WEIGHTS
    data = pd.read_csv(source, parse_dates=["update_date"])
    data = data.loc[
        data["mode"].eq("qlike")
        & data["model_id"].eq("RV-NN-SV")
        & data["available"].astype(bool)
    ].copy()

    counts = data.groupby(["market", "horizon"]).size()
    if len(data) != 450 or not (counts == 50).all():
        raise ValueError("Unexpected QLIKE weight-path dimensions")

    styles = {
        1: dict(color="black", linestyle="-", linewidth=1.55, zorder=3),
        5: dict(
            color="0.32", linestyle=(0, (4.0, 2.0)), linewidth=1.55, zorder=2
        ),
        10: dict(
            color="0.60", linestyle=(0, (1.0, 1.5)), linewidth=1.90, zorder=1
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.4), sharey=True)
    for ax, market in zip(axes, MARKETS):
        for horizon in HORIZONS:
            series = data.loc[
                data["market"].eq(market) & data["horizon"].eq(horizon)
            ].sort_values("update_date")
            ax.plot(
                series["update_date"],
                series["prequential_weight"],
                drawstyle="steps-post",
                label=rf"$\ell={horizon}$",
                **styles[horizon],
            )

        ax.set_title(
            MARKET_LABELS[market].replace("\\&", "&").replace("~", " "),
            fontsize=10.5,
            pad=5,
        )
        ax.set_xlabel("Update date", fontsize=9.5)
        ax.set_ylim(-0.035, 1.055)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="y", color="0.87", linewidth=0.65)
        ax.tick_params(axis="both", labelsize=8.5, length=3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Weight on RV-NN-SV", fontsize=9.5, labelpad=5)
    axes[0].axvline(pd.Timestamp("2020-04-01"), color="0.48", linewidth=1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9.0,
        handlelength=2.4,
        columnspacing=1.8,
    )
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.09, right=0.995, wspace=0.15)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / "combination_weight_paths.pdf", bbox_inches="tight", pad_inches=0.03
    )
    fig.savefig(
        FIGURES / "combination_weight_paths.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT/'forecasts/combination_weights.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'reproduced/figures')
    args = parser.parse_args()
    WEIGHTS = args.weights
    FIGURES = args.output
    build_weight_figure()
