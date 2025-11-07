#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 22:41:26 2025

@author: seung-hwan.kim
"""

# fig1_line_plot_with_bands_updated.py
# Log-scale plot with ±SD shaded bands, safe zero handling, replicate aggregation, no hard-coded colors.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = "mut_freq_data.csv"      # expects long/tidy columns: background, replicate, t, x
OUT = "mutation_frequency_with_bands_log.png"

# -----------------------------
# Load & clean
# -----------------------------
df = pd.read_csv(CSV)

# Drop empty "Unnamed" columns if present
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, na=False)]

# Normalize column names -> background, day, value
colmap = {}
for c in df.columns:
    lc = c.lower()
    if lc in {"background"}: colmap[c] = "background"
    elif lc in {"t", "time", "day", "days"}: colmap[c] = "day"
    elif lc in {"x", "value", "y"}: colmap[c] = "value"
    elif lc in {"rep", "replicate", "rep_id"}: colmap[c] = "replicate"
df = df.rename(columns=colmap)

need = {"background", "day", "value"}
if not need.issubset(df.columns):
    raise ValueError(f"CSV must include columns at least {need} (case-insensitive).")

df["background"] = df["background"].astype(str).str.strip().str.lower()
df["day"] = pd.to_numeric(df["day"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["background", "day", "value"])

# -----------------------------
# Aggregate by timepoint per lineage
# -----------------------------
# keep only the 3 lineages if present
expected = {"wt", "pria", "recg"}  # lowercased keys
df = df[df["background"].isin(expected)]

agg = (
    df.groupby(["background", "day"], as_index=False)
      .agg(mu=("value", "mean"), sd=("value", "std"), n=("value", "count"))
      .sort_values(["background", "day"])
)

present = list(agg["background"].unique())
label_map = {"wt": "WT", "pria": "priA", "recg": "recG"}

# -----------------------------
# Safe zero handling for log scale
# -----------------------------
positive_vals = df.loc[df["value"] > 0, "value"]
eps = (positive_vals.min() / 10.0) if len(positive_vals) else 1e-12  # small positive

# -----------------------------
# Plot (no hard-coded colors)
# -----------------------------
plt.figure(figsize=(8, 5), dpi=300)

for key in present:
    sub = agg[agg["background"] == key]
    t = sub["day"].to_numpy()
    mu = sub["mu"].to_numpy()
    sd = sub["sd"].to_numpy()   # NaN when only one replicate

    # replace non-positive means to allow log plotting
    mu_plot = np.where(mu > 0, mu, eps)
    plt.plot(t, mu_plot, marker="o", linewidth=2.0, label=label_map.get(key, key))

    # draw SD band only where we have finite SD
    if np.isfinite(sd).any():
        sd0 = np.nan_to_num(sd, nan=0.0)
        lo = np.clip(mu - sd0, a_min=eps, a_max=None)
        hi = np.clip(mu + sd0, a_min=eps, a_max=None)
        plt.fill_between(t, lo, hi, alpha=0.15, linewidth=0)

plt.xlabel("Day", fontsize=12)
plt.ylabel("Mutation Frequency", fontsize=12)
plt.title("Mutation Frequency Trajectories with ±SD Bands (Log Scale)", fontsize=13, pad=10)

plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
plt.legend(title="Lineage", fontsize=10, title_fontsize=10, frameon=True)

plt.tight_layout()
plt.savefig(OUT, dpi=1500, bbox_inches="tight")
plt.show()

# --- Optional: switch to linear scale ---
# plt.yscale("linear"); plt.title("... (Linear Scale)"); plt.savefig("..._linear.png", dpi=600)