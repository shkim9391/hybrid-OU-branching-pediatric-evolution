#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 21:16:23 2025

@author: seung-hwan.kim
"""

import matplotlib.pyplot as plt
import numpy as np

# ----- data (in days) -----
human_days = 5 * 365            # 5 years ≈ 1,825 days
ecoli_days = 25.3               # ≈ 1,825 generations at ~20 min/division
avg_dx_days = human_days        # average pediatric diagnosis age

labels = ["H. sapiens (5 years)", "E. coli (25.3 days)"]
days   = np.array([human_days, ecoli_days], dtype=float)

# ----- plot -----
plt.figure(figsize=(10, 4), dpi=300)
y = np.arange(len(labels))
plt.barh(y, days, color=["#93b4d8", "#f5d67b"], edgecolor="black", height=0.55)
plt.xscale("log")

# ticks/grid
xticks = [1, 3, 10, 30, 100, 300, 1000, 1825, 3000]
plt.xticks(xticks, [str(t) for t in xticks])
plt.grid(axis="x", which="both", ls=":", lw=0.7, alpha=0.6)

# y and labels
plt.gca().invert_yaxis()
plt.yticks(y, labels, fontsize=11)
plt.xlabel("Time (days, log scale)", fontsize=12)
plt.title("Lifespan/Time Horizon vs. Pediatric Cancer Diagnosis (~5 years)", fontsize=13)

# ----- pediatric diagnosis marker -----
plt.axvline(avg_dx_days, color="crimson", lw=1.6, ls="--", alpha=0.9)
plt.text(avg_dx_days*1.04, -0.25, "Avg pediatric cancer diagnosis \n≈ 5 years (1,825 days)",
         color="crimson", fontsize=10, va="top")

# annotations for bars
plt.annotate("≈ 1,825 generations (24 hr/div)", xy=(human_days, 0), xytext=(human_days*1.05, 0.15),
             fontsize=10, va="center")
plt.annotate("≈ 1,825 generations (20 min/div) \nLong-Term Evolution Experiment (LTEE)",
             xy=(ecoli_days, 1), xytext=(ecoli_days*1.6, 1.05))

plt.tight_layout()
plt.savefig("supp_figure1a_lifespan_dx_comparison_log.png", dpi=1500, bbox_inches="tight")
plt.show()