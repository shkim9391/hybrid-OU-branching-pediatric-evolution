#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 17:13:21 2025

@author: seung-hwan.kim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------- OU exact-transition negative log-likelihood ----------
def nll_ou(params, t, x):
    """params = [mu, log_theta, log_sigma]; exact OU transitions."""
    mu, log_theta, log_sigma = params
    theta, sigma = np.exp(log_theta), np.exp(log_sigma)

    idx = np.argsort(t)
    t = np.asarray(t)[idx]; x = np.asarray(x)[idx]
    if len(x) < 2:
        return 1e12

    ll = 0.0
    for i in range(1, len(x)):
        dt = t[i] - t[i-1]
        if dt <= 0 or not np.isfinite(dt):
            continue
        e = np.exp(-theta * dt)
        m = mu + (x[i-1] - mu) * e
        v = (sigma**2 / (2.0 * theta)) * (1.0 - e**2)
        v = max(v, 1e-18)
        ll += 0.5 * (np.log(2*np.pi*v) + (x[i] - m)**2 / v)
    return ll

def fit_ou_group(df_bg):
    """Joint MLE across replicates of a background (shared mu, theta, sigma)."""
    if df_bg.empty:
        return dict(mu=np.nan, theta=np.nan, sigma=np.nan, nll=np.inf, ok=False)
    groups = [g.sort_values("t") for _, g in df_bg.groupby("replicate")]
    xall = df_bg["x"].values
    start = np.array([np.mean(xall), np.log(0.1), np.log(np.std(xall) + 1e-12)])

    def nll(params):
        return sum(nll_ou(params, g["t"].values, g["x"].values) for g in groups)

    res = minimize(nll, start, method="L-BFGS-B")
    mu, log_th, log_sg = res.x
    return dict(mu=mu, theta=np.exp(log_th), sigma=np.exp(log_sg), nll=res.fun, ok=res.success)

# ---------- Simulate OU paths (exact) for envelopes ----------
def ou_sim_paths(mu, theta, sigma, t_grid, x0, n_paths=800, seed=123):
    rng = np.random.default_rng(seed)
    t = np.asarray(t_grid)
    X = np.zeros((n_paths, len(t)))
    X[:, 0] = x0
    for k in range(1, len(t)):
        dt = t[k] - t[k-1]
        e  = np.exp(-theta * dt)
        m  = mu + (X[:, k-1] - mu) * e
        v  = (sigma**2 / (2.0 * theta)) * (1.0 - e**2)
        v  = np.maximum(v, 1e-18)
        X[:, k] = rng.normal(m, np.sqrt(v))
    return X

def env_from_paths(X):
    mean = np.nanmean(X, axis=0)
    lo   = np.quantile(X, 0.025, axis=0)
    hi   = np.quantile(X, 0.975, axis=0)
    return mean, lo, hi

# ---------- Load data ----------
df = pd.read_csv("mut_freq_data.csv")

# Robust column checks (assumes columns exist in the CSV)
required = {"background","replicate","t","x"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["background","replicate","t","x"]).copy()
df["background"] = df["background"].astype(str).str.lower()
df["replicate"]  = df["replicate"].astype(int)

# Fit OU per background
fit_wt   = fit_ou_group(df[df.background=="wt"])
fit_priA = fit_ou_group(df[df.background=="pria"])
fit_recG = fit_ou_group(df[df.background=="recg"])

print("wt:   ", fit_wt)
print("priA: ", fit_priA)
print("recG: ", fit_recG)

# Common time grid
t_grid = np.linspace(df.t.min(), df.t.max(), 400)

# Helper to build envelopes for a background
def build_env(df_bg, fit):
    dfg = df_bg.sort_values("t")
    x0  = dfg.x.values[0] if len(dfg)>0 else fit["mu"]
    X   = ou_sim_paths(fit["mu"], fit["theta"], fit["sigma"], t_grid, x0)
    m, lo, hi = env_from_paths(X)
    return dfg, m, lo, hi

# Envelopes: wt, priA, recG
dfG, mG, loG, hiG = build_env(df[df.background=="wt"],   fit_wt)
dfA, mA, loA, hiA = build_env(df[df.background=="pria"], fit_priA)
dfR, mR, loR, hiR = build_env(df[df.background=="recg"], fit_recG)

# ---------- Plot Figure 3A (now wt vs priA vs recG) ----------
fig, ax = plt.subplots(figsize=(9.5, 5.4))

# wt (blue theme)
ax.fill_between(t_grid, loG, hiG, color="#bdd7e7", alpha=0.45, label="WT 95% envelope")
ax.plot(t_grid, mG, color="#08519c", lw=2, label="WT mean")
ax.scatter(dfG.t, dfG.x, color="#2171b5", s=20, zorder=3)

# priA (red theme)
ax.fill_between(t_grid, loA, hiA, color="#fcbba1", alpha=0.45, label="priA 95% envelope")
ax.plot(t_grid, mA, color="#cb181d", lw=2, label="priA mean")
ax.scatter(dfA.t, dfA.x, color="#ef3b2c", s=20, zorder=3)

# recG (green theme)
ax.fill_between(t_grid, loR, hiR, color="#c7e9c0", alpha=0.45, label="recG 95% envelope")
ax.plot(t_grid, mR, color="#238b45", lw=2, label="recG mean")
ax.scatter(dfR.t, dfR.x, color="#41ab5d", s=20, zorder=3)

ax.set_title("OU fits and 95% envelopes (WT vs priA vs recG)")
ax.set_xlabel("Time")
ax.set_ylabel("Mutation frequency")
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig("figure2B_fits_envelopes_wt_priA_recG.png", dpi=1500, bbox_inches="tight")
plt.show()

# ---------- AIC table (WT / priA / recG) ----------
k = 3  # parameters: mu, theta, sigma
rows = []
for name, fit in [("WT", fit_wt), ("priA", fit_priA), ("recG", fit_recG)]:
    rows.append(dict(
        Background = name,
        Mu   = fit["mu"],
        Theta= fit["theta"],
        Sigma= fit["sigma"],
        NLL  = fit["nll"],
        AIC  = 2*k + 2*fit["nll"],
        OK   = fit.get("ok", True)
    ))

aic_df = pd.DataFrame(rows)
aic_df = aic_df.sort_values("AIC").reset_index(drop=True)
aic_df["DeltaAIC"] = aic_df["AIC"] - aic_df["AIC"].min()

print("\n===== AIC Comparison (Figure 2B) =====")
print(aic_df.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))

aic_df.to_csv("AIC_comparison_Fig2B.csv", index=False)
print("\nSaved: AIC_comparison_Fig2B.csv")