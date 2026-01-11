#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# figure5_three_phase_planes.py
# Three phase-plane panels (WT, priA, recG) for a hybrid OU–Branching model
# Updated to use replicate-grouped OU parameters on the log10 scale (Table 1, Option A)

import numpy as np
import matplotlib.pyplot as plt

# ------------------ Global style (Frontiers-friendly) ------------------
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ------------------ Shared utilities ------------------
EPS = 1e-12
MAX_N = 1_000_000
T = 40.0
dt = 0.05
rng = np.random.default_rng(123)

def rates_from_y(y, lam0, mu0, alpha, beta, muY):
    """
    Demographic rates depend on deviation from lineage mean (y - muY).
    y is on the log10 scale.
    """
    lam = lam0 * np.exp(alpha * (y - muY))
    mud = mu0 * np.exp(-beta * (y - muY))
    return np.clip(lam, 0.0, 5.0), np.clip(mud, 0.0, 5.0)

def y_dot(y, muY, theta):
    return theta * (muY - y)

def n_dot(y, n, lam0, mu0, alpha, beta, muY):
    lam, mud = rates_from_y(y, lam0, mu0, alpha, beta, muY)
    return (lam - mud) * n

def build_vector_field(ax, muY, theta, lam0, mu0, alpha, beta,
                       ylim=(1, 5e4), x_span=1.6, Ny=23, Nx=25):
    """
    Build a quiver field over y in [muY-x_span, muY+x_span] and N in log space.
    """
    yg = (muY - x_span, muY + x_span)
    Yg = np.linspace(yg[0], yg[1], Nx)
    Ng = np.geomspace(ylim[0], ylim[1], Ny)
    Ymesh, Nmesh = np.meshgrid(Yg, Ng, indexing='xy')

    dY = y_dot(Ymesh, muY, theta)
    dN = n_dot(Ymesh, Nmesh, lam0, mu0, alpha, beta, muY)

    # Normalize arrows; scale dN by N to balance magnitude vs. log axis
    norm = np.hypot(dY, dN / (Nmesh + EPS))
    norm[norm == 0] = 1.0
    U = dY / norm
    V = (dN / (Nmesh + EPS)) / norm
    
    ax.quiver(
        Ymesh, Nmesh, U, V,
        angles="xy",
        width=0.0022,
        scale=18,
        color="0.65",
        alpha=0.45,
        rasterized=True
    )

    # OU mean manifold (Y = muY)
    ax.axvline(muY, ls='--', lw=1.0, color='0.4')

    # Population nullcline (lam = mu)
    # Solve lam0*exp(alpha*(y-muY)) = mu0*exp(-beta*(y-muY))
    y_star = muY + np.log(mu0 / lam0) / (alpha + beta)
    ax.axvline(y_star, ls=':', lw=1.3, color='#E66100')

def simulate_trajectories(ax, muY, theta, sigma, lam0, mu0, alpha, beta,
                          starts, lw=1.5, alpha_line=0.95):
    """
    starts: list of tuples (Y0, N0, color_hex) on the log10 scale for Y0
    """
    def ou_step(y):
        e = np.exp(-theta * dt)
        m = muY + (y - muY) * e
        v = (sigma**2 / (2.0 * theta)) * (1.0 - e**2)
        v = max(v, EPS)
        return rng.normal(m, np.sqrt(v))

    t = np.arange(0, T + dt, dt)

    for (Y0, N0, color) in starts:
        Y = np.empty_like(t)
        N = np.empty_like(t, dtype=float)
        Y[0], N[0] = Y0, N0

        for k in range(1, t.size):
            Y[k] = ou_step(Y[k-1])
            lam, mud = rates_from_y(Y[k], lam0, mu0, alpha, beta, muY)

            if N[k-1] > 0:
                births = rng.poisson(lam * N[k-1] * dt)
                deaths = rng.poisson(mud * N[k-1] * dt)
                Nk = N[k-1] + births - deaths
            else:
                Nk = 0.0

            N[k] = max(0.0, min(Nk, MAX_N))

        step = 3  # plot every 3rd point
        ax.plot(Y[::step], np.maximum(N[::step], 1.0), color=color, lw=lw, alpha=alpha_line, rasterized=True)

# ------------------ Parameter sets ------------------
# Demography (shared)
lam0, mu0, alpha, beta = 0.30, 0.22, 0.90, 0.60

# OU params per lineage (replicate-grouped MLEs on log10 scale)
params = {
    "WT":   dict(muY=-6.799080, theta=0.775093, sigma=0.726647),
    "priA": dict(muY=-5.000374, theta=0.116660, sigma=0.436483),
    "recG": dict(muY=-7.652025, theta=8.515582, sigma=2.789438),
}

# Colors (Okabe–Ito palette)
COLS = ["#0072B2", "#009E73", "#D55E00"]  # blue, green, vermilion

# Use the same *offset* initial Y values relative to each lineage muY
Y_OFFSETS = [-1.2, -0.3, +0.6]
N_STARTS  = [40, 80, 25]

# ------------------ Build figure ------------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)

panel_labels = ["A", "B", "C"]
x_span = 1.6

for ax, (name, p), letter in zip(axes, params.items(), panel_labels):
    starts = [(p["muY"] + Y_OFFSETS[i], N_STARTS[i], COLS[i]) for i in range(3)]

    build_vector_field(ax, p["muY"], p["theta"], lam0, mu0, alpha, beta,
                       ylim=(1, 5e4), x_span=x_span, Ny=18, Nx=19)
    # and inside build_vector_field:
    simulate_trajectories(ax, p["muY"], p["theta"], p["sigma"],
                          lam0, mu0, alpha, beta, starts, lw=1.1, alpha_line=0.80)

    ax.set_yscale("log")
    ax.set_xlim(p["muY"] - x_span, p["muY"] + x_span)
    ax.set_ylim(1, 5e4)
   
    ax.grid(axis="y", which="both", alpha=0.22, linewidth=0.5)

    ax.set_title(name, pad=4, fontweight="bold")
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, fontweight="bold")
    
fig.supxlabel(r"$Y_t=\log_{10}(\mathrm{mutation\ frequency})$")
fig.subplots_adjust(bottom=0.16, wspace=0.18)
axes[0].set_ylabel(r"Population size $N_t$ (log scale)")

# ------------------ Save ------------------
fig.savefig("figure5_three_phase_planes.png", dpi=600, bbox_inches="tight")
fig.savefig("figure5_three_phase_planes.pdf", bbox_inches="tight")

print("Saved: figure5_three_phase_planes.(png|pdf)")
