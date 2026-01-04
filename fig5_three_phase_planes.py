#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# figure5_three_phase_planes.py
# Three phase-plane panels (WT, priA, recG) for a hybrid OU–Branching model
# Journal-ready: consistent styling, colorblind-safe, 2-column width, 600 dpi

import numpy as np
import matplotlib.pyplot as plt

# ------------------ Global style (Frontiers-friendly) ------------------
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "font.size": 8.5,                     # small but readable for 2-column figs
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

def ou_exact_step(y_prev, dt, mu, theta, sigma):
    e = np.exp(-theta * dt)
    m = mu + (y_prev - mu) * e
    v = (sigma**2 / (2.0 * theta)) * (1.0 - e**2)
    v = max(v, EPS)
    return rng.normal(m, np.sqrt(v))

def rates_from_y(y, lam0, mu0, alpha, beta, muY):
    lam = lam0 * np.exp(alpha * (y - muY))
    mud = mu0 * np.exp(-beta * (y - muY))
    return np.clip(lam, 0.0, 5.0), np.clip(mud, 0.0, 5.0)

def y_dot(y, muY, theta):
    return theta * (muY - y)

def n_dot(y, n, lam0, mu0, alpha, beta, muY):
    lam, mud = rates_from_y(y, lam0, mu0, alpha, beta, muY)
    return (lam - mud) * n

def build_vector_field(ax, muY, theta, lam0, mu0, alpha, beta,
                       ylim=(1, 5e4), yg=(-1.6, 1.6), Ny=23, Nx=25):
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

    ax.quiver(Ymesh, Nmesh, U, V, angles='xy', width=0.0028, scale=16,
              color='0.55', alpha=0.75)

    # Stable OU manifold (Y = muY)
    ax.axvline(muY, ls='--', lw=1.0, color='0.4')

    # Population nullcline (lam = mu)
    y_star = muY + np.log(mu0 / lam0) / (alpha + beta)
    ax.axvline(y_star, ls=':', lw=1.0, color='#E66100')  # orange dotted

def simulate_trajectories(ax, muY, theta, sigma, lam0, mu0, alpha, beta,
                          starts, lw=1.5, alpha_line=0.95):
    """
    starts: list of tuples (Y0, N0, color_hex)
    """
    def ou_step(y):
        e = np.exp(-theta * dt)
        m = muY + (y - muY) * e
        v = (sigma**2 / (2.0 * theta)) * (1.0 - e**2)
        v = max(v, EPS)
        return rng.normal(m, np.sqrt(v))

    for (Y0, N0, color) in starts:
        t = np.arange(0, T + dt, dt)
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

        ax.plot(Y, np.maximum(N, 1.0), color=color, lw=lw, alpha=alpha_line)

# ------------------ Parameter sets ------------------
# Demography (shared)
lam0, mu0, alpha, beta = 0.30, 0.22, 0.90, 0.60

# OU params per lineage
params = {
    "WT":   dict(muY=3.105064e-06,  theta=01.000000e-01,   sigma=8.511509e-06),
    "priA": dict(muY=1.555009e-05, theta=1.136503e-01, sigma=7.751312e-06),
    "recG": dict(muY=1.376895e-07, theta=1.162200e-01, sigma=3.594900e-07),
}

# Colorblind-safe colors for trajectories (Okabe–Ito palette + one)
COLS = ["#0072B2", "#009E73", "#D55E00"]  # blue, green, vermilion

starts_sets = [
    [(-1.2, 40,  COLS[0]), (-0.3, 80, COLS[1]), (0.6, 25, COLS[2])],  # WT
    [(-1.0, 30,  COLS[0]), (0.0,  50, COLS[1]), (0.8, 20, COLS[2])],  # priA
    [(-0.8, 25,  COLS[0]), (0.2,  40, COLS[1]), (1.0, 15, COLS[2])],  # recG
]

# ------------------ Build figure ------------------
# Two-column width ~180 mm ≈ 7.1 in; height ~3.6 in (tight)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)

# Panel letters
panel_labels = ["A", "B", "C"]

for ax, (name, p), starts, letter in zip(axes, params.items(), starts_sets, panel_labels):
    # Vector field
    build_vector_field(ax, p["muY"], p["theta"], lam0, mu0, alpha, beta,
                       ylim=(1, 5e4), yg=(-1.6, 1.6), Ny=23, Nx=25)
    # Stochastic trajectories
    simulate_trajectories(ax, p["muY"], p["theta"], p["sigma"],
                          lam0, mu0, alpha, beta, starts)

    # Axes formatting
    ax.set_yscale("log")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(1, 5e4)
    ax.set_xlabel(r"Phenotype $Y_t$")
    ax.grid(axis="y", which="both", alpha=0.22, linewidth=0.5)

    # Small title + panel letter inside
    ax.set_title(name, pad=4, fontweight="bold")
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, fontweight="bold")

# Shared y-label only on left pane
axes[0].set_ylabel(r"Population size $N_t$ (log scale)")

# Tight layout with top room for a concise super-title (optional)
#fig.suptitle("Phase-plane diagrams of hybrid OU–branching model across lineages",
           #  y=0.995, fontsize=10, fontweight="bold")
#fig.tight_layout(rect=[0, 0, 1, 0.98])

# ------------------ Save ------------------
fig.savefig("figure5_three_phase_planes.png", dpi=1500, bbox_inches="tight")
fig.savefig("figure5_three_phase_planes.pdf", bbox_inches="tight")

print("Saved: figure5_three_phase_planes.(png|pdf)")
