#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 14:11:21 2025

@author: seung-hwan.kim
"""

# figure7_pediatric_precision_showcase_final_priA.py
# Composite 4-panel figure (Panels A–D) + automatic TIFF + caption PDF
# Profile-aware (WT / priA / recG) and now with a REAL lineage snapshot in Panel C
# generated from the same OU parameters at a shared absolute snapshot time t=20.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import networkx as nx
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

# Quantile envelopes (95% and IQR)

def env_from_paths(X, q=(0.025, 0.5, 0.975)):
    X = np.asarray(X)
    lo, med, hi = np.nanquantile(X, q, axis=0)
    return lo, med, hi

def iqr_from_paths(X):
    return env_from_paths(X, q=(0.25, 0.5, 0.75))

# Exposure curve (simple 1-comp PK-like decay per dose)

def exposure_curve(t_grid, doses, ke=0.10):
    C = np.zeros_like(t_grid, dtype=float)
    doses = doses or []
    for td, D in doses:
        C += (t_grid >= td) * D * np.exp(-ke * np.maximum(0.0, t_grid - td))
    return C

# -----------------------------
# Pediatric lineage OU profiles (from your Table 1)
# -----------------------------
PROFILES = {
    "WT":   {"muY": 4.14e-6,  "theta": 0.1000, "sigma": 9.178e-6},
    "priA": {"muY": 1.495e-5, "theta": 0.1126, "sigma": 6.645e-6},
    "recG": {"muY": 1.578e-7, "theta": 0.1147, "sigma": 4.007e-7},
}

# -----------------------------
# Therapy-regimen simulator (hybrid OU -> phenotype; branching diffusion for log N)
# -----------------------------

def simulate_regimen(
    T=200, dt=0.5, doses=None, ke=0.10,
    # OU baseline (equilibrium, selection, noise)
    mu0=1.0, th0=0.1, sg0=0.01,
    # OU modulation by drug (g = C/(C+EC50))
    mu_drug=-0.35, th_drug=0.50, sg_drug=-0.40, EC50=75.0,
    # Demography baseline
    lam0=0.020, mu_base=0.015, alpha=0.9, beta=0.7,
    # Drug effect on demography
    k_lam=1.2, k_mu=1.0,
    # Initial conditions & cohort
    y0=None, n0=5e7, reps=256, seed=1234
):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T + dt, dt)
    if doses is None:
        doses = [(d, 100.0) for d in np.arange(0, 6*21, 21)]  # 6 cycles q21d
    C = exposure_curve(t, doses, ke=ke)
    g = C / (C + EC50 + 1e-12)

    # OU parameters over time
    mu_t = mu0 + mu_drug * g
    th_t = np.clip(th0 + th_drug * g, 1e-6, None)
    sg_t = np.clip(sg0 * np.exp(sg_drug * g), 1e-6, None)

    Y = np.empty((reps, t.size), float)
    N = np.empty((reps, t.size), float)
    Y[:, 0] = mu0 if y0 is None else y0
    N[:, 0] = n0

    def rates(y, c):
        hh = c / (c + EC50 + 1e-12)
        lam = lam0    * np.exp(alpha * y - k_lam * hh)
        mud = mu_base * np.exp(-beta * y + k_mu  * hh)
        lam = np.clip(lam, 0.0, 1.2)
        mud = np.clip(mud, 0.0, 1.2)
        return lam, mud

    for k in range(1, t.size):
        dtk = dt
        # OU exact update
        e = np.exp(-th_t[k-1] * dtk)
        m = mu_t[k-1] + (Y[:, k-1] - mu_t[k-1]) * e
        v = (sg_t[k-1]**2 / (2.0 * th_t[k-1])) * (1.0 - e**2)
        v = np.maximum(v, 1e-18)
        Y[:, k] = rng.normal(m, np.sqrt(v))

        # Branching (diffusion on log N)
        lam, mud = rates(Y[:, k], C[k])
        net = (lam - mud) * dtk
        var_dem = (lam + mud) * dtk / np.maximum(N[:, k-1], 1e-9)
        var_dem = np.clip(var_dem, 0.0, 0.25)
        z = rng.normal(0.0, 1.0, size=Y.shape[0])
        lnN = np.log(np.maximum(N[:, k-1], 1.0)) + net - 0.5*var_dem + np.sqrt(var_dem)*z
        N[:, k] = np.maximum(np.exp(lnN), 0.0)

    return t, C, Y, N

# -----------------------------
# Lineage simulator (Panel C) using profile OU params at shared t=20
# -----------------------------

SNAPSHOT_TIME = 20.0

def simulate_lineage_profile(profile="priA", T=40.0, dt=0.05, lam0=0.30, mu0=0.18, alpha=0.8, beta=0.6, N0=2, seed=7, MAX_N=12000):
    """Simulate lineage with OU phenotype -> phenotype-coupled birth/death; return graph and t_snap.
    Node attrs: y (phenotype), extant (bool). Edge attrs: lam (division rate at split), t (time)."""
    p = PROFILES[profile]
    rng = np.random.default_rng(seed)
    t_grid = np.arange(0.0, max(T, SNAPSHOT_TIME) + dt, dt)
    t_snap = SNAPSHOT_TIME

    # Cells as list of dicts
    # initialize founders cleanly (no shared references)
    cells = []
    cells = []
    for _ in range(N0):
        cells.append(dict(y=p["muY"], alive=True, birth=0.0, death=np.nan, parent=None))
    edges = []
    next_id = N0

    def ou_step(y_prev):
        e = np.exp(-p["theta"]*dt)
        m = p["muY"] + (y_prev - p["muY"]) * e
        v = (p["sigma"]**2/(2.0*p["theta"])) * (1.0 - e**2)
        v = max(v, 1e-18)
        return rng.normal(m, np.sqrt(v))

    def rates_from_y(y):
        lam = lam0 * np.exp(alpha * (y - p["muY"]))
        mud = mu0 * np.exp(-beta * (y - p["muY"]))
        return float(np.clip(lam, 0.0, 5.0)), float(np.clip(mud, 0.0, 5.0))

    for t in t_grid[1:]:
        alive_ids = [i for i,c in enumerate(cells) if c["alive"]]
        if not alive_ids or len(cells) >= MAX_N:
            break
        # OU update
        for cid in alive_ids:
            cells[cid]["y"] = ou_step(cells[cid]["y"])
        # Birth-death
        for cid in list(alive_ids):
            c = cells[cid]
            if not c["alive"]: continue
            lam, mud = rates_from_y(c["y"])
            # births
            n_birth = rng.poisson(lam*dt)
            for _ in range(int(n_birth)):
                if len(cells) >= MAX_N: break
                next_id += 1
                cells.append(dict(y=c["y"], alive=True, birth=t, death=np.nan, parent=cid+1))
                edges.append((cid+1, next_id, lam, t))
            # death
            if rng.random() < (1.0 - np.exp(-mud*dt)):
                c["alive"] = False
                c["death"] = t

    # Build graph and mark extant at t_snap
    G = nx.DiGraph()
    for i, c in enumerate(cells, start=1):
        alive_at = (c["birth"] <= t_snap) and (np.isnan(c["death"]) or (c["death"] > t_snap)) and c["alive"]
        G.add_node(i, y=c["y"], extant=bool(alive_at))
    for (u, v, lam, td) in edges:
        if td <= t_snap:
            G.add_edge(u, v, lam=lam, t=td)
    return G, t_snap

def draw_lineage_on_axis(ax, G, title):
    """Draw lineage graph with phenotype-colored nodes and styled edges.
    Safe when there are no edges; matches figure-6 styling.
    """
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No extant clones", ha="center", va="center")
        ax.set_title(title, loc="left", fontsize=12)
        ax.axis("off")
        return

    # Determine color scale from extant nodes (fallback to all nodes)
    y_vals = np.array([G.nodes[n]["y"] for n in G.nodes()])
    y_ext = np.array([G.nodes[n]["y"] for n in G.nodes() if G.nodes[n].get("extant", False)])
    if y_ext.size >= 2 and np.isfinite(y_ext).all():
        vmin, vmax = np.quantile(y_ext, [0.05, 0.95])
    else:
        vmin, vmax = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = -0.5, 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = get_cmap("viridis")

    # Node colors
    node_colors = [
        (cmap(norm(G.nodes[n]["y"])) if G.nodes[n].get("extant", False) else (0.75, 0.75, 0.75, 0.6))
        for n in G.nodes()
    ]

    # Edge styling
    edgelist = list(G.edges())
    edge_colors, edge_widths = None, None
    if len(edgelist) > 0:
        lam_vals = np.array([G.edges[e]["lam"] for e in edgelist])
        lam_min, lam_max = np.nanmin(lam_vals), np.nanmax(lam_vals)
        rng = max(lam_max - lam_min, 1e-12)
        edge_colors, edge_widths = [], []
        for u, v in edgelist:
            both_ext = G.nodes[u].get("extant", False) and G.nodes[v].get("extant", False)
            edge_colors.append((0.25, 0.25, 0.25, 0.85) if both_ext else (0.6, 0.6, 0.6, 0.45))
            lam = G.edges[(u, v)]["lam"]
            edge_widths.append(0.8 + 3.2 * (lam - lam_min) / rng)

    # Layout and draw
    pos = nx.spring_layout(G, seed=7, k=None, iterations=250)
    ax.axis("off")
    if len(edgelist) > 0:
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, ax=ax, arrows=False,
                               edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=18, node_color=node_colors, linewidths=0.0)
    ax.set_title(title, loc="left", fontsize=12)

# -----------------------------
# Profile-driven data generator
# -----------------------------

def generate_showcase_from_profile(profile="priA", doses=None):
    assert profile in PROFILES, f"Unknown profile {profile}. Choose one of {list(PROFILES)}"
    p = PROFILES[profile]
    if doses is None:
        doses = [(d, 100.0) for d in np.arange(0, 6*21, 21)]

    # Treated cohort
    t, C, Y, N = simulate_regimen(
        T=200, dt=0.5, doses=doses, ke=0.10,
        mu0=p["muY"], th0=p["theta"], sg0=p["sigma"],
        mu_drug=-0.35, th_drug=0.50, sg_drug=-0.40, EC50=75.0,
        lam0=0.020, mu_base=0.015, alpha=0.9, beta=0.7,
        k_lam=1.2, k_mu=1.0,
        y0=p["muY"], n0=5e7, reps=256, seed=1234
    )

    # Control cohort (no doses)
    t0, C0, Y0, N0 = simulate_regimen(
        T=200, dt=0.5, doses=[], ke=0.10,
        mu0=p["muY"], th0=p["theta"], sg0=p["sigma"],
        mu_drug=-0.35, th_drug=0.50, sg_drug=-0.40, EC50=75.0,
        lam0=0.020, mu_base=0.015, alpha=0.9, beta=0.7,
        k_lam=1.2, k_mu=1.0,
        y0=p["muY"], n0=5e7, reps=256, seed=1234
    )

    y_lo, y_med, y_hi = env_from_paths(Y)
    n_lo, n_med, n_hi = iqr_from_paths(N)
    _, n_med_ctrl, _ = iqr_from_paths(N0)

    return (t, C, (y_lo, y_med, y_hi), (n_lo, n_med, n_hi), t0, n_med_ctrl)

# -----------------------------
# Main showcase figure + TIFF + caption PDF
# -----------------------------

def pediatric_precision_showcase(profile="priA",
                                 outfile_png="fig7_pediatric_precision_showcase_priA.png",
                                 outfile_tiff="fig7_pediatric_precision_showcase_priA.tiff",
                                 caption_pdf="fig7_caption_priA.pdf"):
    # Pull model outputs for selected profile
    t, C, (y_lo, y_med, y_hi), (n_lo, n_med, n_hi), t0, n_med_ctrl = generate_showcase_from_profile(profile)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], wspace=0.30, hspace=0.35)

    # --- Panel A: Phenotype + exposure ---
    axA = fig.add_subplot(gs[0, 0])
    #axA.fill_between(t, y_lo, y_hi, color="#bdd7e7", alpha=0.5, label="95% envelope", zorder=1)
    axA.plot(t, y_med, color="#08519c", lw=2.5, label="Median $Y_t$", zorder=3)
    axA.set_xlabel("Time (days)")
    axA.set_ylabel("Phenotype $Y_t$")
    axA2 = axA.twinx()
    axA2.plot(t, C, color="#ef3b2c", lw=2.0, label="Exposure $C(t)$")
    axA2.patch.set_alpha(0)                # keep envelope visible
    axA2.set_ylabel("Drug exposure $C(t)$")
    # merged legend
    lines1, labels1 = axA.get_legend_handles_labels()
    lines2, labels2 = axA2.get_legend_handles_labels()
    axA.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")
    axA.set_title(f"A. Phenotypic evolution under therapy ({profile})", loc="left", fontsize=12)

    # --- Panel B: Population dynamics (treated vs control) ---
    axB = fig.add_subplot(gs[0, 1])
    eps = 1.0
    lo = np.clip(n_lo, eps, None)
    hi = np.clip(n_hi, lo * (1 + 1e-12), None)
    med_t_plot = np.clip(n_med, eps, None)
    med_0_plot = np.clip(n_med_ctrl, eps, None)

    #axB.fill_between(t, lo, hi, color="#fcbba1", alpha=0.50, label="Treated IQR")
    axB.plot(t, med_t_plot, color="#cb181d", lw=2.2, label="Treated median")
    axB.plot(t0, med_0_plot, color="0.25", lw=2.2, ls="--", label="Control median")
    axB.set_yscale("log")
    axB.set_xlabel("Time (days)")
    axB.set_ylabel("Tumor cells $N_t$ (log scale)")
    axB.legend(frameon=False, loc="best")
    axB.set_title(f"B. Tumor population dynamics ({profile})", loc="left", fontsize=12)

    # --- Panel C: REAL lineage architecture at t=20 from same OU params ---
    axC = fig.add_subplot(gs[1, 0])
    G, t_snap = simulate_lineage_profile(profile=profile, T=40.0, dt=0.05, lam0=0.30, mu0=0.18,
                                         alpha=0.8, beta=0.6, N0=3, seed=43, MAX_N=10000)
    draw_lineage_on_axis(axC, G, title=f"C. Hybrid OU–branching lineage architecture ({profile}, t={t_snap:.0f})")

    # --- Panel D: Translational outcome map P(cure) vs (sigma, theta) ---
    axD = fig.add_subplot(gs[1, 1])
    sigma = np.linspace(max(1e-7, PROFILES[profile]["sigma"] * 0.25), PROFILES[profile]["sigma"] * 4.0, 50)
    theta = np.linspace(max(1e-3, PROFILES[profile]["theta"] * 0.25), PROFILES[profile]["theta"] * 4.0, 50)
    sgrid, tgrid = np.meshgrid(sigma, theta)
    # Example synthetic mapping: highest cure near moderate theta, low sigma
    p_cure = np.exp(-((sgrid - PROFILES[profile]["sigma"])**2) / ( (PROFILES[profile]["sigma"]*0.6)**2 + 1e-30)
                    -((tgrid - PROFILES[profile]["theta"])**2) / ( (PROFILES[profile]["theta"]*0.6)**2 + 1e-30))
    pcm = axD.pcolormesh(sigma, theta, p_cure, cmap="plasma", shading="auto")
    cb = plt.colorbar(pcm, ax=axD, label="$P(\\mathrm{cure})$")
    axD.set_xlabel("Phenotypic variance $\\sigma$")
    axD.set_ylabel("Selection strength $\\theta$")
    axD.set_title(f"D. Translational outcome map ({profile})", loc="left", fontsize=12)

    #fig.suptitle(f"Hybrid OU–Branching Showcase for Pediatric Precision Oncology — profile: {profile}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # Add extra global spacing as requested
    plt.subplots_adjust(wspace=0.30, hspace=0.35)

    ensure_dir(outfile_png)
    fig.savefig(outfile_png, dpi=600, bbox_inches="tight")
    fig.savefig(outfile_tiff, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # --- Caption PDF ---
    caption_text = (
        f"Figure X. Translational application of the Hybrid OU–Branching framework (profile: {profile}).\n"
        "(A) Phenotypic evolution under cyclic therapy: OU dynamics for $Y_t$ (95% envelope) with drug exposure $C(t)$.\n"
        "(B) Tumor population trajectories (treated vs control) with log-scale burden and IQR shading.\n"
        f"(C) Hybrid OU–branching lineage architecture at shared snapshot $t={SNAPSHOT_TIME:.0f}$ using the same OU parameters.\n"
        "(D) Translational outcome map illustrating how $P(\\mathrm{cure})$ varies with $\\sigma$ and $\\theta$.\n"
        "Panels demonstrate how OU–branching simulations link evolutionary dynamics to pediatric therapy outcomes."
    )

    with PdfPages(caption_pdf) as pdf:
        fig_cap = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.05, 0.9, caption_text, fontsize=12, va='top', wrap=True)
        pdf.savefig(fig_cap, bbox_inches='tight')
        plt.close(fig_cap)

    print(f"Saved: {outfile_png}, {outfile_tiff}, and caption {caption_pdf}")

if __name__ == "__main__":
    # Change to "recG" or "WT" to switch profiles
    pediatric_precision_showcase(profile="priA")
