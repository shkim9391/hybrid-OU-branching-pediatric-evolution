#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 22:42:05 2025

@author: seung-hwan.kim
"""

# figure5_lineage_graph_three_panels.py
# Hybrid OU–branching lineage snapshots for WT, priA, recG in one figure.

from dataclasses import dataclass, replace
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------- Config ----------------
EPS_VAR = 1e-12
MAX_N   = 8000  # population cap for safety

@dataclass
class Params:
    # time
    T: float = 40.0
    dt: float = 0.05
    snapshot_frac: float = 0.50
    seed: int = 42
    # OU phenotype
    muY: float = 0.0
    theta: float = 0.8
    sigma: float = 0.35
    # demography
    lam0: float = 0.30   # baseline division
    mu0: float = 0.20    # baseline death
    alpha: float = 0.8   # Y↑→ division↑
    beta: float  = 0.6   # Y↑→ death↓
    # initial
    N0: int = 1
    Y0: float = -0.6

# ------------- OU step & rates -------------
def ou_exact_step(y_prev, dt, mu, theta, sigma, rng):
    e = np.exp(-theta * dt)
    m = mu + (y_prev - mu) * e
    v = (sigma**2 / (2.0 * theta)) * (1.0 - e**2)
    v = max(v, EPS_VAR)
    return rng.normal(m, np.sqrt(v))

def rates_from_y(y, p: Params):
    lam = p.lam0 * np.exp(p.alpha * (y - p.muY))
    mud = p.mu0  * np.exp(-p.beta * (y - p.muY))
    return float(np.clip(lam, 0.0, 5.0)), float(np.clip(mud, 0.0, 5.0))

# ------------- Simulation -------------
def simulate_once(p: Params):
    rng = np.random.default_rng(p.seed)
    t_grid = np.arange(0.0, p.T + p.dt, p.dt)
    t_snap = p.snapshot_frac * p.T

    # cells: id -> dict(parent, y, alive, birth, death)
    cells = {}
    next_id = 0
    for _ in range(p.N0):
        cells[next_id] = dict(parent=None, y=p.Y0, alive=True, birth=0.0, death=None)
        next_id += 1

    edges = []  # (parent, child, lam_at_div, t_div)

    for t in t_grid[1:]:
        alive_ids = [cid for cid, c in cells.items() if c["alive"]]
        if len(alive_ids) == 0 or len(cells) > MAX_N:
            break

        # OU updates
        for cid in alive_ids:
            c = cells[cid]
            c["y"] = ou_exact_step(c["y"], p.dt, p.muY, p.theta, p.sigma, rng)

        # Birth–death tau-leap
        for cid in alive_ids:
            c = cells.get(cid)
            if c is None or not c["alive"]:
                continue
            lam, mud = rates_from_y(c["y"], p)

            # births
            n_birth = rng.poisson(lam * p.dt)
            for _ in range(int(n_birth)):
                if len(cells) >= MAX_N:
                    break
                cells[next_id] = dict(parent=cid, y=c["y"], alive=True, birth=t, death=None)
                edges.append((cid, next_id, lam, t))
                next_id += 1

            # death
            if rng.random() < (1.0 - np.exp(-mud * p.dt)):
                c["alive"] = False
                c["death"] = t

    return cells, edges, t_snap

def snapshot_graphs(cells, edges, t_snap):
    G_extant  = nx.DiGraph()
    G_extinct = nx.DiGraph()

    alive_at_snap, extinct_by_snap = [], []
    for cid, c in cells.items():
        alive_flag = (c["birth"] <= t_snap) and (c["death"] is None or c["death"] > t_snap)
        if alive_flag and c["alive"]:
            alive_at_snap.append(cid)
        else:
            # "extinct by snapshot" includes branches that ended before or started after t_snap
            if (c["death"] is not None and c["death"] <= t_snap) or (c["birth"] > t_snap):
                extinct_by_snap.append(cid)

    for cid in alive_at_snap:
        G_extant.add_node(cid, y=cells[cid]["y"])
    for cid in extinct_by_snap:
        if cells[cid]["birth"] <= t_snap:
            G_extinct.add_node(cid, y=cells[cid]["y"])

    ext_e, exn_e, ext_w, exn_w = [], [], [], []
    for (u, v, lam_at_div, t_div) in edges:
        if t_div > t_snap:
            continue
        if (u in G_extant) and (v in G_extant):
            ext_e.append((u, v)); ext_w.append(lam_at_div)
        elif (u in G_extinct) and (v in G_extinct):
            exn_e.append((u, v)); exn_w.append(lam_at_div)
    return G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w

# Safe width scaling for NumPy 2.x
def scale_widths(ws, lo=0.5, hi=3.0):
    if ws is None:
        return []
    w = np.asarray(ws, dtype=float)
    if w.size == 0:
        return []
    wmin = np.nanmin(w)
    rng = np.ptp(w)  # works across NumPy versions
    if not np.isfinite(rng) or rng <= 1e-12:
        mid = lo + 0.5 * (hi - lo)
        return [mid] * w.size
    return (lo + (hi - lo) * (w - wmin) / rng).tolist()

# ------------- Retry to ensure extant clones -------------
def run_with_retries(p0: Params, max_tries=8):
    p = p0
    for k in range(max_tries):
        cells, edges, t_snap = simulate_once(p)
        G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w = snapshot_graphs(cells, edges, t_snap)
        if len(G_extant) > 0:
            return p, cells, edges, t_snap, G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w

        # Relax parameters progressively
        phase = k % 5
        if phase == 0:
            p = replace(p, mu0 = max(0.02, p.mu0 * 0.75))             # reduce death
        elif phase == 1:
            p = replace(p, lam0 = min(1.2, p.lam0 * 1.25))            # increase division
        elif phase == 2:
            p = replace(p, snapshot_frac = max(0.2, p.snapshot_frac - 0.10))  # earlier snapshot
        elif phase == 3:
            p = replace(p, N0 = min(16, p.N0 * 2))                    # more founders
        else:
            p = replace(p, Y0 = p.Y0 + 0.25)                          # more favorable initial Y
    # last resort: return even if no extant
    return p, cells, edges, t_snap, nx.DiGraph(), nx.DiGraph(), [], [], [], []

# ------------- Plot on a given axis -------------
def draw_lineage_on_axis(ax, G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w, title):
    # Layout on the union graph so positions are consistent
    G_union = nx.compose(G_extant, G_extinct)
    if len(G_union) == 0:
        ax.text(0.5, 0.5, "No extant clones at snapshot", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return

    pos = nx.spring_layout(G_union, k=0.12, iterations=350, seed=7)

    # Edge widths
    ext_w_scaled = scale_widths(ext_w, lo=0.7, hi=3.2)
    exn_w_scaled = scale_widths(exn_w, lo=0.3, hi=1.4)

    # Color map for extant nodes by phenotype Y
    cmap = get_cmap("viridis")
    if len(G_extant) > 0:
        Y_vals = np.array([G_extant.nodes[n]["y"] for n in G_extant.nodes], float)
        if len(Y_vals) > 10:
            vmin, vmax = np.percentile(Y_vals, [5, 95])
        else:
            vmin, vmax = Y_vals.min(), Y_vals.max()
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = -1.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    node_colors_extant = [cmap(norm(G_extant.nodes[n]["y"])) for n in G_extant.nodes]

    # Draw extinct edges/nodes (grey, faint)
    if exn_e:
        nx.draw_networkx_edges(G_extinct, pos, edgelist=exn_e,
                               width=exn_w_scaled, edge_color="0.6", alpha=0.35,
                               arrows=False, ax=ax)
    if len(G_extinct) > 0:
        nx.draw_networkx_nodes(G_extinct, pos, node_size=26, node_color="0.75",
                               alpha=0.55, linewidths=0, ax=ax)

    # Draw extant edges/nodes
    if ext_e:
        nx.draw_networkx_edges(G_extant, pos, edgelist=ext_e,
                               width=ext_w_scaled, edge_color="0.45", alpha=0.75,
                               arrows=False, ax=ax)
    if len(G_extant) > 0:
        nx.draw_networkx_nodes(G_extant, pos, node_size=40, node_color=node_colors_extant,
                               linewidths=0, ax=ax)

    # Per-axis colorbar placed tightly to the right
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.02)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Phenotype $Y_t$", rotation=270, labelpad=12)

    # Legend handles (draw as tiny lines)
    ax.plot([], [], color="0.45", lw=2.6, label="Extant branch (width ∝ λ(Y))")
    ax.plot([], [], color="0.60", lw=1.2, alpha=0.45, label="Extinct branch")

    ax.set_title(title, fontsize=11)
    ax.axis("off")
    ax.legend(loc="lower left", frameon=False, fontsize=8, handlelength=2.4, handletextpad=0.6)

# ------------- Main: three panels -------------
def main(outfile="figure5_lineage_WT_priA_recG.png"):
    MODELS = {
        "WT":   Params(muY=3.105064e-06,  theta=1.000000e-01,   sigma=8.511509e-06, seed=41),
        "priA": Params(muY=1.555009e-05, theta=1.136503e-01, sigma=7.751312e-06, seed=42),
        "recG": Params(muY=1.376895e-07, theta=1.162200e-01, sigma=3.594900e-07, seed=43),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    for ax, (name, base_p) in zip(axes, MODELS.items()):
        p, cells, edges, t_snap, G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w = run_with_retries(base_p, max_tries=8)
        title = f"{name}  (snapshot t={t_snap:.1f}; μ={p.muY:.2e}, θ={p.theta:.4f}, σ={p.sigma:.2e})"
        draw_lineage_on_axis(ax, G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w, title=title)

    fig.suptitle("Hybrid OU–branching lineage snapshots: WT vs priA vs recG", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    main()