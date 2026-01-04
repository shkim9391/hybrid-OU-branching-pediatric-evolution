#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# figure6_lineage_graph_three_panels_t20.py
# Hybrid OU–branching lineage snapshots for WT, priA, recG in one figure.
# Shared snapshot at t = 20.0
# Updated: Extant and extinct branches drawn thicker and darker.

from dataclasses import dataclass, replace
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------- Config ----------------
EPS_VAR = 1e-12
MAX_N   = 12_000
SNAPSHOT_TIME = 20.0  # shared absolute snapshot
EDGE_COLOR_EXTANT = "0.25"  # darker gray
EDGE_COLOR_EXTINCT = "0.45"  # darker light gray
EDGE_WIDTH_EXTANT = 3.6      # thicker extant branches
EDGE_WIDTH_EXTINCT = 2.0     # thicker extinct branches

@dataclass
class Params:
    T: float = 40.0
    dt: float = 0.05
    seed: int = 42
    muY: float = 0.0
    theta: float = 0.8
    sigma: float = 0.35
    lam0: float = 0.30
    mu0: float = 0.20
    alpha: float = 0.8
    beta: float  = 0.6
    N0: int = 1
    Y0: float = -0.6

# ---------------- OU and Demography ----------------
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

def nullcline_y_star(p: Params):
    return p.muY + np.log(p.mu0 / p.lam0) / (p.alpha + p.beta)

def favorable_start_y(p: Params, margin=0.35):
    y_star = nullcline_y_star(p)
    y0 = max(p.Y0, y_star + margin)
    if p.theta >= 0.1 and p.sigma <= 1e-6:
        y0 = y_star + max(margin, 0.6)
    return y0

# ---------------- Simulation ----------------
def simulate_once(p: Params, t_snap: float):
    T_eff = max(p.T, t_snap)
    rng = np.random.default_rng(p.seed)
    t_grid = np.arange(0.0, T_eff + p.dt, p.dt)

    cells = {}
    edges = []
    next_id = 0
    start_y = favorable_start_y(p)
    for _ in range(p.N0):
        cells[next_id] = dict(parent=None, y=start_y, alive=True, birth=0.0, death=None)
        next_id += 1

    for t in t_grid[1:]:
        alive_ids = [cid for cid, c in cells.items() if c["alive"]]
        if not alive_ids or len(cells) > MAX_N:
            break

        for cid in alive_ids:
            c = cells[cid]
            c["y"] = ou_exact_step(c["y"], p.dt, p.muY, p.theta, p.sigma, rng)

        for cid in list(alive_ids):
            c = cells[cid]
            lam, mud = rates_from_y(c["y"], p)
            for _ in range(int(rng.poisson(lam * p.dt))):
                if len(cells) >= MAX_N:
                    break
                cells[next_id] = dict(parent=cid, y=c["y"], alive=True, birth=t, death=None)
                edges.append((cid, next_id, lam, t))
                next_id += 1
            if rng.random() < (1.0 - np.exp(-mud * p.dt)):
                c["alive"] = False
                c["death"] = t

    return cells, edges, t_snap

def snapshot_graphs(cells, edges, t_snap):
    G_extant, G_extinct = nx.DiGraph(), nx.DiGraph()
    alive, extinct = [], []
    for cid, c in cells.items():
        alive_flag = (c["birth"] <= t_snap) and (c["death"] is None or c["death"] > t_snap)
        if alive_flag and c["alive"]:
            alive.append(cid)
        elif (c["death"] and c["death"] <= t_snap) or (c["birth"] > t_snap):
            extinct.append(cid)
    for cid in alive:   G_extant.add_node(cid, y=cells[cid]["y"])
    for cid in extinct: G_extinct.add_node(cid, y=cells[cid]["y"])

    ext_e, exn_e, ext_w, exn_w = [], [], [], []
    for (u, v, lam, t_div) in edges:
        if t_div > t_snap:
            continue
        if u in G_extant and v in G_extant:
            ext_e.append((u, v)); ext_w.append(lam)
        elif u in G_extinct and v in G_extinct:
            exn_e.append((u, v)); exn_w.append(lam)
    return G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w

def scale_widths(ws, lo=0.5, hi=3.0):
    if not ws: return []
    w = np.asarray(ws, float)
    if w.size == 0: return []
    wmin, rng = np.nanmin(w), np.ptp(w)
    if not np.isfinite(rng) or rng <= 1e-12:
        return [lo + 0.5 * (hi - lo)] * w.size
    return (lo + (hi - lo) * (w - wmin) / rng).tolist()

# ---------------- Retry ----------------
def run_with_retries(p0: Params, t_snap: float, max_tries=10):
    p = replace(p0, Y0=favorable_start_y(p0))
    for k in range(max_tries):
        cells, edges, _ = simulate_once(p, t_snap)
        G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w = snapshot_graphs(cells, edges, t_snap)
        if len(G_extant) > 0:
            return p, cells, edges, t_snap, G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w
        phase = k % 5
        if phase == 0: p = replace(p, mu0=max(0.02, p.mu0 * 0.75))
        elif phase == 1: p = replace(p, lam0=min(1.5, p.lam0 * 1.25))
        elif phase == 2: p = replace(p, N0=min(24, p.N0 * 2))
        elif phase == 3: p = replace(p, Y0=favorable_start_y(p, margin=0.6))
        else: p = replace(p, sigma=p.sigma * 1.5)
    return p, cells, edges, t_snap, nx.DiGraph(), nx.DiGraph(), [], [], [], []

# ---------------- Drawing ----------------
def draw_lineage_on_axis(ax, G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w, title):
    G_union = nx.compose(G_extant, G_extinct)
    if len(G_union) == 0:
        ax.text(0.5, 0.5, "No extant clones", ha="center", va="center")
        ax.set_title(title); ax.axis("off"); return

    pos = nx.spring_layout(G_union, k=0.12, iterations=350, seed=7)
    ext_w_scaled = scale_widths(ext_w, lo=EDGE_WIDTH_EXTANT*0.5, hi=EDGE_WIDTH_EXTANT)
    exn_w_scaled = scale_widths(exn_w, lo=EDGE_WIDTH_EXTINCT*0.5, hi=EDGE_WIDTH_EXTINCT)

    cmap = get_cmap("viridis")
    Y_vals = np.array([G_extant.nodes[n]["y"] for n in G_extant.nodes]) if len(G_extant) else np.array([-1,1])
    vmin, vmax = (np.percentile(Y_vals, [5,95]) if len(Y_vals)>10 else (Y_vals.min(), Y_vals.max()))
    norm = Normalize(vmin=vmin, vmax=vmax)
    node_colors_extant = [cmap(norm(G_extant.nodes[n]["y"])) for n in G_extant.nodes]

    # Extinct branches (darker & thicker)
    if exn_e:
        nx.draw_networkx_edges(G_extinct, pos, edgelist=exn_e,
                               width=exn_w_scaled, edge_color=EDGE_COLOR_EXTINCT,
                               alpha=0.55, arrows=False, ax=ax)
    if len(G_extinct):
        nx.draw_networkx_nodes(G_extinct, pos, node_size=26, node_color="0.75",
                               alpha=0.55, linewidths=0, ax=ax)

    # Extant branches (even darker & thicker)
    if ext_e:
        nx.draw_networkx_edges(G_extant, pos, edgelist=ext_e,
                               width=ext_w_scaled, edge_color=EDGE_COLOR_EXTANT,
                               alpha=0.9, arrows=False, ax=ax)
    if len(G_extant):
        nx.draw_networkx_nodes(G_extant, pos, node_size=40, node_color=node_colors_extant,
                               linewidths=0, ax=ax)

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.02)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Phenotype $Y_t$", rotation=270, labelpad=12)

    ax.plot([], [], color=EDGE_COLOR_EXTANT, lw=EDGE_WIDTH_EXTANT,
            label="Extant branch (width ∝ λ(Y))")
    ax.plot([], [], color=EDGE_COLOR_EXTINCT, lw=EDGE_WIDTH_EXTINCT,
            alpha=0.7, label="Extinct branch")

    ax.set_title(title, fontsize=11)
    ax.axis("off")
    ax.legend(loc="lower left", frameon=False, fontsize=8, handlelength=2.4, handletextpad=0.6)

# ---------------- Main ----------------
def main(outfile="figure6_lineage_WT_priA_recG_t20_darkbranches.png"):
    MODELS = {
        "A. WT":   Params(muY=4.14e-6,  theta=0.10,   sigma=9.178e-6, seed=41, N0=2),
        "B. priA": Params(muY=1.495e-5, theta=0.1126, sigma=6.645e-6, seed=42, N0=3),
        "C. recG": Params(muY=1.578e-7, theta=0.1147, sigma=4.007e-7, seed=43,
                          N0=4, lam0=0.32, mu0=0.18),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    for ax, (name, base_p) in zip(axes, MODELS.items()):
        base_p = replace(base_p, T=max(base_p.T, SNAPSHOT_TIME))
        p, cells, edges, t_snap, G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w = \
            run_with_retries(base_p, t_snap=SNAPSHOT_TIME)
        title = f"{name}  (snapshot t={SNAPSHOT_TIME:.1f}; μ={p.muY:.2e}, θ={p.theta:.4f}, σ={p.sigma:.2e})"
        draw_lineage_on_axis(ax, G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w, title)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    main()
