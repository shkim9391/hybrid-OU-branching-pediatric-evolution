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
    # Start slightly on the growth-favorable side of the nullcline, but not too far from muY.
    y_star = nullcline_y_star(p)
    return max(p.Y0, y_star + margin)

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

    for cid, c in cells.items():
        # Only consider clones that have been born by t_snap
        if c["birth"] > t_snap:
            continue

        alive_at_snap = (c["death"] is None) or (c["death"] > t_snap)
        if alive_at_snap:
            G_extant.add_node(cid, y=c["y"])
        else:
            G_extinct.add_node(cid, y=c["y"])

    ext_e, exn_e, ext_w, exn_w = [], [], [], []
    for (u, v, lam, t_div) in edges:
        if t_div > t_snap:
            continue
        # Edge is relevant only if BOTH nodes exist by t_snap
        if (u in G_extant and v in G_extant):
            ext_e.append((u, v)); ext_w.append(lam)
        elif (u in G_extinct and v in G_extinct):
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
def draw_lineage_on_axis(ax, G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w, title,
                         max_nodes_extant=None, seed=7, show_legend=False):
    G_union = nx.compose(G_extant, G_extinct)
    rng = np.random.default_rng(seed)

    # Downsample extant nodes for visualization if requested
    if max_nodes_extant is not None and len(G_extant) > max_nodes_extant:
        keep = rng.choice(list(G_extant.nodes), size=max_nodes_extant, replace=False)
        keep = set(keep)
    
        # keep extinct as-is (or also downsample if you want)
        G_extant = G_extant.subgraph(keep).copy()
    
        # filter extant edges/weights consistently
        new_ext_e, new_ext_w = [], []
        for (u, v), w in zip(ext_e, ext_w):
            if u in keep and v in keep:
                new_ext_e.append((u, v)); new_ext_w.append(w)
        ext_e, ext_w = new_ext_e, new_ext_w
    
        # rebuild union after downsampling
        G_union = nx.compose(G_extant, G_extinct)
        
    if len(G_union) == 0:
        ax.text(0.5, 0.5, "No extant clones", ha="center", va="center")
        ax.set_title(title); ax.axis("off"); return

    pos = nx.spring_layout(G_union, k=0.12, iterations=350, seed=seed)
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
    cbar.set_label(r"$Y_t$ (log$_{10}$ freq)", rotation=270, labelpad=14)

    ax.set_title(title, fontsize=12, pad=6)
    ax.axis("off")
    
    if show_legend:
        ax.plot([], [], color=EDGE_COLOR_EXTANT, lw=EDGE_WIDTH_EXTANT,
                label="Extant branch (width ∝ λ(Y))")
        ax.plot([], [], color=EDGE_COLOR_EXTINCT, lw=EDGE_WIDTH_EXTINCT,
                alpha=0.7, label="Extinct branch")
        ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False, fontsize=8,
                  handlelength=2.4, handletextpad=0.6)

# ---------------- Main ----------------
def main(outfile="figure6_lineage_WT_priA_recG_t20_darkbranches.png"):
    # Shared demography (match Figure 5)
    DEMO = dict(lam0=0.30, mu0=0.22, alpha=0.90, beta=0.60)
    
    # Option A (replicate-grouped MLEs on log10 scale) — matches Fig4 + Table 1 (grouped)
    MODELS = {
        "A. WT":   Params(muY=-6.799080, theta=0.775093, sigma=0.726647,
                          seed=41, N0=2, Y0=-6.799080 - 0.6, **DEMO),
    
        "B. priA": Params(muY=-5.000374, theta=0.116660, sigma=0.436483,
                          seed=42, N0=3, Y0=-5.000374 - 0.6, **DEMO),
    
        "C. recG": Params(muY=-7.652025, theta=8.515582, sigma=2.789438,
                          seed=43, N0=4, Y0=-7.652025 - 0.6, **DEMO),
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    
    for ax, (name, base_p) in zip(axes, MODELS.items()):
        base_p = replace(base_p, T=max(base_p.T, SNAPSHOT_TIME))
        p, cells, edges, t_snap, G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w = \
            run_with_retries(base_p, t_snap=SNAPSHOT_TIME)
    
        cap = 2000 if "priA" in name else None
        show_legend = name.startswith("A.")
    
        draw_lineage_on_axis(
            ax, G_extant, G_extinct, ext_e, exn_e, ext_w, exn_w,
            name,
            max_nodes_extant=cap,
            seed=7,
            show_legend=show_legend
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    main()
