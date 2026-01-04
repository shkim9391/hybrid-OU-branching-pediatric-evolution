#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# fig1_schematic_ecoli_pediatric_lineages.py

# Schematic: E. coli LTEE lineages ↔ pediatric tumor lineages
# Top-left: OU trait; Top-right: Branching process (birth–death);
# Bottom: two lineage schematics (bacterial vs pediatric)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

rng = np.random.default_rng(7)

# ---------- helpers ----------
def layered_tree(depth=4, max_children=(2,3), extinct_prob=0.25):
    nid = 0
    nodes = [nid]
    level = {nid: 0}
    alive = {nid: True}
    frontier = [nid]
    edges = []
    for L in range(depth):
        nxt = []
        for u in frontier:
            k = rng.integers(max_children[0], max_children[1]+1)
            for _ in range(int(k)):
                nid += 1
                v = nid
                nodes.append(v)
                level[v] = L+1
                edges.append((u, v))
                alive[v] = (rng.random() > extinct_prob*(L/(depth+0.5)))
                nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return edges, nodes, alive, level

def layered_positions(level, x_spread=1.6, y_step=1.0):
    levels = {}
    for n, L in level.items():
        levels.setdefault(L, []).append(n)
    pos = {}
    for L, ids in levels.items():
        m = len(ids)
        xs = np.linspace(-x_spread, x_spread, m) + rng.normal(0, 0.07, m)
        y = -L * y_step
        for x, n in zip(xs, ids):
            pos[n] = (x, y)
    return pos

def draw_tree(ax, edges, alive, pos, trait, cmap="viridis",
              extinct_color=(0.78,0.78,0.78,0.45), lw_extant=2.2, lw_extinct=1.2):
    # edges
    for (u,v) in edges:
        x1,y1 = pos[u]; x2,y2 = pos[v]
        is_extant = alive[u] and alive[v]
        ax.plot([x1,x2],[y1,y2],
                color=(0.3,0.3,0.3,0.75) if is_extant else extinct_color,
                lw=lw_extant if is_extant else lw_extinct,
                solid_capstyle="round")
    # nodes
    vals = np.array([trait[n] for n in pos.keys() if alive[n]])
    if len(vals) == 0:
        vmin, vmax = -1, 1
    else:
        vmin, vmax = np.percentile(vals, [5,95])
        if vmin == vmax:
            vmin, vmax = vmin-0.5, vmax+0.5
    norm = Normalize(vmin=vmin, vmax=vmax)
    cm = get_cmap(cmap)

    patches, colors = [], []
    for n,(x,y) in pos.items():
        patches.append(Circle((x,y), 0.10))
        colors.append(cm(norm(trait[n])) if alive[n] else (0.78,0.78,0.78,0.45))
    coll = PatchCollection(patches, edgecolor="none", facecolor=colors)
    ax.add_collection(coll)
    return norm, cm

def smooth_ou_curve(T=10, dt=0.02, mu=0.0, theta=0.9, sigma=0.6):
    t = np.arange(0, T+dt, dt)
    y = np.zeros_like(t)
    for k in range(1, t.size):
        e = np.exp(-theta*dt)
        m = mu + (y[k-1] - mu)*e
        v = (sigma**2/(2*theta))*(1-e**2)
        y[k] = rng.normal(m, np.sqrt(max(v,1e-12)))
    return t, y

# ---------- simple branching process panel ----------
def branching_paths(T=8, dt=0.02, reps=6, N0=40, lam=0.05, mu=0.035, seed=11):
    """
    Poisson tau-leap birth–death sample paths.
    """
    rloc = np.random.default_rng(seed)
    t = np.arange(0, T+dt, dt)
    Ns = np.zeros((reps, t.size), dtype=float)
    Ns[:, 0] = N0
    for i in range(reps):
        for k in range(1, t.size):
            nprev = Ns[i, k-1]
            if nprev <= 0:
                Ns[i, k:] = 0
                break
            births = rloc.poisson(lam * nprev * dt)
            deaths = rloc.poisson(mu * nprev * dt)
            Ns[i, k] = max(0.0, nprev + births - deaths)
    return t, Ns

# ---------- build two trees ----------
edges_b, nodes_b, alive_b, level_b = layered_tree(depth=4, max_children=(2,3), extinct_prob=0.22)
pos_b = layered_positions(level_b, x_spread=1.8, y_step=1.0)
trait_b = {n: (-level_b[n] + rng.normal(0,0.35)) for n in nodes_b}

edges_p, nodes_p, alive_p, level_p = layered_tree(depth=4, max_children=(2,3), extinct_prob=0.28)
pos_p = layered_positions(level_p, x_spread=1.8, y_step=1.0)
trait_p = {n: (-level_p[n] + rng.normal(0,0.35)) for n in nodes_p}

# ---------- figure layout ----------
fig = plt.figure(figsize=(10,7), dpi=300)
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.12, 2.2], width_ratios=[1,1], hspace=0.35, wspace=0.30)

# Top-left: OU dynamics
ax_ou = fig.add_subplot(gs[0, 0])
t_ou, y_ou = smooth_ou_curve(T=8, dt=0.02, mu=0.0, theta=0.9, sigma=0.6)
ax_ou.plot(t_ou, y_ou, color="black", lw=2)
ax_ou.axhline(0, ls="--", color="0.5", lw=1)
ax_ou.set_title("A. OU dynamics (trait mean reversion)", fontsize=12)
ax_ou.set_xlabel("Time")
ax_ou.set_ylabel("Trait value")
ax_ou.spines[['right','top']].set_visible(False)

# Top-right: Branching process panel (replaces Time arrow)
ax_bp = fig.add_subplot(gs[0, 1])
tb, Nb = branching_paths(T=8, dt=0.02, reps=6, N0=40, lam=0.055, mu=0.045, seed=21)
for i in range(Nb.shape[0]):
    ax_bp.plot(tb, Nb[i], lw=1.7, alpha=0.85)
ax_bp.set_title("B. Branching process (birth–death)", fontsize=12)
ax_bp.set_xlabel("Time")
ax_bp.set_ylabel("Population size")
ax_bp.set_ylim(0, max(60, np.nanmax(Nb)*1.1))
ax_bp.spines[['right','top']].set_visible(False)

# Bottom-left: bacterial lineages (E. coli LTEE)
ax_b = fig.add_subplot(gs[2, 0])
norm_b, cm_b = draw_tree(ax_b, edges_b, alive_b, pos_b, trait_b, cmap="viridis")
ax_b.set_title("C. Bacterial lineages (E. coli LTEE)", fontsize=13)
ax_b.axis("off")

# Bottom-right: pediatric tumor lineages
ax_p = fig.add_subplot(gs[2, 1])
norm_p, cm_p = draw_tree(ax_p, edges_p, alive_p, pos_p, trait_p, cmap="viridis")
ax_p.set_title("D. Pediatric tumor lineages", fontsize=13)
ax_p.axis("off")

# Shared mini legend (extant vs extinct)
legend_ax = fig.add_axes([0.20, 0.47, 0.30, 0.04])
legend_ax.axis("off")
legend_ax.scatter([0.1],[0.5], s=160, color=cm_b(norm_b(0.0)), edgecolor="none")
legend_ax.text(0.12, 0.5, "Extant", va="center", fontsize=10)
legend_ax.scatter([0.50],[0.5], s=160, color=(0.78,0.78,0.78,0.45), edgecolor="none")
legend_ax.text(0.52, 0.5, "Extinct", va="center", fontsize=10)

# Compact vertical colorbar on far right
cax = fig.add_axes([0.93, 0.18, 0.02, 0.58])
sm = plt.cm.ScalarMappable(norm=Normalize(vmin=-4, vmax=0), cmap="viridis")
sm.set_array([])
cb = plt.colorbar(sm, cax=cax)
cb.set_label("Trait / fitness (schematic)")

#fig.suptitle("Linking OU traits and branching to lineage structures", fontsize=15, y=0.98)
plt.savefig("figure1_schematic_ltee_pediatric_lineages.png", dpi=300, bbox_inches="tight")
plt.show()
