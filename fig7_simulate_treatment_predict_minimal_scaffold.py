#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 23:10:09 2025

@author: seung-hwan.kim
"""

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#Figure 7 Hybrid OU–Branching Framework (Panels A–D)
#Generates: fig7A_schematic.png,fig7B_phenotype.png, fig7C_population.png,fig7D_lineage.png,fig7_Full_final.tiff

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import networkx as nx
from pathlib import Path

# -----------------------------
# 0) Utilities
# -----------------------------
def ensure_dir(p): Path(p).parent.mkdir(parents=True, exist_ok=True)

def env_from_paths(X, q=(0.025, 0.5, 0.975)):
    X = np.asarray(X)
    lo, med, hi = np.nanquantile(X, q, axis=0)
    return lo, med, hi

def iqr_from_paths(X):
    return env_from_paths(X, q=(0.25, 0.5, 0.75))

# PK exposure curve for a time grid
def exposure_curve(t_grid, doses, ke=0.10):
    C = np.zeros_like(t_grid, dtype=float)
    doses = doses or []
    for td, D in doses:
        C += (t_grid >= td) * D * np.exp(-ke * np.maximum(0.0, t_grid - td))
    return C

# Exact OU transition (vectorized)
def ou_exact_step(y_prev, dt, mu, theta, sigma, rng):
    e = np.exp(-theta * dt)
    m = mu + (y_prev - mu) * e
    v = (sigma**2 / (2.0 * theta)) * (1.0 - e**2)
    v = np.maximum(v, 1e-18)
    return rng.normal(m, np.sqrt(v))

# -----------------------------
# 1) Hybrid model simulator
# -----------------------------
def simulate_regimen(
    T=200, dt=0.5, doses=None, ke=0.10,
    # OU baseline
    mu0=1.555009e-05, th0=1.136503e-01, sg0=7.751312e-06,
    # OU modulation by drug (g(C)=C/(C+EC50))
    mu_drug=-0.35, th_drug=0.50, sg_drug=-0.40, EC50=75.0,
    # Demography baseline
    lam0=0.020, mu_base=0.015, alpha=0.9, beta=0.7,
    # Drug effect on demography
    k_lam=1.2, k_mu=1.0,
    # Initial conditions & cohort
    y0=0.10, n0=5e7, reps=256, seed=1234
):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T + dt, dt)
    if doses is None:
        doses = [(d, 100.0) for d in np.arange(0, 6*21, 21)]  # 6 cycles q21d
    C = exposure_curve(t, doses, ke=ke)
    g = C / (C + EC50 + 1e-12)

    # OU parameters over time (piecewise-constant per step)
    mu_t = mu0 + mu_drug * g
    th_t = np.clip(th0 + th_drug * g, 1e-6, None)
    sg_t = np.clip(sg0 * np.exp(sg_drug * g), 1e-6, None)

    # Arrays
    Y = np.empty((reps, t.size), dtype=float)
    N = np.empty((reps, t.size), dtype=float)
    Y[:, 0] = y0
    N[:, 0] = n0

    # PD map reused for demography
    def g_c(c): return c / (c + EC50 + 1e-12)
    def rates(y, c):
        hh = g_c(c)
        lam = lam0    * np.exp(alpha * y - k_lam * hh)
        mud = mu_base * np.exp(-beta * y + k_mu  * hh)
        lam = np.clip(lam, 0.0, 1.2)
        mud = np.clip(mud, 0.0, 1.2)
        return lam, mud

    # Time-stepping
    for k in range(1, t.size):
        dtk = dt
        # OU exact step (vectorized)
        e = np.exp(-th_t[k-1] * dtk)
        m = mu_t[k-1] + (Y[:, k-1] - mu_t[k-1]) * e
        v = (sg_t[k-1]**2 / (2.0 * th_t[k-1])) * (1.0 - e**2)
        v = np.maximum(v, 1e-18)
        Y[:, k] = rng.normal(m, np.sqrt(v))

        # Branching (diffusion on log N for numerical stability)
        lam, mud = rates(Y[:, k], C[k])
        net = (lam - mud) * dtk
        var_dem = (lam + mud) * dtk / np.maximum(N[:, k-1], 1.0)
        var_dem = np.clip(var_dem, 0.0, 0.25)
        z = rng.normal(0.0, 1.0, size=Y.shape[0])
        lnN = np.log(np.maximum(N[:, k-1], 1.0)) + net - 0.5*var_dem + np.sqrt(var_dem)*z
        N[:, k] = np.maximum(np.exp(lnN), 0.0)

    return t, C, Y, N

# -----------------------------
# 2) Figure 7A — Phenotype trajectories + exposure
# -----------------------------
def fig7A_phenotype(t, C, Y, outfile="fig7A_phenotype.png"):
    lo, med, hi = env_from_paths(Y)
    fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
    ax1.fill_between(t, lo, hi, color="#bdd7e7", alpha=0.65, label="95% envelope")
    ax1.plot(t, med, color="#08519c", lw=2.2, label="Median $Y_t$")
    ax1.axhline(0.0, ls="--", color="0.6", lw=1.0)
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Phenotype $Y_t$")

    ax2 = ax1.twinx()
    ax2.plot(t, C, color="#ef3b2c", lw=2.0, alpha=0.9, label="Exposure $C(t)$")
    ax2.set_ylabel("Drug exposure $C(t)$")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, frameon=False, loc="upper right")

    #fig.suptitle("Phenotypic evolution under cyclic therapy", y=0.98, fontsize=13)
    fig.tight_layout()
    ensure_dir(outfile); fig.savefig(outfile, dpi=1500, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# 3) Figure 7B — Population dynamics (treated vs control)
# -----------------------------
def fig7B_population(t_treated, N_treated, sim_kwargs, outfile="fig7B_population.png"):
    # Simulate control (no doses) with matched settings
    ctrl_kwargs = sim_kwargs.copy()
    ctrl_kwargs["doses"] = []
    t0, C0, Y0, N0 = simulate_regimen(**ctrl_kwargs)

    lo_t, med_t, hi_t = iqr_from_paths(N_treated)
    _, med_0, _ = iqr_from_paths(N0)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.fill_between(t_treated, lo_t, hi_t, color="#fcbba1", alpha=0.50, label="Treated IQR")
    ax.plot(t_treated, med_t, color="#cb181d", lw=2.2, label="Treated median")
    ax.plot(t0, med_0, color="0.25", lw=2.2, ls="--", label="Control median")
    ax.set_yscale("log")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Tumor cells $N_t$ (log scale)")
    #ax.set_title("Tumor population dynamics under therapy")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    ensure_dir(outfile); fig.savefig(outfile, dpi=1500, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# 4) Figure 7C — Conceptual schematic (improved)
# -----------------------------
def rbox(ax, xy, w, h, fc, ec="black", lw=1.5, radius=0.06, text="", fontsize=12, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        ec=ec, fc=fc, lw=lw
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=("bold" if bold else None))
    return patch

def add_arrow(ax, xy1, xy2, color="black", text=None, fontsize=11, style="-|>", lw=1.8, ms=12):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, lw=lw,
                        color=color, mutation_scale=ms,
                        connectionstyle="arc3,rad=0.0")
    ax.add_patch(a)
    if text:
        xm, ym = (xy1[0]+xy2[0])/2, (xy1[1]+xy2[1])/2
        ax.text(xm, ym + 0.15, text, color=color, fontsize=fontsize,
                ha="center", va="bottom", fontweight="regular")

def fig7C_schematic(outfile="fig7C_schematic_centered.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # ---------- Title (regular font) ----------
    #ax.text(0.5, 6.75, "Conceptual schematic of hybrid OU–branching framework", fontsize=15, ha="left")

    # ---------- Therapy schedule ----------
    sched_x, sched_y, sched_w, sched_h = 0.5, 5.70, 11.0, 0.60
    ax.text(0.5, 6.45, "Therapy schedule", fontsize=16)
    rbox(ax, (sched_x, sched_y), sched_w, sched_h, fc="#e8e8e8")
    dose_lines = np.linspace(sched_x + 1.0, sched_x + sched_w - 0.2, 6)
    for x in dose_lines:
        ax.plot([x, x], [sched_y, sched_y + sched_h], color="#b30000", lw=2.0)
    # q21d slightly left inside the last box
    ax.text(sched_x + sched_w - 0.5, sched_y + sched_h/2,
            "q21d", color="#b30000", fontsize=12,
            ha="center", va="center")

    # ---------- Drug exposure line ----------
    ax.text(0.5, 5.25, "Drug exposure  C(t)", fontsize=14)
    ax.plot([0.8, 11.0], [5.05, 5.05], color="#e41a1c", lw=2.2)
    add_arrow(ax, (11.0, 5.05), (11.0, 5.45), color="#e41a1c", text="C(t)", fontsize=12)

    # ---------- Headings ----------
    ax.text(0.7, 4.25, "Trait dynamics (OU)", fontsize=14)
    ax.text(6.5, 4.25, "Population dynamics (branching)", fontsize=14)

    # ---------- Main boxes (same size) ----------
    box_w, box_h = 5.2, 1.4
    ou_x, ou_y = 0.6, 2.75
    br_x, br_y = 6.2, 2.75

    ou_eq = (r"$\mathrm{d}Y_t=\theta(\mu-Y_t)\,\mathrm{d}t+\sigma\,\mathrm{d}W_t$" "\n"
             r"$\theta,\mu,\sigma$ modulated by $C(t)$")
    pop_eq = (r"$\mathrm{d}N_t=[\lambda(Y_t,C)-\mu(Y_t,C)]N_t\,\mathrm{d}t$" "\n"
              r"$+\sqrt{V(Y_t,C,N_t)}\,\mathrm{d}W_t'$")

    rbox(ax, (ou_x, ou_y), box_w, box_h, fc="#deebf7", text=ou_eq, fontsize=13)
    rbox(ax, (br_x, br_y), box_w, box_h, fc="#fee0d2", text=pop_eq, fontsize=13)

    # ---------- Blue arrow OU -> Branching ----------
    add_arrow(ax,
              (ou_x + box_w, ou_y + box_h/2),
              (br_x,          br_y + box_h/2),
              color="#2171b5", lw=2.8, ms=18)

    # ---------- Feedback arrow ----------
    add_arrow(ax,
              (br_x + box_w + 0.25, br_y + box_h + 0.25),
              (br_x + box_w + 0.25, br_y),
              color="#666666", style="<|-|>", lw=1.6, ms=11)
    ax.text(br_x + box_w + 0.35, br_y + box_h/2,
            "feedback\n(microenv.)", fontsize=12, color="#666",
            ha="left", va="center")

    # ---------- Outcomes / Readouts (CENTERED) ----------
    out_text = ("Outcomes / Readouts\n"
                r"• $P(\mathrm{cure})$ (low-$N_t$ crossing)" "\n"
                r"• Time to progression (TTP)" "\n"
                r"• Post-therapy phenotype $Y_t$" "\n"
                r"• Residual burden $N_t$")
    # center it horizontally under both main boxes
    out_w, out_h = 3.5, 1.5
    out_x = (ou_x + br_x + box_w - out_w) / 2  # centered midpoint
    out_y = 0.85
    rbox(ax, (out_x, out_y), out_w, out_h, fc="#ffffff", text=out_text, fontsize=12)

    # ---------- Save ----------
    fig.tight_layout()
    ensure_dir(outfile)
    fig.savefig(outfile, dpi=1500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved centered schematic to {outfile}")

# -----------------------------
# 5) Figure 7D — Lineage simulation & plot
# -----------------------------
def simulate_lineage(
    T=40, dt=0.05, snapshot_frac=0.50,
    # OU
    muY=1.555009e-05, theta=1.136503e-01, sigma=7.751312e-06,
    # Demography
    lam0=0.30, mu0=0.12, alpha=0.8, beta=0.6,
    N0=1, Y0=-0.30, MAX_N=10000, seed=7
):
    rng = np.random.default_rng(seed)
    t_grid = np.arange(0, T+dt, dt)
    t_snap = snapshot_frac * T

    # cell objects
    cells = [dict(y=Y0, alive=True, birth=0.0, death=np.nan, parent=None)]
    edges = []  # tuples (parent, child, lam_at_div, t_div)
    next_id = 1

    def ou_step(y_prev):
        e = np.exp(-theta*dt); m = muY + (y_prev - muY)*e
        v = (sigma**2/(2*theta))*(1 - e**2); v = max(v, 1e-12)
        return rng.normal(m, np.sqrt(v))

    def rates_from_y(y):
        lam = lam0*np.exp(alpha*(y - muY))
        mud = mu0*np.exp(-beta*(y - muY))
        return np.clip(lam, 0.0, 5.0), np.clip(mud, 0.0, 5.0)

    for t in t_grid[1:]:
        alive_ids = [i for i,c in enumerate(cells) if c["alive"]]
        if not alive_ids: break

        # OU updates
        for cid in alive_ids:
            cells[cid]["y"] = ou_step(cells[cid]["y"])

        # Birth-death
        for cid in alive_ids:
            c = cells[cid]
            if not c["alive"]: continue
            lam, mud = rates_from_y(c["y"])

            # births
            n_birth = rng.poisson(lam*dt)
            for _ in range(n_birth):
                if len(cells) >= MAX_N: break
                next_id += 1
                cells.append(dict(y=c["y"], alive=True, birth=t, death=np.nan, parent=cid+1))
                edges.append((cid+1, next_id, lam, t))

            # death
            if rng.uniform() < (1 - np.exp(-mud*dt)):
                c["alive"] = False
                c["death"] = t

        if len(cells) >= MAX_N:
            break

    return dict(cells=cells, edges=edges, t_snap=t_snap)

def draw_lineage(sim, outfile="fig7D_lineage.png", seed=7):
    cells, edges, t_snap = sim["cells"], sim["edges"], sim["t_snap"]

    # extant at snapshot
    extant = []
    for c in cells:
        alive_at = (c["birth"] <= t_snap) and (np.isnan(c["death"]) or (c["death"] > t_snap)) and c["alive"]
        extant.append(alive_at)

    # Build graph
    if len(edges) == 0:
        G = nx.DiGraph()
        G.add_node(1, y=cells[0]["y"], extant=extant[0])
    else:
        G = nx.DiGraph()
        for i,c in enumerate(cells, start=1):
            G.add_node(i, y=c["y"], extant=extant[i-1])
        for (p, ch, lam, td) in edges:
            G.add_edge(p, ch, lam=lam, t=td)

    # Colors by phenotype for extant, grey for extinct
    y_vals = np.array([G.nodes[n]["y"] for n in G.nodes()])
    y_ext = np.array([G.nodes[n]["y"] for n in G.nodes() if G.nodes[n]["extant"]])
    if y_ext.size >= 2 and np.isfinite(y_ext).all():
        vmin, vmax = np.quantile(y_ext, [0.05, 0.95])
    else:
        vmin, vmax = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = -0.5, 0.5

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    node_colors = []
    for n in G.nodes():
        if G.nodes[n]["extant"]:
            node_colors.append(cmap(norm(G.nodes[n]["y"])))
        else:
            node_colors.append((0.75,0.75,0.75,0.6))

    # Edges darker if both endpoints extant; width by lam
    edge_colors, edge_widths = [], []
    if G.number_of_edges() > 0:
        lam_vals = np.array([G.edges[e]["lam"] for e in G.edges()])
        lam_min, lam_max = np.nanmin(lam_vals), np.nanmax(lam_vals)
        rng = max(lam_max - lam_min, 1e-12)
        for e in G.edges():
            u, v = e
            both_ext = G.nodes[u]["extant"] and G.nodes[v]["extant"]
            edge_colors.append((0.25,0.25,0.25,0.75) if both_ext else (0.6,0.6,0.6,0.35))
            lam = G.edges[e]["lam"]
            edge_widths.append(0.6 + 3.0 * (lam - lam_min)/rng)
    else:
        edge_colors, edge_widths = None, 0.6

    # Layout
    pos = nx.spring_layout(G, seed=seed, k=None, iterations=200)

    # Draw
    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.88, 0.12], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=16, node_color=node_colors, linewidths=0.0)

    # Colorbar
    axc = fig.add_subplot(gs[0, 1])
    axc.set_title("Phenotype $Y(t)$", fontsize=10, pad=8)
    cb1 = mpl.colorbar.ColorbarBase(axc, cmap=cmap, norm=norm, orientation='vertical')
    cb1.ax.tick_params(labelsize=9)

    #fig.suptitle("Lineage graph under therapy (OU–branching)", fontsize=13, y=0.98)
    ensure_dir(outfile); fig.savefig(outfile, dpi=1500, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# 6) Composite: assemble panels A–D into a single TIFF
# -----------------------------
def assemble_tiff(pA, pB, pC, pD, outfile="fig7_Full_final_tight.tiff"):
    """
    Assemble 4 panels into a single composite figure in the following order:
    A. Conceptual schematic of hybrid OU–branching framework
    B. Phenotypic evolution (priA) under cyclic therapy
    C. Tumor population dynamics (priA) under therapy
    D. Lineage graph (priA) under therapy
    Adjusted for minimal vertical spacing and perfect C/D alignment.
    """
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(14, 9.5))
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[1, 1],
        width_ratios=[1, 1],
        wspace=0.14,     # horizontal space between A/B or C/D
        hspace=0.005      # reduced vertical space between top and bottom rows
    )

    panels = [
        (pA, "A. Conceptual schematic of hybrid OU–branching framework"),
        (pB, "B. Phenotypic evolution (priA) under cyclic therapy"),
        (pC, "C. Tumor population dynamics (priA) under therapy"),
        (pD, "D. Lineage graph (priA) under therapy")
    ]

    axes = []
    for idx, (panel, title) in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        img = mpimg.imread(panel)
        ax.imshow(img, aspect="equal")
        # Defalt padding for A, B, C
        pad_value = 6
        # Lower the title of panel D slightly
        if title.startswith("D."):
            pad_value = -2 #move it visually downward
        ax.set_title(title, loc="left", fontsize=13, pad=pad_value)
        ax.axis("off")
        axes.append(ax)

    # ---- Manually align Panel C upward to match D ----
    box_c = axes[2].get_position()
    axes[2].set_position([box_c.x0, box_c.y0 + 0.012, box_c.width, box_c.height])  # small upward nudge

    # ---- Fine-tune global layout margins ----
    plt.subplots_adjust(left=0.035, right=0.965, top=0.97, bottom=0.035,
                        wspace=0.14, hspace=0.06)

    ensure_dir(outfile)
    fig.savefig(outfile, dpi=1500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved compact and aligned 4-panel figure to {outfile}")

# -----------------------------
# 7) Run all
# -----------------------------
if __name__ == "__main__":
    # Base regimen for simulations (as before)
    doses = [(d, 100.0) for d in np.arange(0, 6*21, 21)]
    common = dict(
        T=200, dt=0.5, doses=doses, ke=0.10,
        mu0=1.555009e-05, th0=1.136503e-01, sg0=7.751312e-06,
        mu_drug=-0.35, th_drug=0.50, sg_drug=-0.40, EC50=75.0,
        lam0=0.020, mu_base=0.015, alpha=0.9, beta=0.7,
        k_lam=1.2, k_mu=1.0,
        y0=0.10, n0=5e7, reps=256, seed=1234
    )

    # Generate individual panels
    # A — schematic
    fig7C_schematic(outfile="fig7A_schematic.png")

    # Simulate data
    t, C, Y, N = simulate_regimen(**common)

    # B — phenotype trajectories
    fig7A_phenotype(t, C, Y, outfile="fig7B_phenotype.png")

    # C — tumor population dynamics
    fig7B_population(t, N, sim_kwargs=common, outfile="fig7C_population.png")

    # D — lineage
    sim_lineage = simulate_lineage(
        T=40, dt=0.05, snapshot_frac=0.50,
        muY=1.555009e-05, theta=1.136503e-01, sigma=7.751312e-06,
        lam0=0.30, mu0=0.12, alpha=0.8, beta=0.6,
        N0=1, Y0=-0.30, MAX_N=10000, seed=7
    )
    draw_lineage(sim_lineage, outfile="fig7D_lineage.png")

    # Assemble in order A–D
    assemble_tiff("fig7A_schematic.png",
                  "fig7B_phenotype.png",
                  "fig7C_population.png",
                  "fig7D_lineage.png",
                  outfile="fig7_Full_final.tiff")

    print("\nSaved: fig7A_schematic.png, fig7B_phenotype.png, "
          "fig7C_population.png, fig7D_lineage.png, fig7_Full_final.tiff")
