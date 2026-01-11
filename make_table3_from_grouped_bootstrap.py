#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULT_TGRID = [0, 3, 6, 9, 12, 15, 18, 21]
DEFAULT_Y0 = {"WT": -7.080, "priA": -6.993, "recG": -8.325}
DEFAULT_PAIRS = [("priA", "recG"), ("priA", "WT"), ("recG", "WT")]

GENO_CANON = {
    "wt": "WT",
    "pri a": "priA",
    "pria": "priA",
    "rec g": "recG",
    "recg": "recG",
}

def parse_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # flexible name
    if "background" in df.columns and "genotype" not in df.columns:
        df = df.rename(columns={"background": "genotype"})
    if "lineage" in df.columns and "genotype" not in df.columns:
        df = df.rename(columns={"lineage": "genotype"})

    for col in ("genotype", "mu", "theta", "sigma"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # normalize genotype labels
    df["genotype"] = df["genotype"].astype(str).str.strip()
    df["genotype"] = df["genotype"].apply(lambda s: GENO_CANON.get(s.lower(), s))

    # numeric
    for col in ("mu", "theta", "sigma"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["genotype", "mu", "theta", "sigma"]).copy()

    # create draw index within genotype (since no explicit iteration column)
    df = df.sort_values(["genotype"]).copy()
    df["draw"] = df.groupby("genotype").cumcount()

    return df[["genotype", "draw", "mu", "theta", "sigma"]]

def ou_pred_mean_var(t: np.ndarray, y0: float, mu: float, theta: float, sigma: float):
    t = np.asarray(t, dtype=float)
    exp1 = np.exp(-theta * t)
    m = mu + (y0 - mu) * exp1
    v = (sigma**2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * t))
    v = np.maximum(v, 1e-18)
    return m, v

def metrics_pair(mA, vA, mB, vB):
    D_mean = float(np.mean(np.abs(mA - mB)))
    sA, sB = np.sqrt(vA), np.sqrt(vB)
    D_W2 = float(np.mean(np.sqrt((mA - mB) ** 2 + (sA - sB) ** 2)))
    denom = np.sqrt(vA + vB)
    P_A_gt_B = float(np.mean(norm.cdf((mA - mB) / denom)))
    return D_mean, D_W2, P_A_gt_B

def q(series: pd.Series):
    return float(series.quantile(0.5)), float(series.quantile(0.025)), float(series.quantile(0.975))

def build_table3(df: pd.DataFrame, tgrid: List[float], y0: Dict[str, float], pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    t = np.array(tgrid, dtype=float)

    wide = df.pivot_table(index="draw", columns="genotype", values=["mu", "theta", "sigma"], aggfunc="first")

    needed = sorted(set([g for a, b in pairs for g in (a, b)]))
    missing = [g for g in needed if g not in wide.columns.get_level_values(1)]
    if missing:
        raise ValueError(f"Missing genotypes in bootstrap CSV: {missing}. Found: {sorted(set(wide.columns.get_level_values(1)))}")

    cols_needed = [(p, g) for p in ("mu", "theta", "sigma") for g in needed]
    wide = wide[cols_needed].dropna(axis=0).copy()  # keep only draw indices present for all genotypes

    rows = []
    for A, B in pairs:
        dmean_list, dw2_list, p_list, fold_list = [], [], [], []

        for _, row in wide.iterrows():
            muA, thA, sgA = float(row[("mu", A)]), float(row[("theta", A)]), float(row[("sigma", A)])
            muB, thB, sgB = float(row[("mu", B)]), float(row[("theta", B)]), float(row[("sigma", B)])

            mA, vA = ou_pred_mean_var(t, y0[A], muA, thA, sgA)
            mB, vB = ou_pred_mean_var(t, y0[B], muB, thB, sgB)
            D_mean, D_W2, P_A_gt_B = metrics_pair(mA, vA, mB, vB)

            dmean_list.append(D_mean)
            dw2_list.append(D_W2)
            p_list.append(P_A_gt_B)
            fold_list.append(10.0 ** D_mean)  # IMPORTANT: fold per draw then quantiles

        s_dmean = pd.Series(dmean_list)
        s_dw2 = pd.Series(dw2_list)
        s_p = pd.Series(p_list)
        s_fold = pd.Series(fold_list)

        d50, d025, d975 = q(s_dmean)
        w50, w025, w975 = q(s_dw2)
        p50, p025, p975 = q(s_p)
        f50, f025, f975 = q(s_fold)

        rows.append({
            "Pair (A vs B)": f"{A} vs {B}",
            "D_mean_q50": d50, "D_mean_q025": d025, "D_mean_q975": d975,
            "ApproxFold_q50": f50, "ApproxFold_q025": f025, "ApproxFold_q975": f975,
            "D_W2_q50": w50, "D_W2_q025": w025, "D_W2_q975": w975,
            "P(A>B)_q50": p50, "P(A>B)_q025": p025, "P(A>B)_q975": p975,
            "n_draws_used": int(len(s_dmean)),
        })

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--time_grid", default="0,3,6,9,12,15,18,21")
    ap.add_argument("--pairs", default="priA:recG,priA:WT,recG:WT")
    ap.add_argument("--y0", default="WT:-7.080,priA:-6.993,recG:-8.325")
    args = ap.parse_args()

    tgrid = [float(x) for x in args.time_grid.split(",") if x.strip() != ""]
    pairs = [(p.split(":")[0].strip(), p.split(":")[1].strip()) for p in args.pairs.split(",")]
    y0 = {kv.split(":")[0].strip(): float(kv.split(":")[1]) for kv in args.y0.split(",")}

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = parse_csv(Path(args.bootstrap_csv))
    tab = build_table3(df, tgrid=tgrid, y0=y0, pairs=pairs)
    out_csv = outdir / "Table3_pairwise_trajectory_metrics.csv"
    tab.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
