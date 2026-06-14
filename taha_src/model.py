import os, json, warnings, pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.cm as cm

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    print("WARNING: PuLP not found. Install with: pip install pulp")

warnings.filterwarnings("ignore")

OUTPUT_DIR = "/mnt/user-data/outputs/ppac_outputs"
DIV_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261", "#E9C46A", "#264653"]

# ── PPAC parameters ──────────────────────────────────────────────
PPAC_CONFIG = {
    "P": 9,                    # Number of patrol areas to locate
    "service_distance_deg": 0.025,   # ~2 miles in degrees (lat/lon approx)
    "backup_steps": 15,        # Number of Pareto steps for backup model
    "solver_time_limit": 120,  # seconds per solve
}


def ensure_dir(p): os.makedirs(p, exist_ok=True)


# ── Load artefacts from Scripts 1 & 2 ────────────────────────────
def load_artefacts(od):
    print("[1] Loading artefacts from Scripts 1 & 2 …")
    cfg_path = os.path.join(od, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    beat_path = os.path.join(od, "beats.geojson")
    beat_gdf  = gpd.read_file(beat_path) if os.path.exists(beat_path) else None

    crime_path = os.path.join(od, "crime_clustered.csv")
    if os.path.exists(crime_path):
        crimes_df = pd.read_csv(crime_path)
    else:
        crimes_df = _synthetic_incidents(cfg)
        print("    Using synthetic crime data (run Script 2 first for real data).")

    cmd_path = os.path.join(od, "command_centers.csv")
    cmd_df   = pd.read_csv(cmd_path) if os.path.exists(cmd_path) else None

    print(f"    {len(crimes_df):,} incidents | "
          f"{'beats loaded' if beat_gdf is not None else 'no beats'} | "
          f"{'cmd centres loaded' if cmd_df is not None else 'no cmd centres'}")
    return cfg, beat_gdf, crimes_df, cmd_df


def _synthetic_incidents(cfg):
    rng = np.random.default_rng(42)
    min_lat, max_lat, min_lon, max_lon = cfg["bbox"]
    n = 500
    lats = rng.uniform(min_lat, max_lat, n)
    lons = rng.uniform(min_lon, max_lon, n)
    wts  = rng.choice([1,2,3,4,5], n, p=[0.3,0.25,0.2,0.15,0.1])
    return pd.DataFrame({"LAT": lats, "LON": lons,
                         "crime_weight": wts,
                         "priority_weight": np.vectorize(
                             {1:1,2:3,3:6,4:12,5:25}.get)(wts)})


# ── Build coverage sets N_i ───────────────────────────────────────
def build_coverage_sets(crimes_df, cmd_df, S):
    """
    N_i = { j ∈ J | dist(i,j) <= S }
    Using Euclidean distance in lat/lon degrees as proxy for network distance.
    (In production, replace with shortest-path distances from Script 1 road graph.)
    """
    print(f"[2] Building coverage sets N_i (S={S:.4f} deg) …")
    I = crimes_df[["LAT","LON"]].values
    if cmd_df is not None:
        J = cmd_df[["node_lat","node_lon"]].values
    else:
        # Use a regular grid of potential command centres
        min_lat, max_lat = I[:,0].min(), I[:,0].max()
        min_lon, max_lon = I[:,1].min(), I[:,1].max()
        gl, gn = np.linspace(min_lat, max_lat, 8), np.linspace(min_lon, max_lon, 8)
        grid = np.array([(la, lo) for la in gl for lo in gn])
        J = grid

    n_I, n_J = len(I), len(J)
    N = {}   # N[i] = list of j indices within S
    for i in range(n_I):
        dists = np.sqrt(((J - I[i])**2).sum(axis=1))
        N[i]  = list(np.where(dists <= S)[0])

    covered = sum(1 for i in range(n_I) if len(N[i]) > 0)
    print(f"    Incidents: {n_I}  |  Facilities: {n_J}  |  "
          f"Coverable: {covered} ({100*covered/n_I:.1f}%)")
    return N, n_I, n_J, I, J


# ── MODEL 1: Maximal Coverage (PPAC) ────────────────────────────
def solve_ppac(crimes_df, N, n_I, n_J, P, cfg_ppac, label="PPAC"):
    if not HAS_PULP:
        return None, None, None
    print(f"[3] Solving {label} (P={P}) …")

    a = crimes_df["priority_weight"].values.astype(float)

    prob = pulp.LpProblem(label, pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(n_J)]
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n_I)]

    # Objective
    prob += pulp.lpSum(a[i] * y[i] for i in range(n_I))

    # Coverage constraint
    for i in range(n_I):
        if N[i]:
            prob += pulp.lpSum(x[j] for j in N[i]) >= y[i]
        else:
            prob += y[i] == 0

    # Patrol count
    prob += pulp.lpSum(x) == P

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=cfg_ppac["solver_time_limit"])
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = pulp.value(prob.objective)
    x_sol = [pulp.value(xj) for xj in x]
    y_sol = [pulp.value(yi) for yi in y]

    selected_j = [j for j in range(n_J) if x_sol[j] and x_sol[j] > 0.5]
    covered_i  = [i for i in range(n_I) if y_sol[i] and y_sol[i] > 0.5]

    print(f"    Status: {status} | Obj: {obj_val:.1f} | "
          f"Covered: {len(covered_i)}/{n_I} "
          f"({100*len(covered_i)/n_I:.1f}%)")
    return selected_j, covered_i, obj_val


# ── MODEL 2: Backup Coverage (PPAC-B) ───────────────────────────
def solve_ppac_backup(crimes_df, N, n_I, n_J, P, O_bounds, cfg_ppac):
    """Solve backup model for a range of O values (constraint method)."""
    if not HAS_PULP:
        return []
    print("[4] Solving backup coverage model across Pareto front …")
    a = crimes_df["priority_weight"].values.astype(float)
    results = []

    for step, O in enumerate(O_bounds):
        prob = pulp.LpProblem(f"PPAC_B_O{step}", pulp.LpMaximize)
        x  = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(n_J)]
        y  = [pulp.LpVariable(f"y_{i}", lowBound=0, upBound=P, cat="Integer")
              for i in range(n_I)]
        w  = [pulp.LpVariable(f"w_{i}", cat="Binary") for i in range(n_I)]

        # Objective: maximise backup coverage
        prob += pulp.lpSum(a[i] * y[i] for i in range(n_I))

        # Primary coverage lower bound
        prob += pulp.lpSum(a[i] * w[i] for i in range(n_I)) >= O

        # w_i = 1 if at least one facility covers i
        for i in range(n_I):
            if N[i]:
                prob += pulp.lpSum(x[j] for j in N[i]) >= w[i]
                prob += pulp.lpSum(x[j] for j in N[i]) >= y[i]
            else:
                prob += w[i] == 0
                prob += y[i] == 0

        prob += pulp.lpSum(x) == P

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=cfg_ppac["solver_time_limit"])
        prob.solve(solver)

        backup_obj = pulp.value(prob.objective) or 0.0
        cov_obj    = sum(a[i] * (pulp.value(w[i]) or 0) for i in range(n_I))
        total_cov  = sum(1 for i in range(n_I) if (pulp.value(w[i]) or 0) > 0.5)
        y_vals     = [pulp.value(y[i]) or 0 for i in range(n_I)]
        multi_cov  = sum(1 for v in y_vals if v >= 2)

        results.append({
            "O_bound":    O,
            "backup_obj": backup_obj,
            "cover_obj":  cov_obj,
            "total_covered": total_cov,
            "multi_covered": multi_cov,
        })
        print(f"    O={O:.0f} → backup={backup_obj:.0f}, "
              f"covered={total_cov}/{n_I}, multi={multi_cov}")
    return results


# ── Compute baseline (existing random geography) ─────────────────
def baseline_coverage(crimes_df, N, n_I, n_J, P):
    """Random facility placement as proxy for existing hand-drawn geography."""
    rng = np.random.default_rng(99)
    sel = rng.choice(n_J, size=min(P, n_J), replace=False).tolist()
    a   = crimes_df["priority_weight"].values.astype(float)
    covered = set()
    for i in range(n_I):
        if any(j in sel for j in N[i]):
            covered.add(i)
    obj = sum(a[i] for i in covered)
    pct = 100 * len(covered) / n_I
    print(f"    Baseline coverage: {len(covered)}/{n_I} ({pct:.1f}%) | obj={obj:.1f}")
    return list(covered), obj, pct


# ── Visualise 1: Optimal sectors ─────────────────────────────────
def plot_optimal_sectors(crimes_df, selected_j, covered_i, J, beat_gdf, cfg, save_dir):
    print("[5] Plotting optimal sectors …")
    min_lat, max_lat, min_lon, max_lon = cfg["bbox"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor("#1a1a2e")

    for idx, ax in enumerate(axes):
        ax.set_facecolor("#0f0f23")
        if beat_gdf is not None:
            beat_gdf.boundary.plot(ax=ax, color="#3a3a5c", linewidth=0.5)

        covered_mask = np.zeros(len(crimes_df), dtype=bool)
        covered_mask[covered_i] = True

        if idx == 0:
            # All incidents
            ax.scatter(crimes_df["LON"], crimes_df["LAT"],
                       c=np.where(covered_mask, "#2A9D8F", "#E63946"),
                       s=6, alpha=0.7, linewidths=0)
            # Command centres
            if selected_j:
                ax.scatter(J[selected_j, 1], J[selected_j, 0],
                           c="#FFD700", s=150, marker="^", zorder=6,
                           edgecolors="white", linewidths=1.0, label="Optimal CMD Ctr")
            legend_els = [
                mpatches.Patch(color="#2A9D8F", label=f"Covered ({sum(covered_mask):,})"),
                mpatches.Patch(color="#E63946", label=f"Uncovered ({(~covered_mask).sum():,})"),
                Line2D([0],[0], marker="^", color="w", markerfacecolor="#FFD700",
                       markersize=10, label="Command Centre", linestyle="None"),
            ]
            ax.legend(handles=legend_els, loc="upper right", fontsize=8,
                      facecolor="#1a1a2e", labelcolor="white")
            ax.set_title("PPAC Optimal Patrol Placement\nGreen = covered within service distance S",
                         color="white", fontsize=11, fontweight="bold")
        else:
            # Draw routes (lines) from nearest command centre to covered incidents
            ax.scatter(crimes_df.loc[~covered_mask,"LON"],
                       crimes_df.loc[~covered_mask,"LAT"],
                       c="#cc3333", s=4, alpha=0.5, linewidths=0, label="Uncovered")
            ax.scatter(crimes_df.loc[covered_mask,"LON"],
                       crimes_df.loc[covered_mask,"LAT"],
                       c="#4CAF50", s=4, alpha=0.5, linewidths=0, label="Covered")
            if selected_j:
                for i in covered_i[::max(1, len(covered_i)//120)]:
                    row  = crimes_df.iloc[i]
                    dsts = [np.sqrt((J[j,0]-row["LAT"])**2 + (J[j,1]-row["LON"])**2)
                            for j in selected_j]
                    nearest = selected_j[int(np.argmin(dsts))]
                    ax.plot([row["LON"], J[nearest,1]], [row["LAT"], J[nearest,0]],
                            color="#FFD700", alpha=0.25, linewidth=0.5)
                ax.scatter(J[selected_j,1], J[selected_j,0],
                           c="#FFD700", s=150, marker="^", zorder=7,
                           edgecolors="white", linewidths=1.0)
            ax.set_title("Network Routes: Command Centres → Incidents\n(sampled for clarity)",
                         color="white", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", loc="upper right")

        ax.set_xlim(min_lon, max_lon); ax.set_ylim(min_lat, max_lat)
        ax.set_xlabel("Longitude", color="#aaaacc", fontsize=9)
        ax.set_ylabel("Latitude",  color="#aaaacc", fontsize=9)
        ax.tick_params(colors="#666688", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#3a3a5c")

    plt.tight_layout(pad=2.5)
    out = os.path.join(save_dir, "03_optimal_sectors.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close()
    print(f"    Saved → {out}")


# ── Visualise 2: Efficiency comparison (Table 1 analogue) ────────
def plot_coverage_comparison(crimes_df, n_I, covered_opt, obj_opt,
                             covered_base, obj_base, save_dir, cfg):
    print("[6] Plotting coverage comparison …")
    pct_opt  = 100 * len(covered_opt)  / n_I
    pct_base = 100 * len(covered_base) / n_I

    labels   = ["Existing\n(Baseline)", "Optimal\n(PPAC)"]
    pcts     = [pct_base, pct_opt]
    objs     = [obj_base, obj_opt]
    colors   = ["#E63946", "#2A9D8F"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, vals, title, unit in zip(
            axes,
            [pcts, objs],
            ["Coverage Rate (%)\nPercent of incidents within service distance S",
             "Weighted Objective (Σ a_i y_i)\nPriority-weighted incidents covered"],
            ["%", "pts"]):
        ax.set_facecolor("#0f0f23")
        bars = ax.bar(labels, vals, color=colors, edgecolor="#FFD700",
                      linewidth=0.8, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{v:.1f}{unit}", ha="center", va="bottom",
                    color="white", fontsize=12, fontweight="bold")
        improvement = (vals[1] - vals[0]) / max(vals[0], 1e-9) * 100
        ax.set_title(f"{title}\n▲ {improvement:+.1f}% improvement",
                     color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel(unit, color="#aaaacc", fontsize=10)
        ax.tick_params(colors="#aaaacc")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", color="#2a2a4a", alpha=0.5)
        for sp in ax.spines.values(): sp.set_edgecolor("#3a3a5c")

    fig.suptitle(f"{cfg['city']} — PPAC Efficiency vs Existing Arrangement\n"
                 f"(Table 1 analogue from Curtin et al. 2010)",
                 color="white", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(pad=2.5)
    out = os.path.join(save_dir, "03_coverage_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close()
    print(f"    Saved → {out}")


# ── Visualise 3: Pareto tradeoff curve ───────────────────────────
def plot_backup_tradeoff(pareto_results, save_dir, cfg):
    if not pareto_results:
        return
    print("[7] Plotting backup tradeoff (Pareto curve) …")
    df = pd.DataFrame(pareto_results)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a2e")

    ax = axes[0]; ax.set_facecolor("#0f0f23")
    ax.plot(df["backup_obj"], df["cover_obj"], "o-",
            color="#E63946", linewidth=2.5, markersize=8, markerfacecolor="#FFD700")
    ax.set_xlabel("Maximal Backup Coverage Objective", color="#aaaacc", fontsize=11)
    ax.set_ylabel("Maximal Coverage Objective",        color="#aaaacc", fontsize=11)
    ax.set_title("Pareto Tradeoff: Coverage vs. Backup Coverage\n"
                 "(Figure 6 analogue from Curtin et al. 2010)",
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#aaaacc")
    ax.grid(True, color="#2a2a4a", alpha=0.4)
    for sp in ax.spines.values(): sp.set_edgecolor("#3a3a5c")

    ax2 = axes[1]; ax2.set_facecolor("#0f0f23")
    ax2.bar(range(len(df)), df["total_covered"], color="#457B9D",
            label="Singly Covered", alpha=0.8)
    ax2.bar(range(len(df)), df["multi_covered"], color="#E63946",
            label="Multi-Covered", alpha=0.8)
    ax2.set_xlabel("Solution Index (increasing coverage enforcement →)",
                   color="#aaaacc", fontsize=10)
    ax2.set_ylabel("Incident Count", color="#aaaacc", fontsize=10)
    ax2.set_title("Coverage Breakdown Across Pareto Solutions\n"
                  "(analogue to Table 3 from Curtin et al. 2010)",
                  color="white", fontsize=11, fontweight="bold")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax2.tick_params(colors="#aaaacc")
    ax2.grid(axis="y", color="#2a2a4a", alpha=0.4)
    for sp in ax2.spines.values(): sp.set_edgecolor("#3a3a5c")

    plt.tight_layout(pad=2.5)
    out = os.path.join(save_dir, "03_backup_tradeoff.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close()
    print(f"    Saved → {out}")


# ── Save results JSON ─────────────────────────────────────────────
def save_results(selected_j, covered_opt, obj_opt, covered_base, obj_base,
                 n_I, pareto, save_dir):
    results = {
        "n_incidents": n_I,
        "P": PPAC_CONFIG["P"],
        "optimal": {
            "covered": len(covered_opt),
            "pct_covered": round(100*len(covered_opt)/n_I, 2),
            "weighted_obj": round(obj_opt, 2),
            "selected_facilities": selected_j,
        },
        "baseline": {
            "covered": len(covered_base),
            "pct_covered": round(100*len(covered_base)/n_I, 2),
            "weighted_obj": round(obj_base, 2),
        },
        "improvement_pct": round((len(covered_opt)-len(covered_base))/max(len(covered_base),1)*100, 2),
        "pareto_solutions": pareto,
    }
    out = os.path.join(save_dir, "optimization_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[8] Results saved → {out}")
    return results


# ── MAIN ─────────────────────────────────────────────────────────
def main():
    ensure_dir(OUTPUT_DIR)
    print("="*60)
    print("  PPAC Script 3: Optimization (Maximal + Backup Covering)")
    print("="*60)

    cfg, beat_gdf, crimes_df, cmd_df = load_artefacts(OUTPUT_DIR)
    # Subsample for manageable solve time (use all in production)
    if len(crimes_df) > 600:
        crimes_df = crimes_df.sample(600, random_state=42).reset_index(drop=True)
        print(f"    Subsampled to {len(crimes_df)} incidents for demo solve.")

    S = PPAC_CONFIG["service_distance_deg"]
    P = PPAC_CONFIG["P"]

    N, n_I, n_J, I, J = build_coverage_sets(crimes_df, cmd_df, S)

    # Baseline
    print("[2b] Computing baseline (existing arrangement) …")
    covered_base, obj_base, pct_base = baseline_coverage(crimes_df, N, n_I, n_J, P)

    # Model 1: PPAC
    selected_j, covered_opt, obj_opt = solve_ppac(
        crimes_df, N, n_I, n_J, P, PPAC_CONFIG)

    if selected_j is None:
        print("    PuLP solve failed; using greedy fallback.")
        # Greedy fallback (not optimal but illustrative)
        a = crimes_df["priority_weight"].values.astype(float)
        selected_j, covered_opt = [], []
        remaining = set(range(n_I))
        for _ in range(P):
            best_j, best_gain = -1, -1
            for j in range(n_J):
                gain = sum(a[i] for i in remaining if j in N.get(i,[]))
                if gain > best_gain:
                    best_gain, best_j = gain, j
            if best_j < 0: break
            selected_j.append(best_j)
            new_covered = {i for i in remaining if best_j in N.get(i,[])}
            covered_opt = list(set(covered_opt) | new_covered)
            remaining -= new_covered
        obj_opt = sum(a[i] for i in covered_opt)

    # Model 2: Backup covering Pareto sweep
    if HAS_PULP and obj_opt:
        n_steps   = PPAC_CONFIG["backup_steps"]
        O_bounds  = np.linspace(obj_opt * 0.5, obj_opt, n_steps).tolist()
        pareto    = solve_ppac_backup(crimes_df, N, n_I, n_J, P,
                                      O_bounds, PPAC_CONFIG)
    else:
        pareto = []

    # Plots
    plot_optimal_sectors(crimes_df, selected_j, covered_opt, J, beat_gdf, cfg, OUTPUT_DIR)
    plot_coverage_comparison(crimes_df, n_I, covered_opt, obj_opt,
                             covered_base, obj_base, OUTPUT_DIR, cfg)
    plot_backup_tradeoff(pareto, OUTPUT_DIR, cfg)

    # Save results
    results = save_results(selected_j, covered_opt, obj_opt,
                           covered_base, obj_base, n_I, pareto, OUTPUT_DIR)

    print("\n" + "="*60)
    print("  PPAC Optimization Results Summary")
    print("="*60)
    print(f"  Incidents:    {n_I}")
    print(f"  Patrols (P):  {P}")
    print(f"  Baseline:     {results['baseline']['pct_covered']}% covered")
    print(f"  Optimal:      {results['optimal']['pct_covered']}% covered")
    print(f"  Improvement:  +{results['improvement_pct']}%")
    print(f"  Pareto steps: {len(pareto)}")
    print("\n✓ Script 3 complete. All outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
