"""
Script 2: Weighted Crime Clustering with K-Means
=================================================
Based on: Curtin, Hayslett-McCall & Qiu (2010)

This script:
1. Loads the LA crime dataset (auto-downloads or uses cache)
2. Filters to the study bounding box
3. Applies weighted K-Means (weight = crime_weight / priority_weight)
4. Visualises clusters over beats from Script 1
5. Exports clustered incidents for Script 3 (PPAC optimisation)
"""

import os, json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans

warnings.filterwarnings("ignore")

OUTPUT_DIR    = "/mnt/user-data/outputs/ppac_outputs"
CRIME_CSV_URL = "https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD"
MAX_ROWS      = 150_000
N_CLUSTERS    = 36
RANDOM_STATE  = 42
WEIGHT_COL    = "crime_weight"
SAMPLE_PLOT   = 12_000
DIV_COLORS    = ["#E63946", "#457B9D", "#2A9D8F"]

PRIORITY_MAP  = {1: 1, 2: 3, 3: 6, 4: 12, 5: 25}


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def load_config():
    cfg_path = os.path.join(OUTPUT_DIR, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"[1] Config: {cfg['city']}, {cfg['state']}")
    else:
        cfg = {"bbox": (34.14, 34.22, -118.43, -118.32),
               "city": "Los Angeles", "state": "California",
               "n_divisions": 3, "output_dir": OUTPUT_DIR}
        print("[1] Using default config.")
    return cfg


def synthetic_crime(cfg):
    rng = np.random.default_rng(42)
    min_lat, max_lat, min_lon, max_lon = cfg["bbox"]
    n = 8000
    cls_lat = rng.uniform(min_lat, max_lat, 8)
    cls_lon = rng.uniform(min_lon, max_lon, 8)
    lats, lons, wts = [], [], []
    for cl, co in zip(cls_lat, cls_lon):
        k = n // 8
        lats.extend(rng.normal(cl, 0.008, k))
        lons.extend(rng.normal(co, 0.008, k))
        wts.extend(rng.choice([1,2,3,4,5], k, p=[0.3,0.25,0.2,0.15,0.1]))
    return pd.DataFrame({"LAT": lats, "LON": lons, WEIGHT_COL: wts,
                         "Crm Cd Desc": "SYNTHETIC", "AREA NAME": "Demo",
                         "Part 1-2": np.random.choice([1,2], len(lats))})


def load_crime_data(cfg):
    cache = os.path.join(OUTPUT_DIR, "crime_raw.csv")
    if os.path.exists(cache):
        print(f"[2] Loading cached data: {cache}")
        df = pd.read_csv(cache, low_memory=False)
    else:
        print(f"[2] Downloading crime data …")
        try:
            df = pd.read_csv(CRIME_CSV_URL, nrows=MAX_ROWS, low_memory=False)
            df.to_csv(cache, index=False)
            print(f"    Cached → {cache}")
        except Exception as e:
            print(f"    Download failed ({e}). Using synthetic data.")
            df = synthetic_crime(cfg)

    df.columns = [c.strip() for c in df.columns]
    df["LAT"] = pd.to_numeric(df.get("LAT", pd.Series(dtype=float)), errors="coerce")
    df["LON"] = pd.to_numeric(df.get("LON", pd.Series(dtype=float)), errors="coerce")

    if WEIGHT_COL not in df.columns:
        df[WEIGHT_COL] = df.get("Part 1-2", pd.Series(1, index=df.index)).map({1:5, 2:2}).fillna(1)
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(1).clip(1, 5)
    df["priority_weight"] = df[WEIGHT_COL].round().astype(int).map(PRIORITY_MAP).fillna(1)

    min_lat, max_lat, min_lon, max_lon = cfg["bbox"]
    mask = ((df["LAT"] >= min_lat) & (df["LAT"] <= max_lat) &
            (df["LON"] >= min_lon) & (df["LON"] <= max_lon) &
            (df["LAT"] != 0) & (df["LON"] != 0))
    df = df[mask].dropna(subset=["LAT","LON"]).reset_index(drop=True)
    print(f"    {len(df):,} incidents in bounding box.")
    return df


def weighted_kmeans(df):
    print(f"[3] Weighted K-Means (K={N_CLUSTERS}) …")
    coords  = df[["LAT","LON"]].values
    weights = df["priority_weight"].values.astype(float)

    if len(df) > 40_000:
        km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
                             batch_size=8000, n_init=5)
    else:
        km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
                    n_init=10, max_iter=500)
    km.fit(coords, sample_weight=weights)
    df = df.copy()
    df["cluster"] = km.labels_

    centres = pd.DataFrame(km.cluster_centers_, columns=["centre_lat","centre_lon"])
    centres["cluster"] = range(N_CLUSTERS)
    stats = df.groupby("cluster").agg(
        n_incidents=("LAT","count"),
        total_weight=("priority_weight","sum"),
        mean_weight=("priority_weight","mean"),
    ).reset_index()
    centres = centres.merge(stats, on="cluster")

    print(f"    Inertia: {km.inertia_:,.0f} | "
          f"Sizes {stats['n_incidents'].min()}–{stats['n_incidents'].max()}")
    return df, centres, km


def elbow_plot(df, save_dir):
    print("[4] Elbow analysis …")
    coords  = df[["LAT","LON"]].values
    weights = df["priority_weight"].values.astype(float)
    ks = range(5, 26)
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5, max_iter=200)
        km.fit(coords, sample_weight=weights)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(9,5))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#0f0f23")
    ax.plot(list(ks), inertias, "o-", color="#E63946", linewidth=2.5, markersize=7,
            markerfacecolor="#FFD700")
    ax.axvline(N_CLUSTERS, color="#2A9D8F", linestyle="--", linewidth=1.5,
               label=f"Selected K={N_CLUSTERS}")
    ax.set_xlabel("K (Clusters)", color="#aaaacc", fontsize=11)
    ax.set_ylabel("Weighted Inertia", color="#aaaacc", fontsize=11)
    ax.set_title("Elbow Analysis — Optimal K\n(analogous to choosing P patrol areas in PPAC)",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#aaaacc"); ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.grid(True, color="#2a2a4a", alpha=0.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#3a3a5c")
    out = os.path.join(save_dir, "02_elbow_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close()
    print(f"    Saved → {out}")


def plot_clusters(df, centres, cfg, beat_gdf_path, save_dir):
    print("[5] Plotting cluster map …")
    min_lat, max_lat, min_lon, max_lon = cfg["bbox"]
    cmap = plt.cm.get_cmap("tab20", N_CLUSTERS)

    beat_gdf = None
    if beat_gdf_path and os.path.exists(beat_gdf_path):
        beat_gdf = gpd.read_file(beat_gdf_path)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor("#1a1a2e")

    sample = df.sample(min(SAMPLE_PLOT, len(df)), random_state=42)

    for idx, ax in enumerate(axes):
        ax.set_facecolor("#0f0f23")
        if beat_gdf is not None:
            beat_gdf.boundary.plot(ax=ax, color="#3a3a5c", linewidth=0.6, alpha=0.8)

        if idx == 0:
            sc = ax.scatter(sample["LON"], sample["LAT"], c=sample[WEIGHT_COL],
                            cmap="YlOrRd", s=3, alpha=0.6, linewidths=0)
            cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
            cbar.set_label("Crime Weight", color="white", fontsize=9)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
            ax.set_title(f"{cfg['city']} — Crime by Weight  (n={len(df):,})",
                         color="white", fontsize=11, fontweight="bold")
        else:
            cols = [cmap(i) for i in sample["cluster"]]
            ax.scatter(sample["LON"], sample["LAT"], c=cols, s=4, alpha=0.55, linewidths=0)
            ax.scatter(centres["centre_lon"], centres["centre_lat"],
                       c="white", s=130, marker="*", zorder=6,
                       edgecolors="#FFD700", linewidths=1.2, label="Cluster Centre")
            top5 = centres.nlargest(5, "total_weight")
            for _, r in top5.iterrows():
                ax.annotate(f"C{int(r['cluster'])}({int(r['n_incidents'])})",
                            (r["centre_lon"], r["centre_lat"]),
                            color="#FFD700", fontsize=6.5,
                            xytext=(4,4), textcoords="offset points")
            ax.set_title(f"Weighted K-Means Clusters (K={N_CLUSTERS})\nPriority-scaled weights",
                         color="white", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", loc="upper right")

        ax.set_xlim(min_lon, max_lon); ax.set_ylim(min_lat, max_lat)
        ax.set_xlabel("Longitude", color="#aaaacc", fontsize=9)
        ax.set_ylabel("Latitude",  color="#aaaacc", fontsize=9)
        ax.tick_params(colors="#666688", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#3a3a5c")

    plt.tight_layout(pad=2.5)
    out = os.path.join(save_dir, "02_crime_clusters.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close()
    print(f"    Saved → {out}")


def plot_cluster_stats(centres, cfg, save_dir):
    print("[6] Cluster statistics plot …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a2e")

    top = centres.nlargest(20, "total_weight")

    ax = axes[0]; ax.set_facecolor("#0f0f23")
    ax.barh(top["cluster"].astype(str), top["total_weight"],
            color="#E63946", edgecolor="#FFD700", linewidth=0.5)
    ax.set_xlabel("Total Priority Weight", color="#aaaacc", fontsize=10)
    ax.set_ylabel("Cluster ID", color="#aaaacc", fontsize=10)
    ax.set_title("Top 20 Clusters by Priority Weight\n(Σ a_i y_i objective)",
                 color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.grid(axis="x", color="#2a2a4a", alpha=0.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#3a3a5c")

    ax2 = axes[1]; ax2.set_facecolor("#0f0f23")
    sc = ax2.scatter(centres["n_incidents"], centres["mean_weight"],
                     c=centres["total_weight"], cmap="YlOrRd",
                     s=80, edgecolors="white", linewidths=0.5, zorder=3)
    cbar = plt.colorbar(sc, ax=ax2, shrink=0.8, pad=0.01)
    cbar.set_label("Total Weight", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax2.set_xlabel("Incident Count", color="#aaaacc", fontsize=10)
    ax2.set_ylabel("Mean Weight", color="#aaaacc", fontsize=10)
    ax2.set_title("Cluster Profile: Volume vs Severity\n(workload balance Σ a_i x_j ≤ M_j)",
                  color="white", fontsize=10, fontweight="bold")
    ax2.tick_params(colors="#aaaacc", labelsize=8)
    ax2.grid(True, color="#2a2a4a", alpha=0.4)
    for sp in ax2.spines.values(): sp.set_edgecolor("#3a3a5c")

    plt.tight_layout(pad=2.5)
    out = os.path.join(save_dir, "02_cluster_stats.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close()
    print(f"    Saved → {out}")


def export_for_optimisation(df, centres, save_dir):
    cols = ["LAT","LON", WEIGHT_COL, "priority_weight", "cluster"]
    df[cols].to_csv(os.path.join(save_dir, "crime_clustered.csv"), index=False)
    centres.to_csv(os.path.join(save_dir, "cluster_centres.csv"), index=False)
    print(f"[7] Exported → {save_dir}")


def main():
    ensure_dir(OUTPUT_DIR)
    print("="*60)
    print("  PPAC Script 2: Weighted Crime Clustering")
    print("="*60)
    cfg = load_config()
    ensure_dir(cfg["output_dir"])

    df = load_crime_data(cfg)
    elbow_plot(df, cfg["output_dir"])
    df, centres, _ = weighted_kmeans(df)
    beat_path = os.path.join(cfg["output_dir"], "beats.geojson")
    plot_clusters(df, centres, cfg, beat_path, cfg["output_dir"])
    plot_cluster_stats(centres, cfg, cfg["output_dir"])
    export_for_optimisation(df, centres, cfg["output_dir"])
    print("\n✓ Script 2 complete. Run script3_ppac_optimization.py next.")


if __name__ == "__main__":
    main()