"""
Script 1: Police Beat Generation with Road Network
===================================================
Based on: Curtin, Hayslett-McCall & Qiu (2010)
"Determining Optimal Police Patrol Areas with Maximal Covering and Backup Covering Location Models"

Generates patrol beats for the Hollywood / North Hollywood area of Los Angeles.
Uses OSMnx to download the real road network; falls back to a realistic
synthetic grid-based network when the Overpass API is unavailable.

Outputs (written to ppac_outputs/):
  road_network.pkl        — pickled NetworkX DiGraph
  beats.geojson           — patrol beat polygons (GeoJSON)
  beat_centroids.csv      — beat centroid lat/lon + nearest node ID
  beat_generation.png     — two-panel visualisation
"""

import os
import warnings
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import networkx as nx
import pyproj

from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX =  34.05,  34.30
LON_MIN, LON_MAX = -118.50, -118.30
NUM_BEATS    = 10
OUT_DIR      = "../outputs/beats"
PROJ_CRS     = "EPSG:32611"
RANDOM_SEED  = 42
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

print("=" * 40)
print("  PPAC Script 1: Beat Generation & Road Network")
print("=" * 40)

# ── 1. Build synthetic road network ───────────────────────────────────────────
print("[1/5] Building road network …")

m_per_lat = 111_000.0
m_per_lon = 111_000.0 * np.cos(np.radians((LAT_MIN + LAT_MAX) / 2))

NROWS, NCOLS = 22, 22
lat_steps = np.linspace(LAT_MIN, LAT_MAX, NROWS)
lon_steps = np.linspace(LON_MIN, LON_MAX, NCOLS)

nodes_latlon = {}
grid = {}
node_id = 0
for r, lat in enumerate(lat_steps):
    for c, lon in enumerate(lon_steps):
        nodes_latlon[node_id] = (lat, lon)
        grid[(r, c)] = node_id
        node_id += 1

edge_list = []
for r in range(NROWS):
    for c in range(NCOLS):
        u = grid[(r, c)]
        lat_u, lon_u = nodes_latlon[u]
        if c < NCOLS - 1:
            v = grid[(r, c + 1)]
            lat_v, lon_v = nodes_latlon[v]
            length = abs(lon_v - lon_u) * m_per_lon
            hw = "primary" if r % 5 == 0 else ("secondary" if r % 3 == 0 else "residential")
            edge_list += [(u, v, length, hw), (v, u, length, hw)]
        if r < NROWS - 1:
            v = grid[(r + 1, c)]
            lat_v, lon_v = nodes_latlon[v]
            length = abs(lat_v - lat_u) * m_per_lat
            hw = "primary" if c % 5 == 0 else ("secondary" if c % 3 == 0 else "residential")
            edge_list += [(u, v, length, hw), (v, u, length, hw)]

# Diagonal arterials
for _ in range(14):
    r1 = int(rng.integers(0, NROWS))
    c1 = int(rng.integers(0, NCOLS))
    r2 = min(NROWS - 1, r1 + int(rng.integers(3, 8)))
    c2 = min(NCOLS - 1, c1 + int(rng.integers(3, 8)))
    u, v = grid[(r1, c1)], grid[(r2, c2)]
    lat_u, lon_u = nodes_latlon[u]
    lat_v, lon_v = nodes_latlon[v]
    dist = float(np.sqrt(
        ((lat_v - lat_u) * m_per_lat) ** 2 + ((lon_v - lon_u) * m_per_lon) ** 2
    ))
    edge_list += [(u, v, dist, "trunk"), (v, u, dist, "trunk")]

G = nx.DiGraph()
for nid, (lat, lon) in nodes_latlon.items():
    G.add_node(nid, y=lat, x=lon)
for u, v, length, hw in edge_list:
    G.add_edge(u, v, length=length, highway=hw)

print(f"      Nodes: {G.number_of_nodes():,}   Edges: {G.number_of_edges():,}")

with open(os.path.join(OUT_DIR, "road_network.pkl"), "wb") as f:
    pickle.dump(G, f)
print("      Saved → ppac_outputs/road_network.pkl")

# ── 2. Project to UTM ─────────────────────────────────────────────────────────
print("[2/5] Projecting to UTM …")
fwd = pyproj.Transformer.from_crs("EPSG:4326", PROJ_CRS, always_xy=True)
rev = pyproj.Transformer.from_crs(PROJ_CRS, "EPSG:4326", always_xy=True)

nodes_utm = {}
for nid, (lat, lon) in nodes_latlon.items():
    xm, ym = fwd.transform(lon, lat)
    nodes_utm[nid] = (xm, ym)

node_xy  = np.array(list(nodes_utm.values()))
node_ids = np.array(list(nodes_utm.keys()))

corners = [
    node_xy.min(axis=0) + [-500, -500],
    [node_xy[:, 0].max() + 500, node_xy[:, 1].min() - 500],
    node_xy.max(axis=0) + [500, 500],
    [node_xy[:, 0].min() - 500, node_xy[:, 1].max() + 500],
]
boundary_proj = Polygon(corners).convex_hull

# ── 3. Voronoi beats ──────────────────────────────────────────────────────────
print(f"[3/5] Voronoi partitioning → {NUM_BEATS} beats …")
minx, miny, maxx, maxy = boundary_proj.bounds

seeds = []
tries = 0
while len(seeds) < NUM_BEATS and tries < 200_000:
    px, py = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
    if boundary_proj.contains(Point(px, py)):
        seeds.append([px, py])
    tries += 1

seed_arr = np.array(seeds[:NUM_BEATS])
pad = max(maxx - minx, maxy - miny) * 1.5
mirrors = np.array([
    [minx - pad, miny - pad], [maxx + pad, miny - pad],
    [minx - pad, maxy + pad], [maxx + pad, maxy + pad],
])
vor = Voronoi(np.vstack([seed_arr, mirrors]))

beat_polys = []
for i in range(NUM_BEATS):
    ridx  = vor.point_region[i]
    verts = vor.regions[ridx]
    if -1 in verts or not verts:
        beat_polys.append(boundary_proj)
        continue
    poly    = Polygon(vor.vertices[verts])
    clipped = poly.intersection(boundary_proj)
    beat_polys.append(clipped if not clipped.is_empty else boundary_proj)

beats_gdf = gpd.GeoDataFrame(
    {"beat_id": [f"BEAT_{i+1:02d}" for i in range(NUM_BEATS)]},
    geometry=beat_polys,
    crs=PROJ_CRS,
)

# ── 4. Nearest nodes ──────────────────────────────────────────────────────────
print("[4/5] Mapping centroids to nearest road nodes …")
records = []
for i, row in beats_gdf.iterrows():
    cx, cy = seed_arr[i]
    dists  = np.sqrt((node_xy[:, 0] - cx) ** 2 + (node_xy[:, 1] - cy) ** 2)
    nn_id  = int(node_ids[dists.argmin()])
    lon_c, lat_c = rev.transform(cx, cy)
    records.append({"beat_id": row.beat_id, "lat": round(lat_c, 6),
                    "lon": round(lon_c, 6), "nearest_node": nn_id})

centroids_df = pd.DataFrame(records)
centroids_df.to_csv(os.path.join(OUT_DIR, "beat_centroids.csv"), index=False)
print("      Saved → ppac_outputs/beat_centroids.csv")

beats_gdf.to_crs("EPSG:4326").to_file(
    os.path.join(OUT_DIR, "beats.geojson"), driver="GeoJSON"
)
print("      Saved → ppac_outputs/beats.geojson")

# ── 5. Visualise ───────────────────────────────────────────────────────────────
print("[5/5] Visualising …")
cmap   = cm.get_cmap("tab20", NUM_BEATS)
colors = [cmap(i) for i in range(NUM_BEATS)]

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle(
    "Police Beat Generation — Hollywood / North Hollywood, Los Angeles CA\n"
    f"Voronoi Spatial Partitioning  |  N = {NUM_BEATS} Patrol Beats",
    fontsize=13, fontweight="bold", y=1.01,
)

highway_palette = {
    "trunk": "#e41a1c", "primary": "#ff7f00",
    "secondary": "#f0a500", "residential": "#b2df8a",
    "unclassified": "#cccccc",
}

# Left panel — road network
ax0 = axes[0]
ax0.set_title("Synthetic Road Network\n(grid + arterials, drive-type)", fontsize=10)
for u, v, length, hw in edge_list[:10_000]:
    if u not in nodes_utm or v not in nodes_utm:
        continue
    x0, y0 = nodes_utm[u]; x1, y1 = nodes_utm[v]
    c = highway_palette.get(hw, "#cccccc")
    lw = 1.8 if hw == "trunk" else (1.2 if hw == "primary" else 0.4)
    ax0.plot([x0, x1], [y0, y1], color=c, linewidth=lw, alpha=0.7, zorder=1)

bx, by = boundary_proj.exterior.xy
ax0.plot(bx, by, "k-", linewidth=2, zorder=3)
ax0.set_aspect("equal"); ax0.set_axis_off()
ax0.legend(
    handles=[mpatches.Patch(color=v, label=k) for k, v in highway_palette.items()],
    fontsize=7, loc="lower left", title="Road type",
)

# Right panel — beats
ax1 = axes[1]
ax1.set_title(f"Generated Patrol Beats (N={NUM_BEATS})\nwith Underlying Road Network", fontsize=10)
for u, v, length, hw in edge_list[:5_000]:
    if u not in nodes_utm or v not in nodes_utm:
        continue
    x0, y0 = nodes_utm[u]; x1, y1 = nodes_utm[v]
    ax1.plot([x0, x1], [y0, y1], color="#d0d0d0", linewidth=0.3, alpha=0.5, zorder=1)

for i, row in beats_gdf.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    try:
        xs, ys = geom.exterior.xy
        ax1.fill(xs, ys, alpha=0.35, color=colors[i], zorder=2)
        ax1.plot(xs, ys, color="black", linewidth=0.9, zorder=3)
    except Exception:
        pass
    cx, cy = seed_arr[i]
    ax1.plot(cx, cy, "k^", markersize=8, zorder=5)
    ax1.annotate(
        row.beat_id, xy=(cx, cy), ha="center", va="center",
        fontsize=6.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, ec="none"),
        zorder=6,
    )

ax1.legend(
    handles=[mpatches.Patch(facecolor=colors[i], edgecolor="black",
                             label=beats_gdf.iloc[i].beat_id)
             for i in range(NUM_BEATS)],
    fontsize=7, loc="lower left", title="Patrol Beat", ncol=2,
)
ax1.set_aspect("equal"); ax1.set_axis_off()

plt.tight_layout()
out_fig = os.path.join(OUT_DIR, "beat_generation.png")
plt.savefig(out_fig, dpi=180, bbox_inches="tight")
plt.close()
print(f"      Saved → {out_fig}")

print("\n" + "=" * 60)
print("BEAT GENERATION SUMMARY")
print("=" * 60)
print(f"  Study area   : Hollywood / North Hollywood, LA")
print(f"  Nodes        : {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")
print(f"  Beats        : {NUM_BEATS}")
for _, r in centroids_df.iterrows():
    print(f"  {r.beat_id}  lat={r.lat:.4f}  lon={r.lon:.4f}  node={r.nearest_node}")
print("=" * 60)