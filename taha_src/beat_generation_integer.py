"""
Optimized Police Patrol Beat Generator — Road-Network Edition (Integer Programming)
=====================================================================================
Key changes from the heuristic version:
-----------------------------------------
1.  STAGE 2 replaced: Sector headquarters are now selected by solving the
    Police Patrol Area Covering (PPAC) MCLP via PuLP + CBC integer programming,
    exactly as in Curtin, Hayslett-McCall & Qiu (2010).  K-Means is retained
    only for Stage 1 beat generation (to produce the candidate set J).

2.  Road-network N_i sets: For every beat centroid i and every candidate HQ j,
    we compute d^road_{ij} via Dijkstra on the OSM graph.  N_i = {j | d_{ij} <= S}.

3.  PPAC objective: Maximise sum_i a_i * y_i subject to the classic MCLP
    constraints.  The solver returns P optimal HQ locations from the candidate set.

4.  Coverage evaluation uses the same ego-graph approach as before, but now
    applied to the *optimal* HQ locations returned by the IP solver.

5.  Vectorized node snapping and Mini-Batch K-Means (Stage 1 only) are retained
    for speed.
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import contextily as cx
import networkx as nx
import osmnx as ox

from sklearn.cluster import MiniBatchKMeans
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi, cKDTree

try:
    import pulp
except ImportError:
    raise ImportError("PuLP is required: pip install pulp")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
CRIME_DATA_PATH    = '../resources/cleaned_data.csv'
BOUNDARY_FILE_PATH = '../resources/LA_AREA.geojson'

NUM_BEATS   = 300        # |J| — candidate facility locations (beat centroids)
NUM_SECTORS = 30         # P   — number of patrol command centres to locate
SERVICE_MI  = 2.0        # S in miles
SERVICE_M   = SERVICE_MI * 1_609.34

OUTPUT_IMG  = '../outputs/beats/patrol_image_roadnet_ip.png'
OUTPUT_CSV  = '../outputs/beats/coverage_summary_roadnet_ip.csv'
OSM_CACHE   = '../resources/la_drive_network.graphml'

# ─────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────

def load_or_download_graph(boundary_gdf: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    if os.path.exists(OSM_CACHE):
        print("  Loading cached OSM road network …")
        G = ox.load_graphml(OSM_CACHE)
    else:
        print("  Downloading OSM road network …")
        poly = boundary_gdf.to_crs(epsg=4326).unary_union.convex_hull
        G = ox.graph_from_polygon(poly, network_type='drive')
        os.makedirs(os.path.dirname(OSM_CACHE), exist_ok=True)
        ox.save_graphml(G, OSM_CACHE)
        print(f"  Saved graph → {OSM_CACHE}")
    return G


def snap_all_to_nodes(G: nx.MultiDiGraph,
                      lons: np.ndarray,
                      lats: np.ndarray) -> np.ndarray:
    """Vectorized snap: one KD-tree call for all points. O(n log n)."""
    return np.array(ox.nearest_nodes(G, X=lons, Y=lats))


def build_road_od_matrix(G: nx.MultiDiGraph,
                         hq_nodes: np.ndarray,
                         inc_nodes: np.ndarray,
                         radius_m: float) -> np.ndarray:
    """
    Build a (len(inc_nodes), len(hq_nodes)) distance matrix using
    single-source Dijkstra from each HQ node with cutoff=radius_m.
    Only entries <= radius_m are populated (others remain inf).

    This is used to construct N_i sets for the PPAC integer programme.
    Complexity: O(P * Dijkstra) — one Dijkstra per candidate HQ.
    """
    n_inc = len(inc_nodes)
    n_hq  = len(hq_nodes)
    D = np.full((n_inc, n_hq), np.inf)

    # Build reverse lookup: OSM node id → list of incident indices
    node_to_incs: dict[int, list[int]] = {}
    for idx, node in enumerate(inc_nodes):
        node_to_incs.setdefault(int(node), []).append(idx)

    for j, hq_node in enumerate(hq_nodes):
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, int(hq_node), cutoff=radius_m, weight='length'
            )
        except Exception:
            continue
        for node, dist in lengths.items():
            for inc_idx in node_to_incs.get(node, []):
                D[inc_idx, j] = dist

    return D


def solve_ppac_ip(a: np.ndarray,
                  D: np.ndarray,
                  P: int,
                  radius_m: float,
                  verbose: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the Police Patrol Area Covering (PPAC) model via integer programming.

    Formulation (Curtin et al., 2010):
        Maximise   Z = sum_i  a_i * y_i
        subject to sum_{j in N_i} x_j >= y_i    for all i   (coverage)
                   sum_j x_j = P                             (cardinality)
                   x_j in {0,1}                 for all j
                   y_i in {0,1}                 for all i

    Parameters
    ----------
    a        : incident weights, shape (n_inc,)
    D        : road OD matrix, shape (n_inc, n_hq), inf where d > S
    P        : number of patrol command centres to locate
    radius_m : service radius in metres (used only to build N_i here via D)
    verbose  : print solver log

    Returns
    -------
    x_sol    : binary array of shape (n_hq,), 1 if HQ j is selected
    y_sol    : binary array of shape (n_inc,), 1 if incident i is covered
    Z        : optimal objective value
    """
    n_inc, n_hq = D.shape

    # Build N_i: for each incident i, the set of HQ indices that can cover it
    N = [list(np.where(D[i, :] <= radius_m)[0]) for i in range(n_inc)]

    # Create PuLP problem
    prob = pulp.LpProblem("PPAC_MCLP", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(n_hq)]
    y = [pulp.LpVariable(f"y_{i}", cat='Binary') for i in range(n_inc)]

    # Objective
    prob += pulp.lpSum(a[i] * y[i] for i in range(n_inc))

    # Coverage constraints: sum_{j in N_i} x_j >= y_i
    for i in range(n_inc):
        if N[i]:
            prob += pulp.lpSum(x[j] for j in N[i]) >= y[i]
        else:
            # No HQ can cover incident i — force y_i = 0
            prob += y[i] == 0

    # Cardinality constraint
    prob += pulp.lpSum(x) == P

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=600)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    print(f"   IP solver status: {status}")

    x_sol = np.array([pulp.value(x[j]) or 0 for j in range(n_hq)], dtype=float)
    y_sol = np.array([pulp.value(y[i]) or 0 for i in range(n_inc)], dtype=float)
    Z     = pulp.value(prob.objective) or 0.0

    return x_sol, y_sol, Z


def coverage_via_ego_graphs(G: nx.MultiDiGraph,
                             hq_nodes: np.ndarray,
                             inc_nodes: np.ndarray,
                             assigned_sector: np.ndarray,
                             weights: np.ndarray,
                             radius_m: float):
    """
    Post-optimisation coverage evaluation.
    For each selected HQ, use ego-graph Dijkstra to mark covered incidents.
    """
    n_inc = len(inc_nodes)
    road_distances = np.full(n_inc, np.inf)

    sector_inc_idx = {s: np.where(assigned_sector == s)[0]
                      for s in range(len(hq_nodes))}

    for sec_idx, hq_node in enumerate(hq_nodes):
        idxs = sector_inc_idx.get(sec_idx, np.array([], dtype=int))
        if len(idxs) == 0:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, int(hq_node), cutoff=radius_m, weight='length'
            )
        except Exception:
            continue
        for idx in idxs:
            node = int(inc_nodes[idx])
            road_distances[idx] = lengths.get(node, np.inf)

    covered_mask     = road_distances <= radius_m
    covered_count    = covered_mask.sum()
    covered_weighted = weights[covered_mask].sum()
    return road_distances, covered_count, covered_weighted


# ─────────────────────────────────────────────
# 2. MAIN PIPELINE
# ─────────────────────────────────────────────

def generate_patrol_map():
    t0 = time.time()

    # ── 2.1  Load data ───────────────────────────────────────────────────────
    if not os.path.exists(CRIME_DATA_PATH) or not os.path.exists(BOUNDARY_FILE_PATH):
        print("ERROR: Missing input files.")
        return

    print("[1/7] Loading data …")
    city_boundary = gpd.read_file(BOUNDARY_FILE_PATH).to_crs(epsg=3857)

    df = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

    geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_filtered = gpd.sjoin(gdf, city_boundary, how='inner', predicate='within').copy()
    print(f"   {len(gdf_filtered):,} crime incidents inside city boundary.  "
          f"({time.time()-t0:.1f}s)")

    # ── 2.2  Stage 1: Beat clustering — Mini-Batch K-Means (candidate set J) ─
    print("[2/7] Stage 1: Mini-Batch K-Means → beat centroids (candidate set J) …")
    t1 = time.time()
    coords  = np.column_stack((gdf_filtered.geometry.x, gdf_filtered.geometry.y))
    weights = gdf_filtered['crime_weight'].values

    km_beats = MiniBatchKMeans(
        n_clusters=NUM_BEATS,
        n_init=5,
        batch_size=10_000,
        random_state=42,
        max_iter=200,
    )
    km_beats.fit(coords, sample_weight=weights)
    beat_centers = km_beats.cluster_centers_   # shape (NUM_BEATS, 2)  — this is J
    beat_labels  = km_beats.labels_            # incident → beat assignment
    print(f"   {NUM_BEATS} beat centroids generated (candidate set |J|).  "
          f"({time.time()-t1:.1f}s)")

    # ── 2.3  Load OSM road network ───────────────────────────────────────────
    print("[3/7] Acquiring OSM road network …")
    t3 = time.time()
    G = load_or_download_graph(city_boundary)
    G_proj = ox.project_graph(G, to_crs='EPSG:4326')
    print(f"   Graph ready: {len(G_proj.nodes):,} nodes, {len(G_proj.edges):,} edges.  "
          f"({time.time()-t3:.1f}s)")

    # ── 2.4  VECTORIZED node snapping ────────────────────────────────────────
    print("[4/7] Snapping points to OSM nodes (vectorized) …")
    t4 = time.time()

    # Beat centroids → EPSG:4326
    beat_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(beat_centers[:, 0], beat_centers[:, 1]), crs=3857
    ).to_crs(4326)
    beat_lon = beat_gdf.geometry.x.values
    beat_lat = beat_gdf.geometry.y.values
    beat_nodes = snap_all_to_nodes(G_proj, beat_lon, beat_lat)
    print(f"   {len(beat_nodes)} beat centroid nodes snapped.  ({time.time()-t4:.1f}s)")

    # Incidents → EPSG:4326
    t4b = time.time()
    inc_gdf = gdf_filtered.to_crs(4326)
    inc_lon = inc_gdf.geometry.x.values
    inc_lat = inc_gdf.geometry.y.values
    inc_nodes = snap_all_to_nodes(G_proj, inc_lon, inc_lat)
    print(f"   {len(inc_nodes):,} incident nodes snapped.  ({time.time()-t4b:.1f}s)")

    # ── 2.5  Build road OD matrix for beat centroids ─────────────────────────
    # Each beat centroid is a candidate HQ (element of J).
    # We compute d^road_{ij} for all i (incidents) × j (beat centroids).
    # This step uses one Dijkstra per beat centroid.
    print("[5/7] Building road OD matrix (incident × beat-centroid) …")
    t5 = time.time()
    D = build_road_od_matrix(G_proj, beat_nodes, inc_nodes, SERVICE_M)
    print(f"   OD matrix shape: {D.shape}  |  "
          f"Coverable pairs: {np.isfinite(D).sum():,}  "
          f"({time.time()-t5:.1f}s)")

    # ── 2.6  Stage 2: PPAC integer programme ─────────────────────────────────
    # Aggregate incident weights to beat level for the IP
    # (the IP works on beat centroids as demand points, weighted by
    #  the sum of crime_weight of incidents in each beat — mirrors the
    #  Curtin et al. approach of using a weighted demand set).
    print("[6/7] Stage 2: Solving PPAC MCLP via integer programming …")
    t6 = time.time()

    # Beat-level weights: sum of incident weights per beat
    beat_weights = np.zeros(NUM_BEATS)
    for b in range(NUM_BEATS):
        mask = beat_labels == b
        beat_weights[b] = weights[mask].sum() if mask.any() else 0.0

    # Build beat-level OD matrix: D_beat[i,j] = min road dist from beat i to beat j
    # We reuse beat_nodes for both rows and columns.
    print("   Building beat-to-beat OD matrix …")
    D_beat = build_road_od_matrix(G_proj, beat_nodes, beat_nodes, SERVICE_M)

    x_sol, y_sol, Z_ip = solve_ppac_ip(
        a=beat_weights,
        D=D_beat,
        P=NUM_SECTORS,
        radius_m=SERVICE_M,
        verbose=True,
    )

    # Selected HQ indices (beat centroid indices where x_j = 1)
    selected_hq_idx  = np.where(x_sol > 0.5)[0]
    sector_hqs       = beat_centers[selected_hq_idx]          # shape (P, 2) in EPSG:3857
    hq_nodes_optimal = beat_nodes[selected_hq_idx]            # OSM node ids

    print(f"   PPAC objective Z = {Z_ip:.2f}  |  "
          f"Selected {len(selected_hq_idx)} HQs.  ({time.time()-t6:.1f}s)")

    # ── 2.6b  Assign each beat to its nearest optimal HQ ─────────────────────
    hq_tree = cKDTree(sector_hqs)
    _, beat_to_sector = hq_tree.query(beat_centers)
    sector_labels = beat_to_sector   # beat_id → sector_id

    # ── 2.7  PPAC Coverage evaluation on incident level ───────────────────────
    print("[6b/7] Evaluating road-network coverage on full incident set …")
    t7 = time.time()

    inc_xy          = np.column_stack((gdf_filtered.geometry.x, gdf_filtered.geometry.y))
    _, nearest_sec  = hq_tree.query(inc_xy)   # each incident → nearest optimal HQ

    road_distances, covered_count, covered_weighted = coverage_via_ego_graphs(
        G_proj, hq_nodes_optimal, inc_nodes, nearest_sec, weights, SERVICE_M
    )

    n_incidents    = len(inc_nodes)
    total_weighted = weights.sum()
    pct_count      = 100 * covered_count    / n_incidents
    pct_weighted   = 100 * covered_weighted / total_weighted

    print(f"\n   ── PPAC Coverage (S = {SERVICE_MI} mi = {SERVICE_M:.0f} m) ──")
    print(f"   IP Objective Z         : {Z_ip:.2f}")
    print(f"   Covered (count)        : {covered_count:,} / {n_incidents:,}  ({pct_count:.1f} %)")
    print(f"   Covered (weighted)     : {covered_weighted:,.1f} / {total_weighted:,.1f}"
          f"  ({pct_weighted:.1f} %)")
    print(f"   ({time.time()-t7:.1f}s)\n")

    # Save summary
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame({
        'metric': [
            'Total incidents', 'Service distance (mi)', 'Service distance (m)',
            'PPAC IP objective Z',
            'Incidents covered (count)', 'Coverage pct (count)',
            'Incidents covered (weighted)', 'Coverage pct (weighted)',
            'Total runtime (s)',
        ],
        'value': [
            n_incidents, SERVICE_MI, SERVICE_M,
            round(Z_ip, 2),
            covered_count, round(pct_count, 2),
            round(covered_weighted, 2), round(pct_weighted, 2),
            round(time.time()-t0, 1),
        ]
    }).to_csv(OUTPUT_CSV, index=False)

    # ── 2.8  Visualisation ───────────────────────────────────────────────────
    print("[7/7] Rendering map …")
    t8 = time.time()

    vor = Voronoi(beat_centers)
    lines = []
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            poly = Polygon([vor.vertices[v] for v in region])
            lines.append({'geometry': poly,
                          'beat_id':   i,
                          'sector_id': int(sector_labels[i])})

    gdf_voronoi       = gpd.GeoDataFrame(lines, crs=3857)
    gdf_beats_clipped = gpd.overlay(gdf_voronoi, city_boundary, how='intersection')
    gdf_sectors       = gdf_beats_clipped.dissolve(by='sector_id')

    fig, ax = plt.subplots(figsize=(16, 13))
    gdf_sectors.plot(ax=ax, column=gdf_sectors.index,
                     cmap='tab20', alpha=0.45, edgecolor='royalblue', linewidth=2.0)
    gdf_beats_clipped.plot(ax=ax, facecolor='none',
                           edgecolor='black', linewidth=0.3, alpha=0.6)
    gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(sector_hqs[:, 0], sector_hqs[:, 1]), crs=3857
    ).plot(ax=ax, color='red', marker='H', markersize=220,
           edgecolor='white', zorder=10)

    cx.add_basemap(ax, crs=3857, source=cx.providers.OpenStreetMap.Mapnik,
                   alpha=0.3, zoom=11)

    example_hq = sector_hqs[len(sector_hqs) // 2]
    ax.add_patch(patches.Circle(
        (example_hq[0], example_hq[1]), SERVICE_M,
        linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.10
    ))
    ax.annotate(
        f"{SERVICE_MI} mi Road-Network Service Radius ($S$)",
        xy=(example_hq[0], example_hq[1] + SERVICE_M + 600),
        ha='center', weight='bold', color='blue', fontsize=9
    )
    ax.text(0.02, 0.02,
            f"Road-Network PPAC Coverage (IP-Optimal)\n"
            f"S={SERVICE_MI} mi | Sectors={NUM_SECTORS} | Beats={NUM_BEATS}\n"
            f"IP Objective Z = {Z_ip:.1f}\n"
            f"Covered: {pct_count:.1f}% of incidents\n"
            f"Weighted coverage: {pct_weighted:.1f}%",
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.title(
        "Optimal Police Patrol Geography — PPAC Integer Programme + Road-Network Distance\n"
        "Sectors (Coloured) · Beats (Outlined) · Optimal HQs (Red Hexagons)",
        fontsize=14
    )
    ax.set_axis_off()

    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Map saved → {OUTPUT_IMG}  ({time.time()-t8:.1f}s)")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    generate_patrol_map()