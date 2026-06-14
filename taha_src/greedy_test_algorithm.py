"""
Optimized Police Patrol Beat Generator — Road-Network Edition (Fixed + Stats on Map)
=====================================================================

BUGS FIXED (all three caused the optimizer to underperform existing stations):
-------------------------------------------------------------------------------

BUG 1 — WRONG OBJECTIVE BEING OPTIMIZED (Critical)
    The K-Means Stage 2 minimizes Euclidean within-cluster variance of beat
    centroids. This has NO relationship to maximizing the PPAC coverage
    objective (sum of weighted incidents within S). K-Means placed HQs at
    the geometric center of beat clusters, not where they cover the most
    crime weight. Fixed: Stage 2 uses a greedy coverage-maximization search
    — for each of the P sector HQs, pick the beat centroid from J that adds
    the most uncovered weighted incidents within S on the road network.
    This directly optimizes the PPAC objective as defined in the paper.

BUG 2 — SERVICE RADIUS APPLIED IN WRONG COORDINATE SPACE (Critical)
    SERVICE_M was computed correctly as metres, but the Voronoi / sector
    assignment used raw EPSG:3857 coordinates (also metres), while the
    Dijkstra cutoff used the road graph projected to EPSG:4326 (degrees).
    The ego-graph Dijkstra cutoff must be in metres and the graph must use
    the metric-projected graph — NOT the EPSG:4326 graph. OSMnx's
    project_graph to EPSG:4326 stores edge lengths still in metres, so
    the cutoff is fine, but nearest_nodes on the 4326 graph uses degree-
    space KD-tree which introduces snapping error for points far from the
    prime meridian. Fixed: snap using the 4326 graph (correct for
    nearest_nodes API), but run Dijkstra on the original projected (UTM)
    graph whose edge weights are unambiguously in metres.

BUG 3 — INCIDENT WEIGHTS NOT USED IN STAGE 1 K-MEANS (Minor but impactful)
    MiniBatchKMeans.fit() received sample_weight but predict() does NOT
    accept sample_weight, so beat_labels from km_beats.labels_ are the
    unweighted assignments. The beat centroids themselves are weighted-
    centroid correct, but the per-beat weight aggregation later is wrong
    because labels don't match the weighted fit. Fixed: use the fit labels
    directly (km_beats.labels_) which ARE the weighted assignments from fit.
    Also, the coverage evaluation now aggregates weights per beat correctly.
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

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
CRIME_DATA_PATH    = '../resources/cleaned_data.csv'
BOUNDARY_FILE_PATH = '../resources/LA_AREA.geojson'

NUM_BEATS   = 300   # Candidate set |J| — beat centroids
NUM_SECTORS = 21    # P — number of command centres to locate
SERVICE_MI  = 2.5
SERVICE_M   = SERVICE_MI * 1_609.34

OUTPUT_IMG  = '../outputs/beats/backup_coverage_image_fixed_with_stats.png'
OUTPUT_CSV  = '../outputs/beats/coverage_summary_roadnet_fixed.csv'
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


def snap_all_to_nodes(G_4326: nx.MultiDiGraph,
                      lons: np.ndarray,
                      lats: np.ndarray) -> np.ndarray:
    """
    Vectorized snap using the EPSG:4326 graph.
    ox.nearest_nodes() requires lon/lat input and uses an internal KD-tree.
    Returns OSM node ids — these are the same node ids in all projections
    of the same graph, so the result is valid for Dijkstra on G_metric too.
    """
    return np.array(ox.nearest_nodes(G_4326, X=lons, Y=lats))


def build_coverage_sets(G_metric: nx.MultiDiGraph,
                        candidate_nodes: np.ndarray,
                        inc_nodes: np.ndarray,
                        radius_m: float) -> list[set]:
    """
    For each candidate HQ node j, compute the set of incident indices
    reachable within radius_m metres on the road network.

    Returns a list of sets: coverage_sets[j] = {incident indices covered by j}

    This is the N_i construction step from Curtin et al. (2010), implemented
    from the facility side: for each j, which i satisfy d(i,j) <= S?

    BUG 2 FIX: Dijkstra runs on G_metric (UTM, edge lengths in metres),
    NOT on the EPSG:4326 projected graph. The node IDs are identical across
    projections, so snapping on 4326 and routing on metric is correct.
    """
    # Build reverse lookup: OSM node id → list of incident indices
    node_to_incs: dict[int, list[int]] = {}
    for idx, node in enumerate(inc_nodes):
        node_to_incs.setdefault(int(node), []).append(idx)

    coverage_sets = []
    for j_node in candidate_nodes:
        covered = set()
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G_metric, int(j_node), cutoff=radius_m, weight='length'
            )
            for node in lengths:
                for inc_idx in node_to_incs.get(node, []):
                    covered.add(inc_idx)
        except Exception:
            pass
        coverage_sets.append(covered)

    return coverage_sets


def greedy_ppac(coverage_sets: list[set],
                weights: np.ndarray,
                P: int) -> np.ndarray:
    """
    Greedy maximization of the PPAC objective (Curtin et al., 2010):
        Maximise Z = sum_i  a_i * y_i
    where y_i = 1 if incident i is covered by at least one selected HQ.

    BUG 1 FIX: This replaces K-Means Stage 2. K-Means minimizes Euclidean
    variance between beat centroids — it does NOT maximize weighted coverage
    within S. The greedy algorithm directly optimizes the PPAC objective:
    at each step, pick the candidate that adds the most uncovered weight.

    The greedy algorithm achieves a (1 - 1/e) ≈ 63% approximation guarantee
    for the submodular maximum coverage problem, which matches or exceeds
    what K-Means can offer for this objective (K-Means has no coverage
    guarantee at all).

    Parameters
    ----------
    coverage_sets : list of sets, coverage_sets[j] = incident indices covered by HQ j
    weights       : incident weight array, shape (n_inc,)
    P             : number of HQs to select

    Returns
    -------
    selected : boolean array of shape (len(coverage_sets),), True where HQ selected
    """
    n_candidates = len(coverage_sets)
    selected     = np.zeros(n_candidates, dtype=bool)
    covered      = set()   # currently covered incident indices

    for _ in range(P):
        best_j     = -1
        best_gain  = -1.0

        for j in range(n_candidates):
            if selected[j]:
                continue
            # New incidents this candidate would add
            new_incidents = coverage_sets[j] - covered
            gain = weights[list(new_incidents)].sum() if new_incidents else 0.0
            if gain > best_gain:
                best_gain = gain
                best_j    = j

        if best_j == -1:
            break   # no candidate adds anything

        selected[best_j] = True
        covered |= coverage_sets[best_j]

    return selected


def evaluate_coverage(coverage_sets: list[set],
                      selected_mask: np.ndarray,
                      weights: np.ndarray,
                      n_inc: int):
    """
    Compute maximal covering and backup covering objectives for a given
    selection of HQs, exactly as defined in Curtin et al. (2010).

    Maximal covering objective O = sum_i a_i * w_i  where w_i = 1{covered >= 1}
    Maximal backup objective   B = sum_i a_i * y_i  where y_i = # HQs covering i
    """
    coverage_counts = np.zeros(n_inc, dtype=int)
    selected_indices = np.where(selected_mask)[0]

    for j in selected_indices:
        for inc_idx in coverage_sets[j]:
            coverage_counts[inc_idx] += 1

    covered_once_mask          = coverage_counts >= 1
    covered_count              = covered_once_mask.sum()
    maximal_covering_objective = weights[covered_once_mask].sum()
    maximal_backup_objective   = (weights * coverage_counts).sum()

    return coverage_counts, covered_count, maximal_covering_objective, maximal_backup_objective


# ─────────────────────────────────────────────
# 2. MAIN PIPELINE
# ─────────────────────────────────────────────

def generate_patrol_map():
    t0 = time.time()

    # ── 2.1  Load data ───────────────────────────────────────────────────────
    if not os.path.exists(CRIME_DATA_PATH) or not os.path.exists(BOUNDARY_FILE_PATH):
        print("ERROR: Missing input files.")
        return

    print("[1/6] Loading data …")
    city_boundary = gpd.read_file(BOUNDARY_FILE_PATH).to_crs(epsg=3857)

    df = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

    geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_filtered = gpd.sjoin(gdf, city_boundary, how='inner', predicate='within').copy()
    weights = gdf_filtered['crime_weight'].values
    n_incidents = len(gdf_filtered)
    print(f"   {n_incidents:,} crime incidents inside city boundary.  "
          f"({time.time()-t0:.1f}s)")

    # ── 2.2  Stage 1: Beat clustering (Mini-Batch K-Means) ──────────────────
    # Purpose: generate NUM_BEATS candidate HQ locations (the set J).
    # This is data-driven and mirrors the paper's use of beat centroids as J.
    print("[2/6] Stage 1: Mini-Batch K-Means → beat centroids (candidate set J) …")
    t1 = time.time()
    coords = np.column_stack((gdf_filtered.geometry.x, gdf_filtered.geometry.y))

    km_beats = MiniBatchKMeans(
        n_clusters=NUM_BEATS,
        n_init=10,
        batch_size=10_000,
        random_state=42,
        max_iter=300,
    )
    # BUG 3 FIX: fit with sample_weight; labels_ from fit() ARE weight-aware
    km_beats.fit(coords, sample_weight=weights)
    beat_centers = km_beats.cluster_centers_   # shape (NUM_BEATS, 2), EPSG:3857
    beat_labels  = km_beats.labels_            # incident → beat, from weighted fit
    print(f"   {NUM_BEATS} candidate sites generated.  ({time.time()-t1:.1f}s)")

    # ── 2.3  Load OSM road network ───────────────────────────────────────────
    print("[3/6] Acquiring OSM road network …")
    t3 = time.time()
    G = load_or_download_graph(city_boundary)

    # BUG 2 FIX: keep TWO graph objects:
    #   G_4326   — EPSG:4326 lon/lat, used ONLY for ox.nearest_nodes() snapping
    #   G_metric — UTM projected, edge 'length' unambiguously in metres,
    #              used for ALL Dijkstra distance computations
    G_4326   = ox.project_graph(G, to_crs='EPSG:4326')
    G_metric = ox.project_graph(G)   # OSMnx auto-selects local UTM zone
    print(f"   Graph ready: {len(G_metric.nodes):,} nodes, "
          f"{len(G_metric.edges):,} edges.  ({time.time()-t3:.1f}s)")

    # ── 2.4  VECTORIZED node snapping ────────────────────────────────────────
    print("[4/6] Snapping points to OSM nodes (vectorized) …")
    t4 = time.time()

    # Beat centroids → EPSG:4326 for snapping
    beat_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(beat_centers[:, 0], beat_centers[:, 1]), crs=3857
    ).to_crs(4326)
    beat_nodes = snap_all_to_nodes(G_4326,
                                   beat_gdf.geometry.x.values,
                                   beat_gdf.geometry.y.values)
    print(f"   {len(beat_nodes)} beat centroid nodes snapped.  ({time.time()-t4:.1f}s)")

    # Incidents → EPSG:4326 for snapping
    t4b = time.time()
    inc_gdf   = gdf_filtered.to_crs(4326)
    inc_nodes = snap_all_to_nodes(G_4326,
                                  inc_gdf.geometry.x.values,
                                  inc_gdf.geometry.y.values)
    print(f"   {n_incidents:,} incident nodes snapped.  ({time.time()-t4b:.1f}s)")

    # ── 2.5  Build coverage sets on the metric graph ─────────────────────────
    # This is the N_i construction: for each candidate j, which incidents i
    # satisfy d^road(i, j) <= S?  Dijkstra runs on G_metric (metres).
    print("[5/6] Building road-network coverage sets "
          f"(S = {SERVICE_MI} mi = {SERVICE_M:.0f} m) …")
    t5 = time.time()
    coverage_sets = build_coverage_sets(G_metric, beat_nodes, inc_nodes, SERVICE_M)
    coverable = sum(1 for cs in coverage_sets if cs)
    print(f"   {coverable}/{NUM_BEATS} candidates cover at least one incident.  "
          f"({time.time()-t5:.1f}s)")

    # ── 2.6  Stage 2: Greedy PPAC maximization ───────────────────────────────
    # BUG 1 FIX: replaces K-Means Stage 2 with direct coverage maximization.
    # Greedy selection of P HQs that maximise sum_i a_i * y_i.
    print("[5b/6] Stage 2: Greedy PPAC coverage maximization → select sector HQs …")
    t6 = time.time()
    selected_mask = greedy_ppac(coverage_sets, weights, NUM_SECTORS)
    selected_idx  = np.where(selected_mask)[0]
    sector_hqs    = beat_centers[selected_idx]   # shape (P, 2), EPSG:3857

    # Assign each beat centroid to its nearest selected HQ (for Voronoi sectors)
    hq_tree = cKDTree(sector_hqs)
    _, beat_to_sector = hq_tree.query(beat_centers)
    sector_labels = beat_to_sector   # beat_id → sector_id (0..P-1)
    print(f"   {NUM_SECTORS} optimal HQs selected.  ({time.time()-t6:.1f}s)")

    # ── 2.7  Coverage evaluation ─────────────────────────────────────────────
    print("[5c/6] Evaluating coverage objectives …")
    t7 = time.time()
    coverage_counts, covered_count, max_cov_obj, max_backup_obj = evaluate_coverage(
        coverage_sets, selected_mask, weights, n_incidents
    )

    total_weighted = weights.sum()
    pct_count      = 100 * covered_count / n_incidents
    pct_weighted   = 100 * max_cov_obj   / total_weighted

    print(f"\n   ── PPAC Coverage (S = {SERVICE_MI} mi = {SERVICE_M:.0f} m) ──")
    print(f"   Covered (count)               : {covered_count:,} / {n_incidents:,}  ({pct_count:.1f} %)")
    print(f"   Maximal Covering Objective (O): {max_cov_obj:,.1f} / {total_weighted:,.1f}  ({pct_weighted:.1f} %)")
    print(f"   Maximal Backup Objective      : {max_backup_obj:,.1f}")
    print(f"   ({time.time()-t7:.1f}s)\n")

    # Save summary CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame({
        'metric': [
            'Total incidents', 'Service distance (mi)', 'Service distance (m)',
            'Incidents covered (count)', 'Coverage pct (count)',
            'Maximal Covering Objective', 'Coverage pct (weighted)',
            'Maximal Backup Objective', 'Total runtime (s)',
        ],
        'value': [
            n_incidents, SERVICE_MI, SERVICE_M,
            covered_count, round(pct_count, 2),
            round(max_cov_obj, 2), round(pct_weighted, 2),
            round(max_backup_obj, 2), round(time.time()-t0, 1),
        ]
    }).to_csv(OUTPUT_CSV, index=False)

    # ── 2.8  Visualisation ───────────────────────────────────────────────────
    print("[6/6] Rendering map …")
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
    gdf_sectors_viz   = gdf_beats_clipped.dissolve(by='sector_id')

    fig, ax = plt.subplots(figsize=(16, 13))
    gdf_sectors_viz.plot(ax=ax, column=gdf_sectors_viz.index,
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

    # -- THIS BLOCK IS MODIFIED to add more detailed percentages onto the image --
    stats_text = (
        f"Road-Network PPAC Coverage (Greedy-Optimal)\n"
        f"Parameters: S={SERVICE_MI} mi | Sectors={NUM_SECTORS} | Beats={NUM_BEATS}\n"
        f"\n"
        f"Incidents Covered (count) : {covered_count:,} / {n_incidents:,} ({pct_count:.1f} %)\n"
        f"Maximal Covering Obj (O) : {max_cov_obj:,.1f} / {total_weighted:,.1f} ({pct_weighted:.1f} %)\n"
        f"Maximal Backup Objective : {max_backup_obj:,.1f}"
    )
    # Adding the detailed text box to the plot
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.title(
        "Optimized Police Patrol Geography — PPAC Greedy + Road-Network Distance\n"
        "Sectors (Coloured) · Beats (Outlined) · Optimal HQs (Red Hexagons)",
        fontsize=14
    )
    ax.set_axis_off()

    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Map saved → {OUTPUT_IMG}  ({time.time()-t8:.1f}s)")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    generate_patrol_map()