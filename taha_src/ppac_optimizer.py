"""
Police Patrol Area Covering (PPAC) — Integer Programming Optimizer
===================================================================
Implements the exact PPAC formulation from Curtin, Hayslett-McCall & Qiu (2010)
using PuLP + CBC branch-and-bound solver for proven-optimal solutions.

Pipeline
--------
Stage 1  Weighted Mini-Batch K-Means on ~1M crime incidents
         → NUM_BEATS beat centroids  (candidate set J)
Stage 2  Road-network OD matrix: one Dijkstra per candidate site
         → coverage sets C_j, beat-level weights a_j
Stage 3  PPAC integer programme (PuLP + CBC)
         → P optimal HQ locations x*_j, objective Z*
Stage 4  Full incident-level coverage evaluation
         → maximal covering objective O, backup objective B
Stage 5  Voronoi beat map + sector map
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
    raise ImportError("PuLP is required:  pip install pulp")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CRIME_DATA_PATH    = '../resources/cleaned_data.csv'
BOUNDARY_FILE_PATH = '../resources/LA_AREA.geojson'

NUM_BEATS   = 300        # |J|  candidate facility locations
NUM_SECTORS = 21         # P    command centres to locate
SERVICE_MI  = 2.5        # S    service radius in miles
SERVICE_M   = SERVICE_MI * 1_609.34   # S in metres

OUTPUT_IMG  = '../outputs/beats/ppac_ip_optimal.png'
OUTPUT_CSV  = '../outputs/beats/ppac_ip_summary.csv'
OSM_CACHE   = '../resources/la_drive_network.graphml'

IP_TIME_LIMIT = 1800     # CBC time limit in seconds (0 = no limit)
IP_MIP_GAP    = 0.0      # 0.0 = proven optimal; raise to e.g. 0.01 for 1% gap


# ─────────────────────────────────────────────────────────────────────────────
# 1.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_or_download_graph(boundary_gdf: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    """Load cached OSM graph or download and cache it."""
    if os.path.exists(OSM_CACHE):
        print("  Loading cached OSM road network …")
        return ox.load_graphml(OSM_CACHE)
    print("  Downloading OSM road network …")
    poly = boundary_gdf.to_crs(epsg=4326).unary_union.convex_hull
    G = ox.graph_from_polygon(poly, network_type='drive')
    os.makedirs(os.path.dirname(OSM_CACHE), exist_ok=True)
    ox.save_graphml(G, OSM_CACHE)
    print(f"  Saved graph → {OSM_CACHE}")
    return G


def snap_to_nodes(G_4326: nx.MultiDiGraph,
                  lons: np.ndarray,
                  lats: np.ndarray) -> np.ndarray:
    """
    Vectorized node snapping using the EPSG:4326 graph.
    ox.nearest_nodes() builds a KD-tree internally; one bulk call is O(n log n).
    Returned node IDs are valid for any projection of the same graph.
    """
    return np.array(ox.nearest_nodes(G_4326, X=lons, Y=lats))


def build_coverage_sets(G_metric: nx.MultiDiGraph,
                        candidate_nodes: np.ndarray,
                        inc_nodes: np.ndarray,
                        radius_m: float) -> list:
    """
    For each candidate HQ node j, compute C_j = {incident indices i | d(i,j) <= S}.

    Uses single-source Dijkstra from each candidate with cutoff=radius_m on the
    UTM-projected graph (edge lengths unambiguously in metres).

    Complexity: O(|J| * Dijkstra)  where one Dijkstra = O(|V| log|V| + |E|).

    Returns
    -------
    coverage_sets : list of sets, len == len(candidate_nodes)
        coverage_sets[j] = set of incident indices covered by candidate j
    """
    # Reverse lookup: OSM node id → list of incident indices
    node_to_incs: dict = {}
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


def solve_ppac_ip(coverage_sets: list,
                  weights: np.ndarray,
                  n_inc: int,
                  P: int,
                  time_limit: int = 1800,
                  mip_gap: float = 0.0) -> tuple:
    """
    Solve the PPAC Maximal Covering Location Problem via integer programming.

    Formulation (Curtin et al., 2010):
    -----------------------------------
    Maximise   Z  =  sum_i  a_i * y_i
    subject to
        sum_{j in N_i} x_j  >=  y_i      for all i in I   [coverage]
        sum_j  x_j          =   P                          [cardinality]
        x_j in {0,1}                      for all j in J
        y_i in {0,1}                      for all i in I

    N_i is built from coverage_sets: N_i = {j | i in coverage_sets[j]}.

    Parameters
    ----------
    coverage_sets : list of sets  (length |J|)
    weights       : incident weight array, shape (n_inc,)
    n_inc         : number of incidents
    P             : number of facilities to locate
    time_limit    : CBC wall-clock time limit in seconds (0 = unlimited)
    mip_gap       : relative MIP gap tolerance (0.0 = proven optimal)

    Returns
    -------
    x_sol     : np.ndarray bool, shape (|J|,)  — selected facilities
    y_sol     : np.ndarray float, shape (n_inc,) — coverage indicators
    Z         : float  — optimal objective value
    gap       : float  — final MIP gap reported by solver (0 if optimal)
    status    : str    — PuLP solver status string
    """
    n_cand = len(coverage_sets)

    # ── Build N_i (incident → list of covering candidates) ──────────────────
    # N[i] = list of candidate indices j such that i ∈ coverage_sets[j]
    N = [[] for _ in range(n_inc)]
    for j, cs in enumerate(coverage_sets):
        for i in cs:
            N[i].append(j)

    # ── Declare PuLP problem ─────────────────────────────────────────────────
    prob = pulp.LpProblem("PPAC_MCLP", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(n_cand)]
    y = [pulp.LpVariable(f"y_{i}", cat='Binary') for i in range(n_inc)]

    # Objective: maximise weighted coverage
    prob += pulp.lpSum(weights[i] * y[i] for i in range(n_inc)), "MaximalCovering"

    # Coverage constraints: sum_{j in N_i} x_j >= y_i  for all i
    for i in range(n_inc):
        if N[i]:
            prob += (pulp.lpSum(x[j] for j in N[i]) >= y[i],
                     f"cov_{i}")
        else:
            # Incident i cannot be covered by any facility → force y_i = 0
            prob += (y[i] == 0, f"uncoverable_{i}")

    # Cardinality constraint: exactly P facilities
    prob += (pulp.lpSum(x) == P, "cardinality")

    # ── Configure and run CBC solver ─────────────────────────────────────────
    solver_kwargs = dict(msg=1)
    if time_limit > 0:
        solver_kwargs['timeLimit'] = time_limit
    if mip_gap > 0:
        solver_kwargs['gapRel'] = mip_gap

    solver = pulp.PULP_CBC_CMD(**solver_kwargs)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    print(f"   Solver status : {status}")
    print(f"   Objective Z*  : {pulp.value(prob.objective):,.2f}")

    x_sol = np.array([pulp.value(x[j]) or 0.0 for j in range(n_cand)]) > 0.5
    y_sol = np.array([pulp.value(y[i]) or 0.0 for i in range(n_inc)])

    # Compute final MIP gap if solver stopped early
    try:
        gap = prob.solver.solverModel.bestBound  # CBC internal bound
    except Exception:
        gap = 0.0

    return x_sol, y_sol, float(pulp.value(prob.objective) or 0.0), gap, status


def evaluate_coverage(coverage_sets: list,
                      selected_mask: np.ndarray,
                      weights: np.ndarray,
                      n_inc: int) -> tuple:
    """
    Compute the maximal covering objective O and backup objective B for the
    selected set of HQs, as defined in Curtin et al. (2010).

    O = sum_i  a_i * 1[covered_count_i >= 1]
    B = sum_i  a_i * covered_count_i          (backup / multiple coverage)
    """
    coverage_counts = np.zeros(n_inc, dtype=int)
    for j in np.where(selected_mask)[0]:
        for inc_idx in coverage_sets[j]:
            coverage_counts[inc_idx] += 1

    covered_mask = coverage_counts >= 1
    O = weights[covered_mask].sum()
    B = (weights * coverage_counts).sum()
    return coverage_counts, int(covered_mask.sum()), float(O), float(B)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_patrol_map():
    t0 = time.time()

    # ── 2.1  Load and filter crime data ──────────────────────────────────────
    print("[1/6] Loading data …")
    if not os.path.exists(CRIME_DATA_PATH) or not os.path.exists(BOUNDARY_FILE_PATH):
        print("ERROR: Missing input files.")
        return

    city_boundary = gpd.read_file(BOUNDARY_FILE_PATH).to_crs(epsg=3857)

    df = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['LON'], df['LAT'])],
        crs='EPSG:4326'
    ).to_crs(epsg=3857)

    gdf = gpd.sjoin(gdf, city_boundary, how='inner', predicate='within').copy()
    weights = gdf['crime_weight'].values
    n_inc   = len(gdf)
    print(f"   {n_inc:,} incidents loaded.  ({time.time()-t0:.1f}s)")

    # ── 2.2  Stage 1: Weighted K-Means → candidate set J ─────────────────────
    # Mirrors Curtin et al.'s use of beat centroids as candidate sites,
    # but derives them from the crime data rather than pre-existing boundaries.
    print(f"[2/6] Stage 1: Mini-Batch K-Means → {NUM_BEATS} beat centroids …")
    t1 = time.time()
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])

    km = MiniBatchKMeans(
        n_clusters=NUM_BEATS,
        n_init=10,
        batch_size=10_000,
        random_state=42,
        max_iter=300,
    )
    km.fit(coords, sample_weight=weights)
    beat_centers = km.cluster_centers_   # EPSG:3857, shape (NUM_BEATS, 2)
    beat_labels  = km.labels_            # incident → beat
    print(f"   Done.  ({time.time()-t1:.1f}s)")

    # Beat-level aggregate weights (demand at each candidate site)
    beat_weights = np.zeros(NUM_BEATS)
    for b in range(NUM_BEATS):
        mask = beat_labels == b
        if mask.any():
            beat_weights[b] = weights[mask].sum()

    # ── 2.3  OSM road network ─────────────────────────────────────────────────
    print("[3/6] Acquiring OSM road network …")
    t2 = time.time()
    G = load_or_download_graph(city_boundary)

    # Two projections (see build_coverage_sets docstring for rationale):
    #   G_4326   — EPSG:4326, used only for ox.nearest_nodes() snapping
    #   G_metric — UTM, edge 'length' in metres, used for all Dijkstra calls
    G_4326   = ox.project_graph(G, to_crs='EPSG:4326')
    G_metric = ox.project_graph(G)          # auto-selects local UTM zone
    print(f"   {len(G_metric.nodes):,} nodes, {len(G_metric.edges):,} edges.  "
          f"({time.time()-t2:.1f}s)")

    # ── 2.4  Vectorized node snapping ─────────────────────────────────────────
    print("[4/6] Snapping beat centroids and incidents to OSM nodes …")
    t3 = time.time()

    beat_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(beat_centers[:, 0], beat_centers[:, 1]),
        crs=3857
    ).to_crs(4326)
    beat_nodes = snap_to_nodes(G_4326,
                               beat_gdf.geometry.x.values,
                               beat_gdf.geometry.y.values)
    print(f"   {NUM_BEATS} beat nodes snapped.  ({time.time()-t3:.1f}s)")

    inc_gdf   = gdf.to_crs(4326)
    inc_nodes = snap_to_nodes(G_4326,
                              inc_gdf.geometry.x.values,
                              inc_gdf.geometry.y.values)
    print(f"   {n_inc:,} incident nodes snapped.  ({time.time()-t3:.1f}s)")

    # ── 2.5  Build coverage sets (N_i construction) ───────────────────────────
    # For each candidate j ∈ J: C_j = {i ∈ I | d^road(i,j) ≤ S}
    # One Dijkstra per candidate on the metric graph, cutoff = SERVICE_M.
    print(f"[5/6] Building road-network coverage sets "
          f"(S = {SERVICE_MI} mi = {SERVICE_M:.0f} m) …")
    t4 = time.time()

    # Coverage sets indexed by beat (candidate sites)
    # Used for both the IP (beat-level demand) and post-hoc incident evaluation
    beat_coverage_sets  = build_coverage_sets(
        G_metric, beat_nodes, beat_nodes, SERVICE_M   # beat-to-beat for IP
    )
    inc_coverage_sets   = build_coverage_sets(
        G_metric, beat_nodes, inc_nodes, SERVICE_M    # beat-to-incident for evaluation
    )

    coverable = sum(1 for cs in beat_coverage_sets if cs)
    print(f"   {coverable}/{NUM_BEATS} candidates cover ≥1 beat.  "
          f"({time.time()-t4:.1f}s)")

    # ── 2.6  Stage 2: PPAC Integer Programme ─────────────────────────────────
    # Solve: max sum_j a_j * y_j
    #        s.t. sum_{j' in N_j} x_{j'} >= y_j  for all j (beat coverage)
    #             sum_j x_j = P
    #             x_j, y_j ∈ {0,1}
    #
    # The IP operates on beat-level demand (aggregated weights) and beat-level
    # coverage sets, keeping the problem size tractable:
    #   Variables: 2 * NUM_BEATS binaries  (vs 2 * n_inc for incident-level)
    #   Constraints: NUM_BEATS coverage + 1 cardinality
    print(f"[5b/6] Stage 2: Solving PPAC IP (P={NUM_SECTORS}, |J|={NUM_BEATS}) …")
    t5 = time.time()

    x_sol, y_sol_beat, Z_ip, gap, status = solve_ppac_ip(
        coverage_sets=beat_coverage_sets,
        weights=beat_weights,
        n_inc=NUM_BEATS,
        P=NUM_SECTORS,
        time_limit=IP_TIME_LIMIT,
        mip_gap=IP_MIP_GAP,
    )

    selected_idx  = np.where(x_sol)[0]
    sector_hqs    = beat_centers[selected_idx]          # EPSG:3857, shape (P,2)
    hq_nodes_opt  = beat_nodes[selected_idx]            # OSM node IDs
    print(f"   {len(selected_idx)} HQs selected.  Z* = {Z_ip:,.2f}  "
          f"({time.time()-t5:.1f}s)")

    # ── 2.7  Full incident-level coverage evaluation ──────────────────────────
    print("[5c/6] Evaluating coverage on full incident set …")
    t6 = time.time()

    # Select the incident-coverage-sets for the chosen HQs
    selected_inc_cs = [inc_coverage_sets[j] for j in selected_idx]

    # Reconstruct a boolean mask over inc_coverage_sets using selected_idx
    # (evaluate_coverage expects a mask over the full list)
    inc_selected_mask = np.zeros(NUM_BEATS, dtype=bool)
    inc_selected_mask[selected_idx] = True

    coverage_counts, covered_count, O, B = evaluate_coverage(
        inc_coverage_sets, inc_selected_mask, weights, n_inc
    )

    total_w    = weights.sum()
    pct_count  = 100 * covered_count / n_inc
    pct_weight = 100 * O             / total_w

    print(f"\n   ── PPAC IP Coverage (S = {SERVICE_MI} mi = {SERVICE_M:.0f} m) ──")
    print(f"   IP Status                     : {status}")
    if gap:
        print(f"   MIP Gap                       : {gap:.4f}")
    print(f"   Beat-level IP Objective (Z*)  : {Z_ip:,.2f}")
    print(f"   Incident coverage (count)     : {covered_count:,} / {n_inc:,}  "
          f"({pct_count:.1f} %)")
    print(f"   Maximal Covering Obj (O)      : {O:,.1f} / {total_w:,.1f}  "
          f"({pct_weight:.1f} %)")
    print(f"   Maximal Backup Obj (B)        : {B:,.1f}")
    print(f"   ({time.time()-t6:.1f}s)\n")

    # Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame({
        'metric': [
            'Total incidents', 'Service distance (mi)', 'Service distance (m)',
            'Num beats (|J|)', 'Num sectors (P)',
            'IP solver status', 'MIP gap',
            'Beat-level IP objective (Z*)',
            'Incidents covered (count)', 'Coverage pct (count)',
            'Maximal Covering Objective (O)', 'Coverage pct (weighted)',
            'Maximal Backup Objective (B)',
            'Total runtime (s)',
        ],
        'value': [
            n_inc, SERVICE_MI, SERVICE_M,
            NUM_BEATS, NUM_SECTORS,
            status, gap,
            round(Z_ip, 2),
            covered_count, round(pct_count, 2),
            round(O, 2), round(pct_weight, 2),
            round(B, 2),
            round(time.time()-t0, 1),
        ]
    }).to_csv(OUTPUT_CSV, index=False)

    # ── 2.8  Assign beats to sectors and visualise ────────────────────────────
    print("[6/6] Rendering map …")
    t7 = time.time()

    hq_tree = cKDTree(sector_hqs)
    _, beat_to_sector = hq_tree.query(beat_centers)
    sector_labels = beat_to_sector   # beat_id → sector_id (0..P-1)

    vor = Voronoi(beat_centers)
    rows = []
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            rows.append({
                'geometry':  Polygon([vor.vertices[v] for v in region]),
                'beat_id':   i,
                'sector_id': int(sector_labels[i]),
            })

    gdf_voronoi       = gpd.GeoDataFrame(rows, crs=3857)
    gdf_beats_clipped = gpd.overlay(gdf_voronoi, city_boundary, how='intersection')
    gdf_sectors_viz   = gdf_beats_clipped.dissolve(by='sector_id')

    fig, ax = plt.subplots(figsize=(16, 13))
    gdf_sectors_viz.plot(ax=ax, column=gdf_sectors_viz.index,
                         cmap='tab20', alpha=0.45,
                         edgecolor='royalblue', linewidth=2.0)
    gdf_beats_clipped.plot(ax=ax, facecolor='none',
                           edgecolor='black', linewidth=0.3, alpha=0.6)
    gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(sector_hqs[:, 0], sector_hqs[:, 1]), crs=3857
    ).plot(ax=ax, color='red', marker='H', markersize=220,
           edgecolor='white', zorder=10)

    cx.add_basemap(ax, crs=3857,
                   source=cx.providers.OpenStreetMap.Mapnik,
                   alpha=0.3, zoom=11)

    mid_hq = sector_hqs[len(sector_hqs) // 2]
    ax.add_patch(patches.Circle(
        (mid_hq[0], mid_hq[1]), SERVICE_M,
        linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.10
    ))
    ax.annotate(
        f"{SERVICE_MI} mi Road-Network Service Radius ($S$)",
        xy=(mid_hq[0], mid_hq[1] + SERVICE_M + 600),
        ha='center', weight='bold', color='blue', fontsize=9
    )
    ax.text(
        0.02, 0.02,
        f"PPAC Integer Programming (Proven Optimal)\n"
        f"S = {SERVICE_MI} mi  |  P = {NUM_SECTORS}  |  |J| = {NUM_BEATS}\n"
        f"IP Status: {status}\n"
        f"Coverage: {pct_count:.1f}%  of incidents\n"
        f"Maximal Covering Obj (O): {O:,.0f}\n"
        f"Maximal Backup Obj  (B): {B:,.0f}",
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )
    plt.title(
        "Optimal Police Patrol Geography — PPAC Integer Programme + Road-Network\n"
        "Sectors (Coloured)  ·  Beats (Outlined)  ·  Optimal HQs (Red Hexagons)",
        fontsize=14,
    )
    ax.set_axis_off()

    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Map saved → {OUTPUT_IMG}  ({time.time()-t7:.1f}s)")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    generate_patrol_map()