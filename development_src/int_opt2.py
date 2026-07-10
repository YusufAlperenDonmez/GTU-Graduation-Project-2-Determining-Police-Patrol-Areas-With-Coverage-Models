"""
Police Patrol Area Covering (PPAC) — Exact Incident-Level Optimizer
===================================================================
Implements the exact PPAC formulation from Curtin, Hayslett-McCall & Qiu (2010).
Optimizes directly against individual crime incidents (no aggregation).

Filters data strictly to the INTERSECTION of the user polygon and LA_AREA.geojson.

Pipeline
--------
Stage 1  Weighted Mini-Batch K-Means on filtered crime incidents -> candidate set J
Stage 2  Road-network OD matrix -> incident-level coverage sets
Stage 3  PPAC integer programme -> P optimal HQ locations, checking every incident
Stage 4  Full incident-level coverage evaluation
Stage 5  Bounded Voronoi beat map + sector map clipped to unified area intersection
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
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import voronoi_diagram
from scipy.spatial import cKDTree

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

# User-defined sub-area boundary coordinates
USER_POLYGON_COORDS =[[-118.280182, 34.164091], [-118.314514, 34.059486], [-118.138733, 34.008273], [-118.134613, 34.155], [-118.280182, 34.164091]]


NUM_BEATS   = 300        # |J|  candidate facility locations
NUM_SECTORS = 12         # P    command centres to locate
SERVICE_MI  = 2.0        # S    service radius in miles
SERVICE_M   = SERVICE_MI * 1_609.34   # S in metres

OUTPUT_IMG  = '../outputs/beats/ppac_exact_optimal.png'
OUTPUT_CSV  = '../outputs/beats/ppac_exact_summary.csv'
OSM_CACHE   = '../resources/la_drive_network.graphml'

IP_TIME_LIMIT = 1800     # CBC time limit in seconds (0 = no limit)
IP_MIP_GAP    = 0.0      # 0.0 = proven optimal; raise to e.g. 0.01 for 1% gap


# ─────────────────────────────────────────────────────────────────────────────
# 1.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_or_download_graph(boundary_gdf: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    if os.path.exists(OSM_CACHE):
        print("  Loading cached OSM road network ...")
        return ox.load_graphml(OSM_CACHE)
    print("  Downloading OSM road network ...")
    poly = boundary_gdf.to_crs(epsg=4326).unary_union.convex_hull
    G = ox.graph_from_polygon(poly, network_type='drive')
    os.makedirs(os.path.dirname(OSM_CACHE), exist_ok=True)
    ox.save_graphml(G, OSM_CACHE)
    print(f"  Saved graph -> {OSM_CACHE}")
    return G


def snap_to_nodes(G_4326: nx.MultiDiGraph,
                  lons: np.ndarray,
                  lats: np.ndarray) -> np.ndarray:
    return np.array(ox.nearest_nodes(G_4326, X=lons, Y=lats))


def build_coverage_sets(G_metric: nx.MultiDiGraph,
                        candidate_nodes: np.ndarray,
                        inc_nodes: np.ndarray,
                        radius_m: float) -> list:
    node_to_incs: dict = {}
    for idx, node in enumerate(inc_nodes):
        node_to_incs.setdefault(int(node), []).append(idx)

    coverage_sets = []
    total_cands = len(candidate_nodes)
    print_interval = max(1, total_cands // 20)  # Print every 5%

    for j, j_node in enumerate(candidate_nodes):
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
        
        if (j + 1) % print_interval == 0 or (j + 1) == total_cands:
            pct = ((j + 1) / total_cands) * 100
            print(f"      Routing progress: {pct:.0f}% complete ({j + 1}/{total_cands} candidates processed)")

    return coverage_sets


def solve_ppac_ip(coverage_sets: list,
                  weights: np.ndarray,
                  n_inc: int,
                  P: int,
                  time_limit: int = 1800,
                  mip_gap: float = 0.0) -> tuple:
    n_cand = len(coverage_sets)

    print("    Building incident-to-candidate reverse mapping (N_i)...")
    N = [[] for _ in range(n_inc)]
    for j, cs in enumerate(coverage_sets):
        for i in cs:
            N[i].append(j)

    print("    Initializing PuLP model variables...")
    prob = pulp.LpProblem("PPAC_MCLP_Exact", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(n_cand)]
    y = [pulp.LpVariable(f"y_{i}", cat='Binary') for i in range(n_inc)]

    print("    Adding objective function...")
    prob += pulp.lpSum(weights[i] * y[i] for i in range(n_inc)), "MaximalCovering"

    print(f"    Building {n_inc:,} coverage constraints...")
    print_interval = max(1, n_inc // 20)  # Print every 5%
    
    for i in range(n_inc):
        if N[i]:
            prob += (pulp.lpSum(x[j] for j in N[i]) >= y[i], f"cov_{i}")
        else:
            prob += (y[i] == 0, f"uncoverable_{i}")
            
        if (i + 1) % print_interval == 0 or (i + 1) == n_inc:
            pct = ((i + 1) / n_inc) * 100
            print(f"      Constraint build progress: {pct:.0f}% complete ({i + 1}/{n_inc} variables)")

    prob += (pulp.lpSum(x) == P, "cardinality")

    print("    Passing exact formulation to CBC solver...")
    solver_kwargs = dict(msg=1)
    if time_limit > 0:
        solver_kwargs['timeLimit'] = time_limit
    if mip_gap > 0:
        solver_kwargs['gapRel'] = mip_gap

    solver = pulp.PULP_CBC_CMD(**solver_kwargs)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    print(f"    Solver status : {status}")
    print(f"    Objective Z* : {pulp.value(prob.objective):,.2f}")

    x_sol = np.array([pulp.value(x[j]) or 0.0 for j in range(n_cand)]) > 0.5
    y_sol = np.array([pulp.value(y[i]) or 0.0 for i in range(n_inc)])

    try:
        gap = prob.solver.solverModel.bestBound
    except Exception:
        gap = 0.0

    return x_sol, y_sol, float(pulp.value(prob.objective) or 0.0), gap, status


def evaluate_coverage(coverage_sets: list,
                      selected_mask: np.ndarray,
                      weights: np.ndarray,
                      n_inc: int) -> tuple:
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

    # ── 2.1  Load boundaries and compute Spatial Intersection ────────────────
    print("[1/6] Loading geometry boundaries and generating intersection...")
    if not os.path.exists(BOUNDARY_FILE_PATH) or not os.path.exists(CRIME_DATA_PATH):
        print("ERROR: Missing geometry or crime data input files.")
        return

    # Load official LA Area GeoJSON
    la_geojson = gpd.read_file(BOUNDARY_FILE_PATH).to_crs(epsg=4326)

    # Build user custom bounding polygon
    user_poly = Polygon(USER_POLYGON_COORDS)
    user_gdf = gpd.GeoDataFrame({'geometry': [user_poly]}, crs='EPSG:4326')

    # Calculate exact geometric intersection overlay of both layers
    # This prevents the map regions from rendering outside the genuine LA border walls
    intersection_poly = la_geojson.unary_union.intersection(user_gdf.unary_union)
    
    if intersection_poly.is_empty:
        print("ERROR: The custom polygon coordinates do not overlap with the LA GeoJSON file.")
        return

    # Construct the definitive target boundary workspace
    city_boundary = gpd.GeoDataFrame({'geometry': [intersection_poly]}, crs='EPSG:4326').to_crs(epsg=3857)
    print("   Successfully locked bounding zone intersection.")

    # ── 2.2  Load and filter crime data ──────────────────────────────────────
    df = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['LON'], df['LAT'])],
        crs='EPSG:4326'
    ).to_crs(epsg=3857)

    # Filter incidents to strictly fall within our intersection zone
    gdf = gpd.sjoin(gdf, city_boundary, how='inner', predicate='within').copy()
    weights = gdf['crime_weight'].values
    n_inc   = len(gdf)
    print(f"   {n_inc:,} incidents matched inside region.  ({time.time()-t0:.1f}s)")

    if n_inc == 0:
        print("ERROR: No crime incidents found inside the computed boundary intersection.")
        return

    # Adjust configurations safely if the data load is smaller than expected
    global NUM_BEATS, NUM_SECTORS
    if n_inc < NUM_BEATS:
        NUM_BEATS = n_inc
        print(f"   Forced adjustment: NUM_BEATS scaled down to matches count ({NUM_BEATS})")
    if NUM_BEATS <= NUM_SECTORS:
        NUM_SECTORS = max(1, NUM_BEATS // 2)
        print(f"   Forced adjustment: NUM_SECTORS scaled down to safely fit candidate size ({NUM_SECTORS})")

    # ── 2.3  Stage 1: Weighted K-Means -> candidate set J ─────────────────────
    print(f"[2/6] Generating Candidate Pool: {NUM_BEATS} potential HQ sites ...")
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
    print(f"   Done.  ({time.time()-t1:.1f}s)")

    # ── 2.4  OSM road network ─────────────────────────────────────────────────
    print("[3/6] Acquiring OSM road network ...")
    t2 = time.time()
    G = load_or_download_graph(city_boundary)

    G_4326   = ox.project_graph(G, to_crs='EPSG:4326')
    G_metric = ox.project_graph(G)          
    print(f"   {len(G_metric.nodes):,} nodes, {len(G_metric.edges):,} edges.  "
          f"({time.time()-t2:.1f}s)")

    # ── 2.5  Vectorized node snapping ─────────────────────────────────────────
    print("[4/6] Snapping potential HQs and actual incidents to road nodes ...")
    t3 = time.time()

    beat_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(beat_centers[:, 0], beat_centers[:, 1]),
        crs=3857
    ).to_crs(4326)
    beat_nodes = snap_to_nodes(G_4326, beat_gdf.geometry.x.values, beat_gdf.geometry.y.values)
    print(f"   {NUM_BEATS} candidate HQ nodes snapped.")

    inc_gdf   = gdf.to_crs(4326)
    inc_nodes = snap_to_nodes(G_4326, inc_gdf.geometry.x.values, inc_gdf.geometry.y.values)
    print(f"   {n_inc:,} exact incident nodes snapped.  ({time.time()-t3:.1f}s)")

    # ── 2.6  Build coverage sets (N_i construction) ───────────────────────────
    print(f"[5/6] Building EXACT incident road-network coverage sets (S = {SERVICE_MI} mi)...")
    t4 = time.time()

    inc_coverage_sets = build_coverage_sets(
        G_metric, beat_nodes, inc_nodes, SERVICE_M
    )
    print(f"   Coverage sets built. ({time.time()-t4:.1f}s)")

    # ── 2.7  Stage 2: PPAC Integer Programme ─────────────────────────────────
    print(f"[5b/6] Stage 2: Solving Exact PPAC IP (P={NUM_SECTORS}, |J|={NUM_BEATS}, |I|={n_inc}) ...")
    t5 = time.time()

    x_sol, y_sol_inc, Z_ip, gap, status = solve_ppac_ip(
        coverage_sets=inc_coverage_sets,
        weights=weights,
        n_inc=n_inc,
        P=NUM_SECTORS,
        time_limit=IP_TIME_LIMIT,
        mip_gap=IP_MIP_GAP,
    )

    selected_idx  = np.where(x_sol)[0]
    sector_hqs    = beat_centers[selected_idx]          # EPSG:3857, shape (P,2)
    print(f"   {len(selected_idx)} HQs selected.  Z* = {Z_ip:,.2f}  "
          f"({time.time()-t5:.1f}s)")

    # ── 2.8  Full incident-level coverage evaluation ──────────────────────────
    print("[5c/6] Evaluating final metrics ...")
    t6 = time.time()

    coverage_counts, covered_count, O, B = evaluate_coverage(
        inc_coverage_sets, x_sol, weights, n_inc
    )

    total_w    = weights.sum()
    pct_count  = 100 * covered_count / n_inc
    pct_weight = 100 * O             / total_w

    print(f"\n   ── EXACT PPAC IP Coverage (S = {SERVICE_MI} mi = {SERVICE_M:.0f} m) ──")
    print(f"   IP Status                     : {status}")
    if gap:
        print(f"   MIP Gap                       : {gap:.4f}")
    print(f"   Exact IP Objective (Z*)       : {Z_ip:,.2f}")
    print(f"   Incident coverage (count)     : {covered_count:,} / {n_inc:,}  "
          f"({pct_count:.1f} %)")
    print(f"   Maximal Covering Obj (O)      : {O:,.1f} / {total_w:,.1f}  "
          f"({pct_weight:.1f} %)")
    print(f"   Maximal Backup Obj (B)        : {B:,.1f}")
    print(f"   ({time.time()-t6:.1f}s)\n")

    # Save CSV Results Summary
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame({
        'metric': [
            'Total incidents', 'Service distance (mi)', 'Service distance (m)',
            'Num candidates (|J|)', 'Num sectors (P)',
            'IP solver status', 'MIP gap',
            'Exact IP objective (Z*)',
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

    # ── 2.9  Assign beats to sectors and visualise ────────────────────────────
    print("[6/6] Rendering map ...")
    t7 = time.time()

    hq_tree = cKDTree(sector_hqs)
    _, beat_to_sector = hq_tree.query(beat_centers)
    sector_labels = beat_to_sector   # beat_id -> sector_id (0..P-1)

    # 1. Bounded Voronoi processing envelope configuration
    envelope = city_boundary.unary_union.envelope.buffer(100000) 
    centers_mp = MultiPoint(beat_centers)
    vor_collection = voronoi_diagram(centers_mp, envelope=envelope)
    
    vor_polys = list(vor_collection.geoms)
    gdf_voronoi = gpd.GeoDataFrame(geometry=vor_polys, crs=3857)

    # 2. Bind spatial attributes to beat mappings
    beats_gdf = gpd.GeoDataFrame(
        {'beat_id': range(len(beat_centers)), 'sector_id': sector_labels},
        geometry=[Point(x, y) for x, y in beat_centers], 
        crs=3857
    )
    gdf_vor_mapped = gpd.sjoin(gdf_voronoi, beats_gdf, how='inner', predicate='contains')

    # 3. Clean-clip the boundary layers strictly along the computed intersection wall
    gdf_beats_clipped = gpd.overlay(gdf_vor_mapped, city_boundary, how='intersection')
    gdf_sectors_viz   = gdf_beats_clipped.dissolve(by='sector_id')

    fig, ax = plt.subplots(figsize=(16, 13))
    
    # Render colored structural sectors
    gdf_sectors_viz.plot(ax=ax, column=gdf_sectors_viz.index,
                         cmap='tab20', alpha=0.45,
                         edgecolor='royalblue', linewidth=2.0)
                         
    # Overlay beat border lines
    gdf_beats_clipped.plot(ax=ax, facecolor='none',
                           edgecolor='black', linewidth=0.3, alpha=0.6)
                           
    # Hexagon marker anchors for located installations
    gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(sector_hqs[:, 0], sector_hqs[:, 1]), crs=3857
    ).plot(ax=ax, color='red', marker='H', markersize=220,
           edgecolor='white', zorder=10)

    # Match background map structure canvas
    cx.add_basemap(ax, crs=3857,
                   source=cx.providers.OpenStreetMap.Mapnik,
                   alpha=0.3, zoom=11)

    # Draw specific coverage distance vector radius
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
    
    # Statistical dashboard anchor placements
    ax.text(
        0.02, 0.02,
        f"Exact PPAC Integer Programming\n"
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
        "Optimal Police Patrol Geography — Exact PPAC Formulation\n"
        "Sectors (Coloured)  ·  Beats (Outlined)  ·  Optimal HQs (Red Hexagons)",
        fontsize=14,
    )
    ax.set_axis_off()

    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Map saved -> {OUTPUT_IMG}  ({time.time()-t7:.1f}s)")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    generate_patrol_map()