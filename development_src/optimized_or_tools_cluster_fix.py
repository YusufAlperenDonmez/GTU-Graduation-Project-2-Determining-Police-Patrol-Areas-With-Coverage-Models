import os
import warnings
import time
import json

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
    from ortools.linear_solver import pywraplp
except ImportError:
    raise ImportError("Google OR-Tools is required:  pip install ortools")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CRIME_DATA_PATH    = '../resources/cleaned_data.csv'
BOUNDARY_FILE_PATH = '../resources/LA_AREA.geojson'

# User-defined sub-area boundary coordinates
# USER_POLYGON_COORDS = None
USER_POLYGON_COORDS = [[-118.753967, 34.354774], [-118.096161, 34.354774], [-118.122253, 33.694638], [-118.850098, 33.680925], [-118.753967, 34.354774]]
# USER_POLYGON_COORDS = [[-118.322754, 34.195901], [-118.482056, 34.025348], [-118.151093, 33.950195], [-118.035736, 34.179998], [-118.322754, 34.195901]]

NUM_BEATS   = 500      # |J|  candidate facility locations
NUM_SECTORS = 28         # P    command centres to locate
SERVICE_MI  = 2.0        # S    service radius in miles police station sayisi arttirmak yerine mile dusuruldu 
                            #    bunun sebebi hem ayni sayida polis istasyonu ile kiyas yapabilmek ve  
                            #    LA AREA icerisindeki trafik ile bakınca 2 mile gercekci bir uzaklik olmayabilir.
                            #    2.0 * 2.0 ile 1.8 * 1.8 kiyas yapildiginda oran 4.0 a 3.24 yuzdelik kiyas ise 100 - 83
CLUSTER_RADIUS = 50.0
SERVICE_M   = SERVICE_MI * 1_609.34   # S in metres

OUTPUT_IMG  = '../outputs/optimized/ppac_exact_optimal_28P.png'
OUTPUT_CSV  = '../outputs/optimized/ppac_exact_summary_28P.csv'
OSM_CACHE   = '../resources/la_drive_network.graphml'

# ── NEW: UI Export paths ──────────────────────────────────────────────────────
OUTPUT_STATIONS    = '../outputs/optimized/stations_28P.csv'
OUTPUT_INCIDENTS   = '../outputs/optimized/incidents_export_28P.csv'
OUTPUT_OPT_SUMMARY = '../outputs/optimized/optimization_summary_28P.csv'
OUTPUT_BEATS_GEO   = '../outputs/optimized/beat_polygons_28P.geojson'

IP_TIME_LIMIT = 720000     # iki yüz saat yetismesi icin

IP_MIP_GAP    = 0.000      # feda edilecek percentage yok optimum sonuc


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

def cluster_incidents(gdf: gpd.GeoDataFrame,
                      weights: np.ndarray,
                      cluster_radius_m: float = 200.0) -> tuple:
    """
    Pre-process step: merge spatially proximate incidents into
    representative super-incidents before passing to the IP.

    Uses DBSCAN so you don't need to specify the number of clusters —
    only the neighbourhood radius. Incidents that fall outside every
    cluster (noise) are kept as singletons.

    Parameters
    ----------
    gdf              : GeoDataFrame of incidents (EPSG:3857)
    weights          : per-incident crime_weight array
    cluster_radius_m : merge incidents within this road-Euclidean distance

    Returns
    -------
    gdf_clustered    : GeoDataFrame with one row per super-incident
    new_weights      : summed weights for each super-incident
    inc_to_cluster   : array[n_original] mapping original idx -> cluster idx
    """
    from sklearn.cluster import DBSCAN

    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])

    db = DBSCAN(eps=cluster_radius_m, min_samples=1,
                algorithm='ball_tree', metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(coords)

    n_clusters = labels.max() + 1
    print(f"   Incident clustering: {len(gdf):,} incidents -> "
          f"{n_clusters:,} super-incidents  "
          f"(radius = {cluster_radius_m:.0f} m)")

    # Weighted centroid per cluster
    cx_list, cy_list, w_list = [], [], []
    for cid in range(n_clusters):
        mask = labels == cid
        w_sub = weights[mask]
        w_tot = w_sub.sum()
        x_c = (coords[mask, 0] * w_sub).sum() / w_tot
        y_c = (coords[mask, 1] * w_sub).sum() / w_tot
        cx_list.append(x_c)
        cy_list.append(y_c)
        w_list.append(w_tot)

    from shapely.geometry import Point
    gdf_clustered = gpd.GeoDataFrame(
        {'cluster_id': range(n_clusters)},
        geometry=[Point(x, y) for x, y in zip(cx_list, cy_list)],
        crs=gdf.crs
    )
    new_weights = np.array(w_list)
    inc_to_cluster = labels  # shape (n_original,)

    return gdf_clustered, new_weights, inc_to_cluster

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
    print_interval = max(1, total_cands // 20)

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
            print(f"       Routing progress: {pct:.0f}% complete ({j + 1}/{total_cands} candidates processed)")

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

    print("    Initializing Google OR-Tools SCIP solver...")
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver could not be created in Google OR-Tools.")

    if time_limit > 0:
        solver.SetTimeLimit(int(time_limit * 1000))
    
    if mip_gap >= 0.0:
        solver.SetSolverSpecificParametersAsString(f"limits/gap = {mip_gap}")

    print("    Initializing OR-Tools model variables...")
    x = [solver.BoolVar(f"x_{j}") for j in range(n_cand)]
    y = [solver.BoolVar(f"y_{i}") for i in range(n_inc)]

    print("    Adding objective function...")
    objective = solver.Objective()
    for i in range(n_inc):
        objective.SetCoefficient(y[i], float(weights[i]))
    objective.SetMaximization()

    print(f"    Building {n_inc:,} coverage constraints...")
    print_interval = max(1, n_inc // 20)

    for i in range(n_inc):
        if N[i]:
            constraint = solver.Constraint(0, solver.infinity(), f"cov_{i}")
            constraint.SetCoefficient(y[i], -1.0)
            for j in N[i]:
                constraint.SetCoefficient(x[j], 1.0)
        else:
            constraint = solver.Constraint(0, 0, f"uncoverable_{i}")
            constraint.SetCoefficient(y[i], 1.0)

        if (i + 1) % print_interval == 0 or (i + 1) == n_inc:
            pct = ((i + 1) / n_inc) * 100
            print(f"       Constraint build progress: {pct:.0f}% complete ({i + 1}/{n_inc} variables)")

    cardinality_constraint = solver.Constraint(float(P), float(P), "cardinality")
    for j in range(n_cand):
        cardinality_constraint.SetCoefficient(x[j], 1.0)

    print("    Passing exact formulation to SCIP solver...")
    solver.EnableOutput()
    result_status = solver.Solve()

    status_mapping = {
        pywraplp.Solver.OPTIMAL: "Optimal",
        pywraplp.Solver.FEASIBLE: "Feasible",
        pywraplp.Solver.INFEASIBLE: "Infeasible",
        pywraplp.Solver.UNBOUNDED: "Unbounded",
        pywraplp.Solver.ABNORMAL: "Abnormal",
        pywraplp.Solver.NOT_SOLVED: "NotSolved"
    }
    status = status_mapping.get(result_status, "Unknown")
    
    print(f"    Solver status : {status}")
    obj_val = objective.Value()
    print(f"    Objective Z* : {obj_val:,.2f}")

    x_sol = np.array([x[j].solution_value() for j in range(n_cand)]) > 0.5
    y_sol = np.array([y[i].solution_value() for i in range(n_inc)])

    try:
        gap = solver.Objective().BestBound()
    except Exception:
        gap = 0.0

    return x_sol, y_sol, float(obj_val), gap, status


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
# 2.  UI DATA EXPORT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def export_stations_csv(sector_hqs_3857: np.ndarray,
                        sector_hqs_4326_lon: np.ndarray,
                        sector_hqs_4326_lat: np.ndarray,
                        coverage_sets: list,
                        x_sol: np.ndarray,
                        weights: np.ndarray,
                        gdf_incidents: gpd.GeoDataFrame,
                        beat_to_sector: np.ndarray,
                        beat_centers: np.ndarray,
                        output_path: str):
    selected_idx = np.where(x_sol)[0]
    n_inc = len(weights)
    total_weight = weights.sum()

    rows = []
    for rank, cand_idx in enumerate(selected_idx):
        covered_incs = list(coverage_sets[cand_idx])
        n_covered = len(covered_incs)
        weighted_cov = weights[covered_incs].sum() if covered_incs else 0.0

        backup_count = 0
        for inc_idx in covered_incs:
            coverers = sum(
                1 for sel in selected_idx if inc_idx in coverage_sets[sel]
            )
            if coverers > 1:
                backup_count += 1

        beats_in_sector = int((beat_to_sector == rank).sum())

        rows.append({
            'station_id': rank,
            'candidate_idx': int(cand_idx),
            'lat': float(sector_hqs_4326_lat[rank]),
            'lon': float(sector_hqs_4326_lon[rank]),
            'x_3857': float(sector_hqs_3857[rank, 0]),
            'y_3857': float(sector_hqs_3857[rank, 1]),
            'incidents_covered': n_covered,
            'weighted_coverage': round(float(weighted_cov), 3),
            'backup_incidents': backup_count,
            'beats_assigned': beats_in_sector,
            'coverage_pct': round(100.0 * n_covered / n_inc, 2),
            'weighted_coverage_pct': round(100.0 * float(weighted_cov) / total_weight, 2),
        })

    df_stations = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_stations.to_csv(output_path, index=False)
    print(f"   Stations CSV exported -> {output_path}  ({len(df_stations)} rows)")
    return df_stations


def export_incidents_csv(gdf_incidents: gpd.GeoDataFrame,
                         coverage_sets: list,
                         x_sol: np.ndarray,
                         weights: np.ndarray,
                         coverage_counts: np.ndarray,
                         inc_to_cluster: np.ndarray,      # <-- ADD parameter
                         output_path: str):
    selected_idx = np.where(x_sol)[0]
    inc_4326 = gdf_incidents.to_crs(4326)
    n_raw = len(weights)

    # Build cluster -> [station_rank] mapping from the IP solution
    n_clusters = coverage_counts.shape[0]
    cluster_to_stations = [[] for _ in range(n_clusters)]
    for rank, cand_idx in enumerate(selected_idx):
        for cluster_idx in coverage_sets[cand_idx]:
            cluster_to_stations[cluster_idx].append(rank)

    rows = []
    for i in range(n_raw):
        geom = inc_4326.geometry.iloc[i]
        c = int(inc_to_cluster[i])          # which super-incident this raw incident belongs to
        cov_count  = int(coverage_counts[c])
        stations   = cluster_to_stations[c]

        row = {
            'incident_id':       i,
            'lat':               float(geom.y),
            'lon':               float(geom.x),
            'crime_weight':      float(weights[i]),
            'covered':           int(cov_count >= 1),
            'backup_count':      cov_count,
            'covering_stations': json.dumps(stations),
            'primary_station':   int(stations[0]) if stations else -1,
        }
        orig = gdf_incidents.iloc[i]
        for col in ['Crm Cd Desc', 'AREA NAME', 'DATE OCC', 'Vict Age', 'Vict Sex']:
            if col in gdf_incidents.columns:
                row[col.lower().replace(' ', '_')] = orig[col]

        rows.append(row)

    df_incidents = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_incidents.to_csv(output_path, index=False)
    print(f"   Incidents CSV exported -> {output_path}  ({len(df_incidents):,} rows)")
    return df_incidents


def export_optimization_summary(n_inc, covered_count, pct_count, pct_weight,
                                 Z_ip, O, B, total_weight, gap, status,
                                 SERVICE_MI, SERVICE_M, NUM_BEATS, NUM_SECTORS,
                                 runtime_s, output_path: str):
    df = pd.DataFrame([{
        'total_incidents': n_inc,
        'service_radius_mi': SERVICE_MI,
        'service_radius_m': round(SERVICE_M, 1),
        'num_candidates': NUM_BEATS,
        'num_sectors': NUM_SECTORS,
        'ip_status': status,
        'mip_gap': gap,
        'objective_z': round(Z_ip, 3),
        'incidents_covered': covered_count,
        'coverage_pct_count': round(pct_count, 2),
        'maximal_covering_obj': round(O, 3),
        'coverage_pct_weighted': round(pct_weight, 2),
        'maximal_backup_obj': round(B, 3),
        'total_weight': round(float(total_weight), 3),
        'runtime_seconds': round(runtime_s, 1),
    }])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   Optimization summary exported -> {output_path}")


def export_beat_polygons_geojson(gdf_beats_clipped: gpd.GeoDataFrame,
                                 beat_to_sector: np.ndarray,
                                 output_path: str):
    out = gdf_beats_clipped.copy()
    out = out.to_crs(4326)
    if 'sector_id' not in out.columns and 'beat_id' in out.columns:
        out['sector_id'] = out['beat_id'].map(
            lambda bid: int(beat_to_sector[bid]) if bid < len(beat_to_sector) else -1
        )
    keep_cols = ['geometry', 'beat_id', 'sector_id']
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_file(output_path, driver='GeoJSON')
    print(f"   Beat polygons GeoJSON exported -> {output_path}  ({len(out)} beats)")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_patrol_map():
    t0 = time.time()

    # ── 3.1  Load boundaries and compute Spatial Intersection ────────────────
    print("[1/6] Loading geometry boundaries and generating intersection...")
    if not os.path.exists(BOUNDARY_FILE_PATH) or not os.path.exists(CRIME_DATA_PATH):
        print("ERROR: Missing geometry or crime data input files.")
        return

    la_geojson = gpd.read_file(BOUNDARY_FILE_PATH).to_crs(epsg=4326)
    user_poly  = Polygon(USER_POLYGON_COORDS)
    user_gdf   = gpd.GeoDataFrame({'geometry': [user_poly]}, crs='EPSG:4326')

    intersection_poly = la_geojson.unary_union.intersection(user_gdf.unary_union)

    if intersection_poly.is_empty:
        print("ERROR: The custom polygon coordinates do not overlap with the LA GeoJSON file.")
        return

    city_boundary = gpd.GeoDataFrame(
        {'geometry': [intersection_poly]}, crs='EPSG:4326'
    ).to_crs(epsg=3857)
    print("   Successfully locked bounding zone intersection.")

    # ── 3.2  Load and filter crime data ──────────────────────────────────────
    df = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['LON'], df['LAT'])],
        crs='EPSG:4326'
    ).to_crs(epsg=3857)

    gdf = gpd.sjoin(gdf, city_boundary, how='inner', predicate='within').copy()
    print(f"   {len(gdf):,} raw incidents matched inside region.  ({time.time()-t0:.1f}s)")

    if len(gdf) == 0:
        print("ERROR: No crime incidents found inside the computed boundary intersection.")
        return

    # ── NEW: Apply incident clustering before optimization ───────────────────
    print("   Collapsing proximate incidents into super-incidents...")
    raw_weights = gdf['crime_weight'].values
    n_raw_incidents = len(gdf)  # Keep track of true initial unaggregated counts
    gdf_raw = gdf.copy()          # <-- ADD THIS: preserve original incidents
    
    gdf, weights, inc_to_cluster = cluster_incidents(gdf, raw_weights, CLUSTER_RADIUS)
    n_inc = len(gdf) 
    # ─────────────────────────────────────────────────────────────────────────

    global NUM_BEATS, NUM_SECTORS
    if n_inc < NUM_BEATS:
        NUM_BEATS = n_inc
        print(f"   Forced adjustment: NUM_BEATS scaled down to {NUM_BEATS}")
    if NUM_BEATS <= NUM_SECTORS:
        NUM_SECTORS = max(1, NUM_BEATS // 2)
        print(f"   Forced adjustment: NUM_SECTORS scaled down to {NUM_SECTORS}")

    # ── 3.3  Stage 1: Weighted K-Means -> candidate set J ────────────────────
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
    beat_centers = km.cluster_centers_
    print(f"   Done.  ({time.time()-t1:.1f}s)")

    # ── 3.4  OSM road network ─────────────────────────────────────────────────
    print("[3/6] Acquiring OSM road network ...")
    t2 = time.time()
    G = load_or_download_graph(city_boundary)

    G_4326   = ox.project_graph(G, to_crs='EPSG:4326')
    G_metric = ox.project_graph(G)
    print(f"   {len(G_metric.nodes):,} nodes, {len(G_metric.edges):,} edges.  "
          f"({time.time()-t2:.1f}s)")

    # ── 3.5  Node snapping ────────────────────────────────────────────────────
    print("[4/6] Snapping HQs and incidents to road nodes ...")
    t3 = time.time()

    beat_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(beat_centers[:, 0], beat_centers[:, 1]),
        crs=3857
    ).to_crs(4326)
    beat_nodes = snap_to_nodes(G_4326, beat_gdf.geometry.x.values, beat_gdf.geometry.y.values)
    print(f"   {NUM_BEATS} candidate HQ nodes snapped.")

    inc_gdf  = gdf.to_crs(4326)
    inc_nodes = snap_to_nodes(G_4326, inc_gdf.geometry.x.values, inc_gdf.geometry.y.values)
    print(f"   {n_inc:,} exact incident nodes snapped.  ({time.time()-t3:.1f}s)")

    # ── 3.6  Coverage sets ────────────────────────────────────────────────────
    print(f"[5/6] Building EXACT incident road-network coverage sets (S = {SERVICE_MI} mi)...")
    t4 = time.time()
    inc_coverage_sets = build_coverage_sets(G_metric, beat_nodes, inc_nodes, SERVICE_M)
    print(f"   Coverage sets built. ({time.time()-t4:.1f}s)")

    # ── 3.7  PPAC Integer Programme ───────────────────────────────────────────
    print(f"[5b/6] Solving Exact PPAC IP (P={NUM_SECTORS}, |J|={NUM_BEATS}, |I|={n_inc}) ...")
    t5 = time.time()

    x_sol, y_sol_inc, Z_ip, gap, status = solve_ppac_ip(
        coverage_sets=inc_coverage_sets,
        weights=weights,
        n_inc=n_inc,
        P=NUM_SECTORS,
        time_limit=IP_TIME_LIMIT,
        mip_gap=IP_MIP_GAP,
    )

    selected_idx = np.where(x_sol)[0]
    sector_hqs   = beat_centers[selected_idx]   # EPSG:3857
    print(f"   {len(selected_idx)} HQs selected.  Z* = {Z_ip:,.2f}  ({time.time()-t5:.1f}s)")

    # ── 3.8  Coverage evaluation ──────────────────────────────────────────────
    print("[5c/6] Evaluating final metrics ...")
    t6 = time.time()

    coverage_counts, covered_count, O, B = evaluate_coverage(
        inc_coverage_sets, x_sol, weights, n_inc
    )

    #  Map clustered coverage back to raw incidents to secure unaggregated prints
    cluster_covered_mask = coverage_counts >= 1
    raw_covered_mask = cluster_covered_mask[inc_to_cluster]
    true_raw_covered_count = int(raw_covered_mask.sum())

    total_w    = weights.sum()
    pct_count  = 100 * true_raw_covered_count / n_raw_incidents
    pct_weight = 100 * O / total_w

    print(f"\n   ── EXACT PPAC IP Coverage (S = {SERVICE_MI} mi = {SERVICE_M:.0f} m) ──")
    print(f"   IP Status                      : {status}")
    if gap:
        print(f"   MIP Gap                        : {gap:.4f}")
    print(f"   Exact IP Objective (Z*)        : {Z_ip:,.2f}")
    print(f"   Incident coverage (count)      : {true_raw_covered_count:,} / {n_raw_incidents:,}  ({pct_count:.1f} %)")
    print(f"   Maximal Covering Obj (O)      : {O:,.1f} / {total_w:,.1f}  ({pct_weight:.1f} %)")
    print(f"   Maximal Backup Obj (B)        : {B:,.1f}")
    print(f"   ({time.time()-t6:.1f}s)\n")

    # ── 3.9  Beat / Voronoi geometry ──────────────────────────────────────────
    hq_tree = cKDTree(sector_hqs)
    _, beat_to_sector = hq_tree.query(beat_centers)
    sector_labels = beat_to_sector

    envelope = city_boundary.unary_union.envelope.buffer(100000)
    centers_mp = MultiPoint(beat_centers)
    vor_collection = voronoi_diagram(centers_mp, envelope=envelope)
    vor_polys = list(vor_collection.geoms)
    gdf_voronoi = gpd.GeoDataFrame(geometry=vor_polys, crs=3857)

    beats_gdf = gpd.GeoDataFrame(
        {'beat_id': range(len(beat_centers)), 'sector_id': sector_labels},
        geometry=[Point(x, y) for x, y in beat_centers],
        crs=3857
    )
    gdf_vor_mapped = gpd.sjoin(gdf_voronoi, beats_gdf, how='inner', predicate='contains')
    gdf_beats_clipped = gpd.overlay(gdf_vor_mapped, city_boundary, how='intersection')
    gdf_sectors_viz   = gdf_beats_clipped.dissolve(by='sector_id')

    # ── 3.10  Convert selected HQs to EPSG:4326 for export ───────────────────
    hq_gdf_4326 = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(sector_hqs[:, 0], sector_hqs[:, 1]),
        crs=3857
    ).to_crs(4326)
    hq_lons = hq_gdf_4326.geometry.x.values
    hq_lats = hq_gdf_4326.geometry.y.values

    # ── 3.11  UI CSV exports ──────────────────────────────────────────────────
    print("\n[UI Export] Writing data files for the visualisation interface...")

    df_stations = export_stations_csv(
        sector_hqs_3857=sector_hqs,
        sector_hqs_4326_lon=hq_lons,
        sector_hqs_4326_lat=hq_lats,
        coverage_sets=inc_coverage_sets,
        x_sol=x_sol,
        weights=weights,
        gdf_incidents=gdf,
        beat_to_sector=beat_to_sector,
        beat_centers=beat_centers,
        output_path=OUTPUT_STATIONS,
    )

    df_incidents_export = export_incidents_csv(
        gdf_incidents=gdf_raw,           # <-- was: gdf
        coverage_sets=inc_coverage_sets,
        x_sol=x_sol,
        weights=raw_weights,             # <-- was: weights (these are cluster weights)
        coverage_counts=coverage_counts,
        inc_to_cluster=inc_to_cluster,   # <-- ADD
        output_path=OUTPUT_INCIDENTS,
    )

    export_optimization_summary(
        n_inc=n_raw_incidents,
        covered_count=true_raw_covered_count,
        pct_count=pct_count,
        pct_weight=pct_weight,
        Z_ip=Z_ip,
        O=O,
        B=B,
        total_weight=total_w,
        gap=gap,
        status=status,
        SERVICE_MI=SERVICE_MI,
        SERVICE_M=SERVICE_M,
        NUM_BEATS=NUM_BEATS,
        NUM_SECTORS=NUM_SECTORS,
        runtime_s=time.time() - t0,
        output_path=OUTPUT_OPT_SUMMARY,
    )

    export_beat_polygons_geojson(
        gdf_beats_clipped=gdf_beats_clipped,
        beat_to_sector=beat_to_sector,
        output_path=OUTPUT_BEATS_GEO,
    )

    # ── 3.12  Matplotlib map (unchanged) ─────────────────────────────────────
    print("[6/6] Rendering static map ...")
    t7 = time.time()

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

    # ── 3.13  Legacy summary CSV ──────────────────────────────────────────────
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
            n_raw_incidents, SERVICE_MI, SERVICE_M,
            NUM_BEATS, NUM_SECTORS,
            status, gap,
            round(Z_ip, 2),
            true_raw_covered_count, round(pct_count, 2),
            round(O, 2), round(pct_weight, 2),
            round(B, 2),
            round(time.time() - t0, 1),
        ]
    }).to_csv(OUTPUT_CSV, index=False)

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print("\n── UI Data Files Written ──────────────────────────────────────────────")
    print(f"  Stations:             {OUTPUT_STATIONS}")
    print(f"  Incidents:            {OUTPUT_INCIDENTS}")
    print(f"  Optimization summary: {OUTPUT_OPT_SUMMARY}")
    print(f"  Beat polygons:        {OUTPUT_BEATS_GEO}")
    print(f"  Static map:           {OUTPUT_IMG}")


if __name__ == '__main__':
    generate_patrol_map()

