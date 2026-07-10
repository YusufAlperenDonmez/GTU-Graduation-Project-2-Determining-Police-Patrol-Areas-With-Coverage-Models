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

NEW: Exports rich CSV files for UI consumption
  - outputs/beats/stations.csv        : one row per selected HQ / police station
  - outputs/beats/incidents_export.csv: one row per crime incident with assignment info
  - outputs/beats/optimization_summary.csv: overall run metrics
  - outputs/beats/beat_polygons.csv   : voronoi beat polygon vertices for drawing
"""

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
USER_POLYGON_COORDS = [[-118.322754, 34.195901], [-118.482056, 34.025348], [-118.151093, 33.950195], [-118.035736, 34.179998], [-118.322754, 34.195901]]


NUM_BEATS   = 300        # |J|  candidate facility locations
NUM_SECTORS = 10         # P    command centres to locate
SERVICE_MI  = 1.8        # S    service radius in miles
SERVICE_M   = SERVICE_MI * 1_609.34   # S in metres

OUTPUT_IMG  = '../outputs/beats/ppac_backup_exact_optimal.png'
OUTPUT_CSV  = '../outputs/beats/ppac_backup_exact_summary.csv'
OSM_CACHE   = '../resources/la_drive_network.graphml'

# ── UI Export paths ───────────────────────────────────────────────────────────
OUTPUT_STATIONS    = '../outputs/beats/backup_stations.csv'
OUTPUT_INCIDENTS   = '../outputs/beats/backup_incidents_export.csv'
OUTPUT_OPT_SUMMARY = '../outputs/beats/backup_optimization_summary.csv'
OUTPUT_BEATS_GEO   = '../outputs/beats/backup_beat_polygons.geojson'

IP_TIME_LIMIT = 7200     # iki saat cunku backup uzun suruyor
IP_MIP_GAP    = 0.005    
                        

# ── Multiobjective backup coverage (Curtin et al. 2010, Section 3.3) ─────────
# Set BACKUP_COVERAGE_O_FRACTION to a value in [0.0, 1.0].
#   1.0  -> pure maximal covering (standard MCLP, no backup enforcement)
#   0.0  -> pure maximal backup coverage (O constraint disabled)
#   0.95 -> enforce at least 95% of optimal covering O*, then maximise backup
# When < 1.0, a second IP is solved after the first using the constraint method:
#   - w_i variables replace y_i in the objective (backup coverage objective B)
#   - y_i bounds are relaxed to allow coverage_count >= 1 (not just binary)
#   - constraint  Σ a_i * w_i >= O_threshold  enforces minimum coverage level
BACKUP_COVERAGE_O_FRACTION = 0.95   # 1.0 = standard MCLP only (change to e.g. 0.95)


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
            print(f"      Routing progress: {pct:.0f}% complete ({j + 1}/{total_cands} candidates processed)")

    return coverage_sets


def solve_ppac_ip(coverage_sets: list,
                  weights: np.ndarray,
                  n_inc: int,
                  P: int,
                  time_limit: int = 1800,
                  mip_gap: float = 0.0) -> tuple:
    """
    Stage 1: Standard MCLP (Maximal Covering Location Problem).
    Maximises  Σ a_i * y_i  subject to facility cardinality P.
    Returns the upper bound O* on maximal coverage.
    """
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
    print_interval = max(1, n_inc // 20)

    for i in range(n_inc):
        if N[i]:
            prob += (pulp.lpSum(x[j] for j in N[i]) >= y[i], f"cov_{i}")
        else:
            prob += (y[i] == 0, f"uncoverable_{i}")

        if (i + 1) % print_interval == 0 or (i + 1) == n_inc:
            pct = ((i + 1) / n_inc) * 100
            print(f"      Constraint build progress: {pct:.0f}% ({i + 1}/{n_inc})")

    prob += (pulp.lpSum(x) == P, "cardinality")

    print("    Passing Stage 1 (MCLP) to CBC solver...")
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

    return x_sol, y_sol, float(pulp.value(prob.objective) or 0.0), gap, status, N


def solve_backup_ip(coverage_sets: list,
                    weights: np.ndarray,
                    n_inc: int,
                    P: int,
                    N: list,
                    O_star: float,
                    o_fraction: float,
                    time_limit: int = 1800,
                    mip_gap: float = 0.0) -> tuple:
    """
    Stage 2: Maximal Backup Coverage IP (Curtin et al. 2010, Section 3.3).

    After solving for the maximal covering upper bound O*, this model relaxes
    the y_i binary constraint and adds w_i decision variables (defined identically
    to the original y_i) along with the constraint:

        Σ a_i * w_i >= O_threshold          (eq. 7 from the paper)

    where O_threshold = o_fraction * O_star enforces a minimum coverage level.

    The objective is then to maximise backup coverage:

        maximise  Σ a_i * (number of facilities covering incident i) * y_i

    which is linearised as  Σ_j [ x_j * Σ_{i in S_j} a_i * y_i ]
    or equivalently by counting coverage:

        maximise  Σ_i  a_i * coverage_count_i

    We implement this as in the paper: y_i is now a continuous [0,1] variable
    representing fractional coverage (allowing backup), w_i is binary (covered
    by at least one), and the backup objective counts total facility-incident
    coverage mass.
    """
    n_cand = len(coverage_sets)
    O_threshold = o_fraction * O_star
    print(f"    Stage 2: Backup IP — enforcing O >= {O_threshold:,.2f} "
          f"({o_fraction*100:.1f}% of O* = {O_star:,.2f})")

    prob2 = pulp.LpProblem("PPAC_Backup_Exact", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(n_cand)]

    # y_i: relaxed to [0, len(selected)] — counts how many facilities cover i
    # We keep them binary here for tractability; the backup objective is captured
    # via the sum of coverage contributions across all selected x_j.
    # Per the paper: w_i = original binary covered/not indicator (enforces O)
    #                y_i = can now be > 1 in the objective (backup counted via x_j)
    w = [pulp.LpVariable(f"w_{i}", cat='Binary') for i in range(n_inc)]

    # Backup objective: maximise total weighted coverage count
    # = Σ_j x_j * Σ_{i in S_j} a_i  (each facility contributes its incident weights)
    # This counts every facility-incident pair, so incidents covered twice count twice.
    prob2 += pulp.lpSum(
        x[j] * sum(weights[i] for i in coverage_sets[j])
        for j in range(n_cand)
    ), "MaximalBackupCoverage"

    print(f"    Building coverage constraints for w_i (minimum covering)...")
    print_interval = max(1, n_inc // 20)
    for i in range(n_inc):
        if N[i]:
            # w_i = 1 only if at least one facility covers incident i
            prob2 += (pulp.lpSum(x[j] for j in N[i]) >= w[i], f"wcov_{i}")
        else:
            prob2 += (w[i] == 0, f"wuncov_{i}")

        if (i + 1) % print_interval == 0 or (i + 1) == n_inc:
            pct = ((i + 1) / n_inc) * 100
            print(f"      w_i constraints: {pct:.0f}% ({i + 1}/{n_inc})")

    # Minimum coverage enforcement constraint (eq. 7)
    prob2 += (
        pulp.lpSum(weights[i] * w[i] for i in range(n_inc)) >= O_threshold,
        "MinCovering"
    )

    prob2 += (pulp.lpSum(x) == P, "cardinality")

    print("    Passing Stage 2 (Backup) to CBC solver...")
    solver_kwargs = dict(msg=1)
    if time_limit > 0:
        solver_kwargs['timeLimit'] = time_limit
    if mip_gap > 0:
        solver_kwargs['gapRel'] = mip_gap

    solver = pulp.PULP_CBC_CMD(**solver_kwargs)
    prob2.solve(solver)

    status2 = pulp.LpStatus[prob2.status]
    B_val   = float(pulp.value(prob2.objective) or 0.0)
    print(f"    Backup solver status : {status2}")
    print(f"    Backup Objective B*  : {B_val:,.2f}")

    x_sol2 = np.array([pulp.value(x[j]) or 0.0 for j in range(n_cand)]) > 0.5
    w_sol  = np.array([pulp.value(w[i]) or 0.0 for i in range(n_inc)])

    try:
        gap2 = prob2.solver.solverModel.bestBound
    except Exception:
        gap2 = 0.0

    return x_sol2, w_sol, B_val, gap2, status2


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
    """
    Export one row per selected police station (HQ) with:
      station_id, lat, lon, incidents_covered, weighted_coverage,
      backup_coverage, beats_assigned, sector_label, coverage_pct
    """
    selected_idx = np.where(x_sol)[0]
    n_inc = len(weights)
    total_weight = weights.sum()

    # Build per-station coverage from the full coverage_sets
    # station k covers all incidents reachable from candidate selected_idx[k]
    rows = []
    for rank, cand_idx in enumerate(selected_idx):
        covered_incs = list(coverage_sets[cand_idx])
        n_covered = len(covered_incs)
        weighted_cov = weights[covered_incs].sum() if covered_incs else 0.0

        # Backup: how many of those incidents are also covered by another station
        backup_count = 0
        for inc_idx in covered_incs:
            # count how many selected candidates cover this incident
            coverers = sum(
                1 for sel in selected_idx if inc_idx in coverage_sets[sel]
            )
            if coverers > 1:
                backup_count += 1

        # Count beats assigned to this sector
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
                         output_path: str):
    """
    Export one row per crime incident with:
      incident_id, lat, lon, crime_weight, covered (bool),
      backup_count (how many stations cover it),
      covering_stations (list of station IDs as JSON array),
      crime_type (if available)
    """
    selected_idx = np.where(x_sol)[0]
    inc_4326 = gdf_incidents.to_crs(4326)

    n_inc = len(weights)

    # Build incident -> list of covering station_ids mapping
    inc_to_stations = [[] for _ in range(n_inc)]
    for rank, cand_idx in enumerate(selected_idx):
        for inc_idx in coverage_sets[cand_idx]:
            inc_to_stations[inc_idx].append(rank)

    rows = []
    for i in range(n_inc):
        geom = inc_4326.geometry.iloc[i]
        row = {
            'incident_id': i,
            'lat': float(geom.y),
            'lon': float(geom.x),
            'crime_weight': float(weights[i]),
            'covered': int(coverage_counts[i] >= 1),
            'backup_count': int(coverage_counts[i]),
            'covering_stations': json.dumps(inc_to_stations[i]),
            'primary_station': int(inc_to_stations[i][0]) if inc_to_stations[i] else -1,
        }
        # Carry over any extra columns from original data
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
                                 runtime_s, output_path: str,
                                 o_fraction: float = 1.0,
                                 O_star: float = None,
                                 backup_status: str = None,
                                 backup_gap: float = None,
                                 B_backup: float = None):
    """Export a single-row summary of the full optimization run."""
    row = {
        'mode': 'backup' if o_fraction < 1.0 else 'maximal_covering',
        'total_incidents': n_inc,
        'service_radius_mi': SERVICE_MI,
        'service_radius_m': round(SERVICE_M, 1),
        'num_candidates': NUM_BEATS,
        'num_sectors': NUM_SECTORS,
        'stage1_ip_status': status,
        'stage1_mip_gap': gap,
        'stage1_objective_O_star': round(O_star or Z_ip, 3),
        'backup_o_fraction': o_fraction,
        'backup_o_threshold': round((o_fraction * (O_star or Z_ip)), 3),
        'stage2_ip_status': backup_status or '',
        'stage2_mip_gap': backup_gap or '',
        'stage2_backup_obj_B': round(B_backup, 3) if B_backup is not None else '',
        'incidents_covered': covered_count,
        'coverage_pct_count': round(pct_count, 2),
        'maximal_covering_obj': round(O, 3),
        'coverage_pct_weighted': round(pct_weight, 2),
        'maximal_backup_obj': round(B, 3),
        'total_weight': round(float(total_weight), 3),
        'runtime_seconds': round(runtime_s, 1),
    }
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   Optimization summary exported -> {output_path}")


def export_beat_polygons_geojson(gdf_beats_clipped: gpd.GeoDataFrame,
                                  beat_to_sector: np.ndarray,
                                  output_path: str):
    """
    Export clipped Voronoi beat polygons as GeoJSON (EPSG:4326) for Leaflet rendering.
    Each feature carries beat_id and sector_id attributes.
    """
    out = gdf_beats_clipped.copy()
    out = out.to_crs(4326)
    # Ensure sector_id is present
    if 'sector_id' not in out.columns and 'beat_id' in out.columns:
        out['sector_id'] = out['beat_id'].map(
            lambda bid: int(beat_to_sector[bid]) if bid < len(beat_to_sector) else -1
        )
    # Keep only geometry + key attributes
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
    weights = gdf['crime_weight'].values
    n_inc   = len(gdf)
    print(f"   {n_inc:,} incidents matched inside region.  ({time.time()-t0:.1f}s)")

    if n_inc == 0:
        print("ERROR: No crime incidents found inside the computed boundary intersection.")
        return

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

    inc_gdf   = gdf.to_crs(4326)
    inc_nodes = snap_to_nodes(G_4326, inc_gdf.geometry.x.values, inc_gdf.geometry.y.values)
    print(f"   {n_inc:,} exact incident nodes snapped.  ({time.time()-t3:.1f}s)")

    # ── 3.6  Coverage sets ────────────────────────────────────────────────────
    print(f"[5/6] Building EXACT incident road-network coverage sets (S = {SERVICE_MI} mi)...")
    t4 = time.time()
    inc_coverage_sets = build_coverage_sets(G_metric, beat_nodes, inc_nodes, SERVICE_M)
    print(f"   Coverage sets built. ({time.time()-t4:.1f}s)")

    # ── 3.7  Stage 1: MCLP ───────────────────────────────────────────────────
    print(f"[5b/6] Stage 1 — MCLP (P={NUM_SECTORS}, |J|={NUM_BEATS}, |I|={n_inc}) ...")
    t5 = time.time()

    x_sol, y_sol_inc, Z_ip, gap, status, N_map = solve_ppac_ip(
        coverage_sets=inc_coverage_sets,
        weights=weights,
        n_inc=n_inc,
        P=NUM_SECTORS,
        time_limit=IP_TIME_LIMIT,
        mip_gap=IP_MIP_GAP,
    )

    selected_idx = np.where(x_sol)[0]
    sector_hqs   = beat_centers[selected_idx]   # EPSG:3857
    O_star       = Z_ip                          # upper bound from Stage 1
    print(f"   {len(selected_idx)} HQs selected.  O* = {O_star:,.2f}  ({time.time()-t5:.1f}s)")

    # ── 3.7b  Stage 2: Backup IP (only when fraction < 1.0) ──────────────────
    backup_status = None
    backup_gap    = 0.0
    B_backup      = None   # will hold backup objective if Stage 2 runs

    if BACKUP_COVERAGE_O_FRACTION < 1.0:
        print(f"[5b2/6] Stage 2 — Backup IP "
              f"(O >= {BACKUP_COVERAGE_O_FRACTION*100:.0f}% of O*={O_star:,.2f}) ...")
        t5b = time.time()
        x_sol, _, B_backup, backup_gap, backup_status = solve_backup_ip(
            coverage_sets=inc_coverage_sets,
            weights=weights,
            n_inc=n_inc,
            P=NUM_SECTORS,
            N=N_map,
            O_star=O_star,
            o_fraction=BACKUP_COVERAGE_O_FRACTION,
            time_limit=IP_TIME_LIMIT,
            mip_gap=IP_MIP_GAP,
        )
        selected_idx = np.where(x_sol)[0]
        sector_hqs   = beat_centers[selected_idx]
        print(f"   Backup solve done.  B* = {B_backup:,.2f}  ({time.time()-t5b:.1f}s)")

    # ── 3.8  Coverage evaluation ──────────────────────────────────────────────
    print("[5c/6] Evaluating final metrics ...")
    t6 = time.time()

    coverage_counts, covered_count, O, B = evaluate_coverage(
        inc_coverage_sets, x_sol, weights, n_inc
    )

    total_w    = weights.sum()
    pct_count  = 100 * covered_count / n_inc
    pct_weight = 100 * O / total_w

    print(f"\n   ── PPAC Results (S = {SERVICE_MI} mi = {SERVICE_M:.0f} m) ──")
    print(f"   Mode                          : {'Backup (Stage 2)' if BACKUP_COVERAGE_O_FRACTION < 1.0 else 'Maximal Covering (Stage 1)'}")
    print(f"   Stage 1 IP Status             : {status}")
    if gap:
        print(f"   Stage 1 MIP Gap               : {gap:.4f}")
    print(f"   Stage 1 Objective O*          : {O_star:,.2f}")
    if backup_status:
        print(f"   Stage 2 IP Status             : {backup_status}")
        if backup_gap:
            print(f"   Stage 2 MIP Gap               : {backup_gap:.4f}")
        print(f"   O threshold enforced          : {BACKUP_COVERAGE_O_FRACTION*O_star:,.2f}  ({BACKUP_COVERAGE_O_FRACTION*100:.0f}% of O*)")
        print(f"   Stage 2 Backup Obj B*         : {B_backup:,.2f}")
    print(f"   Incident coverage (count)     : {covered_count:,} / {n_inc:,}  ({pct_count:.1f} %)")
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
        gdf_incidents=gdf,
        coverage_sets=inc_coverage_sets,
        x_sol=x_sol,
        weights=weights,
        coverage_counts=coverage_counts,
        output_path=OUTPUT_INCIDENTS,
    )

    export_optimization_summary(
        n_inc=n_inc,
        covered_count=covered_count,
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
        o_fraction=BACKUP_COVERAGE_O_FRACTION,
        O_star=O_star,
        backup_status=backup_status,
        backup_gap=backup_gap,
        B_backup=B_backup,
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
            n_inc, SERVICE_MI, SERVICE_M,
            NUM_BEATS, NUM_SECTORS,
            status, gap,
            round(Z_ip, 2),
            covered_count, round(pct_count, 2),
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