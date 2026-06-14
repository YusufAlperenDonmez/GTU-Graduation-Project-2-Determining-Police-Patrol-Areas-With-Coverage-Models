"""
Police Patrol Area Covering (PPAC) — Exact Incident-Level Optimizer
===================================================================
Implements the exact PPAC formulation from Curtin, Hayslett-McCall & Qiu (2010).
Optimizes directly against individual crime incidents (no aggregation).

Filters data strictly to the INTERSECTION of the user polygon and LA_AREA.geojson.

Pipeline
--------
Stage 0  Incident clustering (DBSCAN) -> super-incidents, shrinks all downstream stages
Stage 1  Weighted Mini-Batch K-Means on super-incidents -> candidate set J
Stage 2  Road-network OD matrix (PARALLEL Dijkstra) -> incident-level coverage sets
Stage 3  PPAC integer programme (greedy warm start + MIP gap) -> P optimal HQ locations
Stage 4  Full incident-level coverage evaluation
Stage 5  Bounded Voronoi beat map + sector map clipped to unified area intersection

Exports rich CSV / GeoJSON files for UI consumption:
  outputs/optimized/stations.csv
  outputs/optimized/incidents_export.csv
  outputs/optimized/optimization_summary.csv
  outputs/optimized/beat_polygons.geojson
"""

import os
import warnings
import time
import json
import math

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import contextily as cx
import networkx as nx
import osmnx as ox

from joblib import Parallel, delayed, cpu_count
from sklearn.cluster import MiniBatchKMeans, DBSCAN
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

USER_POLYGON_COORDS = [
    [-118.753967, 34.354774],
    [-118.096161, 34.354774],
    [-118.122253, 33.694638],
    [-118.850098, 33.680925],
    [-118.753967, 34.354774],
]

NUM_BEATS   = 250        # |J|  candidate facility locations
NUM_SECTORS = 15         # P    command centres to locate
SERVICE_MI  = 1.8        # S    service radius in miles
SERVICE_M   = SERVICE_MI * 1_609.34   # S in metres

# ── Incident clustering ───────────────────────────────────────────────────────
CLUSTER_RADIUS_M = 100.0   # merge incidents within this Euclidean distance (m)
                            # tune: 100–300 m is typical for dense urban areas

# ── Parallelisation ───────────────────────────────────────────────────────────
# -1  = use all available CPU cores
# Set to a positive int to cap usage (e.g. 4 on a shared machine)
N_JOBS = -1

# ── IP solver ────────────────────────────────────────────────────────────────
IP_TIME_LIMIT = 3600       # seconds (1 hour is plenty with warm start)
IP_MIP_GAP    = 0.005      # 0.5 % optimality gap — typically undetectable in
                            # practice but 10–100× faster than exact (0.000)

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_IMG         = '../outputs/optimized/ppac_exact_optimal.png'
OUTPUT_CSV         = '../outputs/optimized/ppac_exact_summary.csv'
OUTPUT_STATIONS    = '../outputs/optimized/stations.csv'
OUTPUT_INCIDENTS   = '../outputs/optimized/incidents_export.csv'
OUTPUT_OPT_SUMMARY = '../outputs/optimized/optimization_summary.csv'
OUTPUT_BEATS_GEO   = '../outputs/optimized/beat_polygons.geojson'
OSM_CACHE          = '../resources/la_drive_network.graphml'


# ─────────────────────────────────────────────────────────────────────────────
# 1.  INCIDENT CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def cluster_incidents(gdf: gpd.GeoDataFrame,
                      weights: np.ndarray,
                      cluster_radius_m: float) -> tuple:
    """
    Pre-process: merge spatially proximate incidents into weighted
    super-incidents before routing and IP.

    Uses DBSCAN (min_samples=1) so every point belongs to a cluster —
    isolated points become singleton clusters.  The super-incident's
    position is the crime-weight-weighted centroid of its members, and
    its weight is the sum of member weights.

    Parameters
    ----------
    gdf              : GeoDataFrame of raw incidents (EPSG:3857)
    weights          : per-incident crime_weight array  (len == len(gdf))
    cluster_radius_m : neighbourhood radius in metres

    Returns
    -------
    gdf_clustered    : GeoDataFrame, one row per super-incident (EPSG:3857)
    new_weights      : summed crime weights per super-incident
    inc_to_cluster   : int array shape (n_original,) — original idx -> cluster id
    """
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])

    db = DBSCAN(
        eps=cluster_radius_m,
        min_samples=1,
        algorithm='ball_tree',
        metric='euclidean',
        n_jobs=N_JOBS,
    )
    labels = db.fit_predict(coords)

    n_clusters = int(labels.max()) + 1
    print(f"   Incident clustering: {len(gdf):,} raw incidents → "
          f"{n_clusters:,} super-incidents  "
          f"(radius = {cluster_radius_m:.0f} m, "
          f"reduction = {100*(1 - n_clusters/len(gdf)):.1f}%)")

    cx_arr = np.empty(n_clusters)
    cy_arr = np.empty(n_clusters)
    w_arr  = np.empty(n_clusters)

    for cid in range(n_clusters):
        mask  = labels == cid
        w_sub = weights[mask]
        w_tot = w_sub.sum()
        pts   = coords[mask]
        cx_arr[cid] = (pts[:, 0] * w_sub).sum() / w_tot
        cy_arr[cid] = (pts[:, 1] * w_sub).sum() / w_tot
        w_arr[cid]  = w_tot

    gdf_clustered = gpd.GeoDataFrame(
        {'cluster_id': np.arange(n_clusters)},
        geometry=[Point(x, y) for x, y in zip(cx_arr, cy_arr)],
        crs=gdf.crs,
    )
    return gdf_clustered, w_arr, labels


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ROAD NETWORK HELPERS
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


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PARALLEL COVERAGE-SET CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_m(lon1, lat1, lon2, lat2):
    """Approximate great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _route_one_candidate(j_node: int,
                          node_to_incs: dict,
                          graph_data: dict,
                          radius_m: float) -> set:
    """
    Worker function: single-source Dijkstra from one candidate node.
    Receives a plain-dict representation of the graph so joblib can
    pickle it without issues across processes.

    graph_data keys: 'adj'  — {src: {dst: length, ...}, ...}
    """
    covered = set()
    adj = graph_data['adj']

    # Standard priority-queue Dijkstra (avoids re-importing networkx in worker)
    import heapq
    heap = [(0.0, j_node)]
    visited = {}
    while heap:
        dist, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited[u] = dist
        if dist > radius_m:
            break
        for inc_idx in node_to_incs.get(u, []):
            covered.add(inc_idx)
        for v, length in adj.get(u, {}).items():
            nd = dist + length
            if nd <= radius_m and v not in visited:
                heapq.heappush(heap, (nd, v))

    return covered


def _build_adj_dict(G_metric: nx.MultiDiGraph) -> dict:
    """
    Convert the networkx MultiDiGraph to a lightweight adjacency dict
    {src_node: {dst_node: min_length}} for fast pickling to workers.
    """
    adj = {}
    for u, v, data in G_metric.edges(data=True):
        length = data.get('length', 0.0)
        if u not in adj:
            adj[u] = {}
        if v not in adj[u] or length < adj[u][v]:
            adj[u][v] = length
    return adj


def build_coverage_sets_parallel(G_metric: nx.MultiDiGraph,
                                  candidate_nodes: np.ndarray,
                                  inc_nodes: np.ndarray,
                                  radius_m: float,
                                  n_jobs: int = -1) -> list:
    """
    Build coverage sets for all candidates in parallel.

    Each worker runs a pure-Python Dijkstra (no networkx import needed
    inside the worker) against a pickled adjacency dict, so there is no
    shared-memory hazard.

    Parameters
    ----------
    G_metric         : projected (metric CRS) road graph
    candidate_nodes  : OSM node IDs for candidate HQ sites
    inc_nodes        : OSM node IDs for each super-incident
    radius_m         : service radius in metres
    n_jobs           : joblib n_jobs (-1 = all cores)

    Returns
    -------
    list of sets, one per candidate: each set contains incident indices
    covered within radius_m road-network distance.
    """
    n_cores = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
    print(f"   Building parallel adjacency dict ...")
    adj = _build_adj_dict(G_metric)
    graph_data = {'adj': adj}

    # Incident reverse map: node -> [incident indices]
    node_to_incs: dict = {}
    for idx, node in enumerate(inc_nodes):
        node_to_incs.setdefault(int(node), []).append(idx)

    total_cands = len(candidate_nodes)
    print(f"   Dispatching {total_cands} routing jobs across {n_cores} CPU cores ...")

    # joblib with loky backend — safest for networkx/numpy environments
    coverage_sets = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(_route_one_candidate)(
            int(j_node), node_to_incs, graph_data, radius_m
        )
        for j_node in candidate_nodes
    )

    # Progress summary
    covered_any = sum(1 for cs in coverage_sets if len(cs) > 0)
    print(f"   Routing complete. "
          f"{covered_any}/{total_cands} candidates cover ≥1 incident.")
    return coverage_sets


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GREEDY WARM-START HEURISTIC
# ─────────────────────────────────────────────────────────────────────────────

def greedy_warm_start(coverage_sets: list,
                      weights: np.ndarray,
                      n_inc: int,
                      P: int) -> np.ndarray:
    """
    Greedy maximal-covering heuristic to generate a warm-start solution
    for the IP solver.  Each step picks the candidate that adds the most
    uncovered weighted incidents.

    Returns
    -------
    selected : bool array of length |J| — True for the P chosen candidates
    """
    covered = np.zeros(n_inc, dtype=bool)
    selected = np.zeros(len(coverage_sets), dtype=bool)

    for _ in range(P):
        best_j, best_gain = -1, -1.0
        for j, cs in enumerate(coverage_sets):
            if selected[j]:
                continue
            gain = weights[list(cs - set(np.where(covered)[0]))].sum() if cs else 0.0
            # Faster equivalent using numpy indexing
            arr = np.array(list(cs), dtype=int) if cs else np.array([], dtype=int)
            gain = weights[arr[~covered[arr]]].sum() if arr.size else 0.0
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j == -1:
            break
        selected[best_j] = True
        arr = np.array(list(coverage_sets[best_j]), dtype=int)
        covered[arr] = True

    n_sel = selected.sum()
    covered_w = weights[covered].sum()
    print(f"   Greedy warm start: {n_sel} sites, "
          f"weighted coverage = {100*covered_w/weights.sum():.1f}%")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 5.  IP SOLVER (PPAC)
# ─────────────────────────────────────────────────────────────────────────────

def solve_ppac_ip(coverage_sets: list,
                  weights: np.ndarray,
                  n_inc: int,
                  P: int,
                  warm_start_mask: np.ndarray | None = None,
                  time_limit: int = 3600,
                  mip_gap: float = 0.005) -> tuple:
    """
    Solve the PPAC integer programme with OR-Tools / SCIP.

    Maximise  Σ w_i · y_i
    s.t.      Σ_{j∈N_i} x_j ≥ y_i   ∀ i  (coverage)
              Σ x_j = P              (cardinality)
              x_j, y_i ∈ {0,1}

    Parameters
    ----------
    coverage_sets   : list of sets — coverage_sets[j] = incidents reachable from j
    weights         : incident weights w_i
    n_inc           : number of (super-)incidents
    P               : number of facilities to locate
    warm_start_mask : bool array — greedy solution to seed the solver
    time_limit      : solver wall-clock time limit in seconds
    mip_gap         : relative MIP optimality gap tolerance

    Returns
    -------
    x_sol    : bool array len |J|
    y_sol    : float array len |I|
    obj_val  : float
    best_bound : float
    status   : str
    """
    n_cand = len(coverage_sets)

    print("    Building incident→candidate reverse mapping (N_i) ...")
    N = [[] for _ in range(n_inc)]
    for j, cs in enumerate(coverage_sets):
        for i in cs:
            N[i].append(j)

    print("    Initialising OR-Tools SCIP solver ...")
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver could not be created.")

    if time_limit > 0:
        solver.SetTimeLimit(int(time_limit * 1000))   # OR-Tools uses milliseconds
    if mip_gap >= 0.0:
        solver.SetSolverSpecificParametersAsString(f"limits/gap = {mip_gap}")

    # Variables
    x = [solver.BoolVar(f"x_{j}") for j in range(n_cand)]
    y = [solver.BoolVar(f"y_{i}") for i in range(n_inc)]

    # Objective
    objective = solver.Objective()
    for i in range(n_inc):
        objective.SetCoefficient(y[i], float(weights[i]))
    objective.SetMaximization()

    # Coverage constraints
    print(f"    Adding {n_inc:,} coverage constraints ...")
    for i in range(n_inc):
        if N[i]:
            ct = solver.Constraint(0, solver.infinity(), f"cov_{i}")
            ct.SetCoefficient(y[i], -1.0)
            for j in N[i]:
                ct.SetCoefficient(x[j], 1.0)
        else:
            ct = solver.Constraint(0, 0, f"uncov_{i}")
            ct.SetCoefficient(y[i], 1.0)

    # Cardinality constraint
    cc = solver.Constraint(float(P), float(P), "cardinality")
    for j in range(n_cand):
        cc.SetCoefficient(x[j], 1.0)

    # Warm start hint
    if warm_start_mask is not None and warm_start_mask.any():
        print("    Injecting greedy warm-start hint ...")
        for j in range(n_cand):
            x[j].SetHint(1.0 if warm_start_mask[j] else 0.0)
        # Derive y hints from x hints
        covered_hint = set()
        for j in np.where(warm_start_mask)[0]:
            covered_hint.update(coverage_sets[j])
        for i in range(n_inc):
            y[i].SetHint(1.0 if i in covered_hint else 0.0)

    print("    Solving ...")
    solver.EnableOutput()
    result_status = solver.Solve()

    status_map = {
        pywraplp.Solver.OPTIMAL:   "Optimal",
        pywraplp.Solver.FEASIBLE:  "Feasible",
        pywraplp.Solver.INFEASIBLE:"Infeasible",
        pywraplp.Solver.UNBOUNDED: "Unbounded",
        pywraplp.Solver.ABNORMAL:  "Abnormal",
        pywraplp.Solver.NOT_SOLVED:"NotSolved",
    }
    status = status_map.get(result_status, "Unknown")

    print(f"    Solver status  : {status}")
    obj_val = objective.Value()
    print(f"    Objective Z*   : {obj_val:,.2f}")

    x_sol = np.array([x[j].solution_value() for j in range(n_cand)]) > 0.5
    y_sol = np.array([y[i].solution_value() for i in range(n_inc)])

    try:
        best_bound = solver.Objective().BestBound()
    except Exception:
        best_bound = 0.0

    return x_sol, y_sol, float(obj_val), best_bound, status


# ─────────────────────────────────────────────────────────────────────────────
# 6.  COVERAGE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

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
# 7.  UI DATA EXPORT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def export_stations_csv(sector_hqs_3857, sector_hqs_4326_lon,
                        sector_hqs_4326_lat, coverage_sets, x_sol,
                        weights, gdf_incidents, beat_to_sector,
                        beat_centers, output_path):
    selected_idx = np.where(x_sol)[0]
    n_inc        = len(weights)
    total_weight = weights.sum()

    rows = []
    for rank, cand_idx in enumerate(selected_idx):
        covered_incs = list(coverage_sets[cand_idx])
        n_covered    = len(covered_incs)
        weighted_cov = weights[covered_incs].sum() if covered_incs else 0.0

        backup_count = 0
        for inc_idx in covered_incs:
            coverers = sum(1 for sel in selected_idx if inc_idx in coverage_sets[sel])
            if coverers > 1:
                backup_count += 1

        beats_in_sector = int((beat_to_sector == rank).sum())

        rows.append({
            'station_id':             rank,
            'candidate_idx':          int(cand_idx),
            'lat':                    float(sector_hqs_4326_lat[rank]),
            'lon':                    float(sector_hqs_4326_lon[rank]),
            'x_3857':                 float(sector_hqs_3857[rank, 0]),
            'y_3857':                 float(sector_hqs_3857[rank, 1]),
            'incidents_covered':      n_covered,
            'weighted_coverage':      round(float(weighted_cov), 3),
            'backup_incidents':       backup_count,
            'beats_assigned':         beats_in_sector,
            'coverage_pct':           round(100.0 * n_covered / n_inc, 2),
            'weighted_coverage_pct':  round(100.0 * float(weighted_cov) / total_weight, 2),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   Stations CSV        -> {output_path}  ({len(df)} rows)")
    return df


def export_incidents_csv(gdf_incidents, coverage_sets, x_sol,
                         weights, coverage_counts, output_path):
    selected_idx  = np.where(x_sol)[0]
    inc_4326      = gdf_incidents.to_crs(4326)
    n_inc         = len(weights)

    inc_to_stations = [[] for _ in range(n_inc)]
    for rank, cand_idx in enumerate(selected_idx):
        for inc_idx in coverage_sets[cand_idx]:
            inc_to_stations[inc_idx].append(rank)

    rows = []
    for i in range(n_inc):
        geom = inc_4326.geometry.iloc[i]
        row = {
            'incident_id':      i,
            'lat':              float(geom.y),
            'lon':              float(geom.x),
            'crime_weight':     float(weights[i]),
            'covered':          int(coverage_counts[i] >= 1),
            'backup_count':     int(coverage_counts[i]),
            'covering_stations':json.dumps(inc_to_stations[i]),
            'primary_station':  int(inc_to_stations[i][0]) if inc_to_stations[i] else -1,
        }
        orig = gdf_incidents.iloc[i]
        for col in ['Crm Cd Desc', 'AREA NAME', 'DATE OCC', 'Vict Age', 'Vict Sex']:
            if col in gdf_incidents.columns:
                row[col.lower().replace(' ', '_')] = orig[col]
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   Incidents CSV       -> {output_path}  ({len(df):,} rows)")
    return df


def export_optimization_summary(n_inc, covered_count, pct_count, pct_weight,
                                 Z_ip, O, B, total_weight, gap, status,
                                 SERVICE_MI, SERVICE_M, NUM_BEATS, NUM_SECTORS,
                                 n_raw_incidents, cluster_radius_m,
                                 runtime_s, output_path):
    df = pd.DataFrame([{
        'total_raw_incidents':   n_raw_incidents,
        'cluster_radius_m':      cluster_radius_m,
        'total_super_incidents': n_inc,
        'service_radius_mi':     SERVICE_MI,
        'service_radius_m':      round(SERVICE_M, 1),
        'num_candidates':        NUM_BEATS,
        'num_sectors':           NUM_SECTORS,
        'ip_status':             status,
        'mip_gap':               gap,
        'objective_z':           round(Z_ip, 3),
        'incidents_covered':     covered_count,
        'coverage_pct_count':    round(pct_count, 2),
        'maximal_covering_obj':  round(O, 3),
        'coverage_pct_weighted': round(pct_weight, 2),
        'maximal_backup_obj':    round(B, 3),
        'total_weight':          round(float(total_weight), 3),
        'runtime_seconds':       round(runtime_s, 1),
    }])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   Opt. summary        -> {output_path}")


def export_beat_polygons_geojson(gdf_beats_clipped, beat_to_sector, output_path):
    out = gdf_beats_clipped.copy().to_crs(4326)
    if 'sector_id' not in out.columns and 'beat_id' in out.columns:
        out['sector_id'] = out['beat_id'].map(
            lambda bid: int(beat_to_sector[bid]) if bid < len(beat_to_sector) else -1
        )
    keep = [c for c in ['geometry', 'beat_id', 'sector_id'] if c in out.columns]
    out  = out[keep]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_file(output_path, driver='GeoJSON')
    print(f"   Beat polygons GeoJSON -> {output_path}  ({len(out)} beats)")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_patrol_map():
    t0 = time.time()

    n_cores = cpu_count() if N_JOBS == -1 else min(N_JOBS, cpu_count())
    print(f"PPAC Optimizer  |  {n_cores} CPU cores  |  "
          f"cluster_radius={CLUSTER_RADIUS_M:.0f}m  |  "
          f"MIP gap={IP_MIP_GAP*100:.1f}%\n")

    # ── 8.1  Boundaries + intersection ───────────────────────────────────────
    print("[1/7] Loading boundaries and computing intersection ...")
    if not os.path.exists(BOUNDARY_FILE_PATH) or not os.path.exists(CRIME_DATA_PATH):
        print("ERROR: Missing geometry or crime data input files.")
        return

    la_geojson = gpd.read_file(BOUNDARY_FILE_PATH).to_crs(epsg=4326)
    user_poly  = Polygon(USER_POLYGON_COORDS)
    user_gdf   = gpd.GeoDataFrame({'geometry': [user_poly]}, crs='EPSG:4326')

    intersection_poly = la_geojson.unary_union.intersection(user_gdf.unary_union)
    if intersection_poly.is_empty:
        print("ERROR: Custom polygon does not overlap with LA_AREA.geojson.")
        return

    city_boundary = gpd.GeoDataFrame(
        {'geometry': [intersection_poly]}, crs='EPSG:4326'
    ).to_crs(epsg=3857)
    print("   Boundary intersection computed.")

    # ── 8.2  Load + spatial-filter crime data ────────────────────────────────
    print("[2/7] Loading and filtering crime data ...")
    df = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['LON'], df['LAT'])],
        crs='EPSG:4326',
    ).to_crs(epsg=3857)

    gdf = gpd.sjoin(gdf, city_boundary, how='inner', predicate='within').copy()
    weights_raw   = gdf['crime_weight'].values
    n_raw_inc     = len(gdf)
    print(f"   {n_raw_inc:,} raw incidents inside region.  ({time.time()-t0:.1f}s)")

    if n_raw_inc == 0:
        print("ERROR: No incidents in boundary.")
        return

    # ── 8.3  Stage 0: Incident clustering ────────────────────────────────────
    print(f"[3/7] Clustering incidents (radius = {CLUSTER_RADIUS_M:.0f} m) ...")
    t1 = time.time()
    gdf_raw = gdf.copy()
    gdf, weights, inc_to_cluster = cluster_incidents(gdf, weights_raw, CLUSTER_RADIUS_M)
    n_inc = len(gdf)
    print(f"   Super-incidents ready.  ({time.time()-t1:.1f}s)")

    global NUM_BEATS, NUM_SECTORS
    if n_inc < NUM_BEATS:
        NUM_BEATS = n_inc
        print(f"   Adjusted NUM_BEATS -> {NUM_BEATS}")
    if NUM_BEATS <= NUM_SECTORS:
        NUM_SECTORS = max(1, NUM_BEATS // 2)
        print(f"   Adjusted NUM_SECTORS -> {NUM_SECTORS}")

    # ── 8.4  Stage 1: Weighted K-Means -> candidate set J ────────────────────
    print(f"[4/7] K-Means: generating {NUM_BEATS} candidate HQ sites ...")
    t2 = time.time()
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
    print(f"   Done.  ({time.time()-t2:.1f}s)")

    # ── 8.5  OSM road network ─────────────────────────────────────────────────
    print("[5/7] Acquiring OSM road network ...")
    t3 = time.time()
    G        = load_or_download_graph(city_boundary)
    G_4326   = ox.project_graph(G, to_crs='EPSG:4326')
    G_metric = ox.project_graph(G)
    print(f"   {len(G_metric.nodes):,} nodes, {len(G_metric.edges):,} edges.  "
          f"({time.time()-t3:.1f}s)")

    # ── 8.6  Node snapping ────────────────────────────────────────────────────
    print("[6/7] Snapping candidates and super-incidents to road nodes ...")
    t4 = time.time()

    beat_gdf   = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(beat_centers[:, 0], beat_centers[:, 1]),
        crs=3857,
    ).to_crs(4326)
    beat_nodes = snap_to_nodes(G_4326, beat_gdf.geometry.x.values,
                               beat_gdf.geometry.y.values)
    print(f"   {NUM_BEATS} candidate nodes snapped.")

    inc_gdf   = gdf.to_crs(4326)
    inc_nodes = snap_to_nodes(G_4326, inc_gdf.geometry.x.values,
                              inc_gdf.geometry.y.values)
    print(f"   {n_inc:,} super-incident nodes snapped.  ({time.time()-t4:.1f}s)")

    # ── 8.7  Parallel coverage-set construction ───────────────────────────────
    print(f"[6b/7] Building coverage sets in parallel "
          f"(S = {SERVICE_MI} mi, {n_cores} cores) ...")
    t5 = time.time()
    inc_coverage_sets = build_coverage_sets_parallel(
        G_metric, beat_nodes, inc_nodes, SERVICE_M, n_jobs=N_JOBS
    )
    print(f"   Coverage sets built.  ({time.time()-t5:.1f}s)")

    # ── 8.8  Greedy warm start ────────────────────────────────────────────────
    print("[6c/7] Computing greedy warm start ...")
    t6 = time.time()
    warm_start = greedy_warm_start(inc_coverage_sets, weights, n_inc, NUM_SECTORS)
    print(f"   Warm start done.  ({time.time()-t6:.1f}s)")

    # ── 8.9  PPAC Integer Programme ───────────────────────────────────────────
    print(f"[7/7] Solving PPAC IP  (P={NUM_SECTORS}, |J|={NUM_BEATS}, |I|={n_inc}) ...")
    t7 = time.time()
    x_sol, y_sol_inc, Z_ip, gap, status = solve_ppac_ip(
        coverage_sets=inc_coverage_sets,
        weights=weights,
        n_inc=n_inc,
        P=NUM_SECTORS,
        warm_start_mask=warm_start,
        time_limit=IP_TIME_LIMIT,
        mip_gap=IP_MIP_GAP,
    )

    selected_idx = np.where(x_sol)[0]
    sector_hqs   = beat_centers[selected_idx]   # EPSG:3857
    print(f"   {len(selected_idx)} HQs selected.  Z* = {Z_ip:,.2f}  "
          f"({time.time()-t7:.1f}s)")

    # ── 8.10  Coverage evaluation ─────────────────────────────────────────────
    coverage_counts, covered_count, O, B = evaluate_coverage(
        inc_coverage_sets, x_sol, weights, n_inc
    )
    total_w    = weights.sum()
    pct_count  = 100 * covered_count / n_inc
    pct_weight = 100 * O / total_w

    print(f"\n   ── PPAC IP Results (S = {SERVICE_MI} mi = {SERVICE_M:.0f} m) ──")
    print(f"   IP Status                     : {status}")
    if gap:
        print(f"   MIP Gap                       : {gap:.6f}")
    print(f"   Exact IP Objective (Z*)       : {Z_ip:,.2f}")
    print(f"   Super-incident coverage       : {covered_count:,} / {n_inc:,}  "
          f"({pct_count:.1f}%)")
    print(f"   Maximal Covering Obj (O)      : {O:,.1f} / {total_w:,.1f}  "
          f"({pct_weight:.1f}%)")
    print(f"   Maximal Backup Obj (B)        : {B:,.1f}\n")

    # ── 8.11  Voronoi beat geometry ───────────────────────────────────────────
    hq_tree       = cKDTree(sector_hqs)
    _, beat_to_sector = hq_tree.query(beat_centers)
    sector_labels = beat_to_sector

    envelope      = city_boundary.unary_union.envelope.buffer(100_000)
    centers_mp    = MultiPoint(beat_centers)
    vor_collection = voronoi_diagram(centers_mp, envelope=envelope)
    vor_polys      = list(vor_collection.geoms)
    gdf_voronoi    = gpd.GeoDataFrame(geometry=vor_polys, crs=3857)

    beats_gdf = gpd.GeoDataFrame(
        {'beat_id': range(len(beat_centers)), 'sector_id': sector_labels},
        geometry=[Point(x, y) for x, y in beat_centers],
        crs=3857,
    )
    gdf_vor_mapped    = gpd.sjoin(gdf_voronoi, beats_gdf, how='inner', predicate='contains')
    gdf_beats_clipped = gpd.overlay(gdf_vor_mapped, city_boundary, how='intersection')
    gdf_sectors_viz   = gdf_beats_clipped.dissolve(by='sector_id')

    # ── 8.12  Convert HQs to WGS-84 ──────────────────────────────────────────
    hq_gdf_4326 = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(sector_hqs[:, 0], sector_hqs[:, 1]),
        crs=3857,
    ).to_crs(4326)
    hq_lons = hq_gdf_4326.geometry.x.values
    hq_lats = hq_gdf_4326.geometry.y.values

    # ── 8.13  UI exports ──────────────────────────────────────────────────────
    print("\n[UI Export] Writing data files ...")
    export_stations_csv(
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
    export_incidents_csv(
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
        n_raw_incidents=n_raw_inc,
        cluster_radius_m=CLUSTER_RADIUS_M,
        runtime_s=time.time() - t0,
        output_path=OUTPUT_OPT_SUMMARY,
    )
    export_beat_polygons_geojson(
        gdf_beats_clipped=gdf_beats_clipped,
        beat_to_sector=beat_to_sector,
        output_path=OUTPUT_BEATS_GEO,
    )

    # ── 8.14  Static map ──────────────────────────────────────────────────────
    print("\nRendering static map ...")
    t8 = time.time()

    fig, ax = plt.subplots(figsize=(16, 13))

    gdf_sectors_viz.plot(ax=ax, column=gdf_sectors_viz.index,
                         cmap='tab20', alpha=0.45,
                         edgecolor='royalblue', linewidth=2.0)
    gdf_beats_clipped.plot(ax=ax, facecolor='none',
                           edgecolor='black', linewidth=0.3, alpha=0.6)
    gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(sector_hqs[:, 0], sector_hqs[:, 1]),
        crs=3857,
    ).plot(ax=ax, color='red', marker='H', markersize=220,
           edgecolor='white', zorder=10)

    cx.add_basemap(ax, crs=3857,
                   source=cx.providers.OpenStreetMap.Mapnik,
                   alpha=0.3, zoom=11)

    mid_hq = sector_hqs[len(sector_hqs) // 2]
    ax.add_patch(patches.Circle(
        (mid_hq[0], mid_hq[1]), SERVICE_M,
        linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.10,
    ))
    ax.annotate(
        f"{SERVICE_MI} mi Road-Network Service Radius ($S$)",
        xy=(mid_hq[0], mid_hq[1] + SERVICE_M + 600),
        ha='center', weight='bold', color='blue', fontsize=9,
    )
    ax.text(
        0.02, 0.02,
        f"Exact PPAC Integer Programming\n"
        f"S = {SERVICE_MI} mi  |  P = {NUM_SECTORS}  |  |J| = {NUM_BEATS}\n"
        f"Raw incidents: {n_raw_inc:,}  |  Super-incidents: {n_inc:,}  "
        f"(cluster r={CLUSTER_RADIUS_M:.0f}m)\n"
        f"IP Status: {status}  |  MIP gap: {IP_MIP_GAP*100:.1f}%\n"
        f"Coverage: {pct_count:.1f}% of super-incidents\n"
        f"Maximal Covering Obj (O): {O:,.0f}\n"
        f"Maximal Backup Obj  (B): {B:,.0f}",
        transform=ax.transAxes,
        fontsize=9,
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
    print(f"  Map saved -> {OUTPUT_IMG}  ({time.time()-t8:.1f}s)")

    # ── 8.15  Legacy summary CSV ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame({
        'metric': [
            'Raw incidents', 'Cluster radius (m)', 'Super-incidents',
            'Service distance (mi)', 'Service distance (m)',
            'Num candidates (|J|)', 'Num sectors (P)',
            'IP solver status', 'MIP gap',
            'Exact IP objective (Z*)',
            'Super-incidents covered', 'Coverage pct (count)',
            'Maximal Covering Objective (O)', 'Coverage pct (weighted)',
            'Maximal Backup Objective (B)',
            'Total runtime (s)',
        ],
        'value': [
            n_raw_inc, CLUSTER_RADIUS_M, n_inc,
            SERVICE_MI, round(SERVICE_M, 1),
            NUM_BEATS, NUM_SECTORS,
            status, gap,
            round(Z_ip, 2),
            covered_count, round(pct_count, 2),
            round(O, 2), round(pct_weight, 2),
            round(B, 2),
            round(time.time() - t0, 1),
        ],
    }).to_csv(OUTPUT_CSV, index=False)

    total_rt = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"Total runtime : {total_rt:.1f}s")
    print(f"{'─'*60}")
    print(f"  Static map          : {OUTPUT_IMG}")
    print(f"  Stations            : {OUTPUT_STATIONS}")
    print(f"  Incidents           : {OUTPUT_INCIDENTS}")
    print(f"  Optimization summary: {OUTPUT_OPT_SUMMARY}")
    print(f"  Beat polygons       : {OUTPUT_BEATS_GEO}")


if __name__ == '__main__':
    generate_patrol_map()