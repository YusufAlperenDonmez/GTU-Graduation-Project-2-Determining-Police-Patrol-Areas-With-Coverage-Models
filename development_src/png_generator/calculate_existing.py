"""
Existing Police Patrol Coverage Evaluator — Exact Optimizer Mirror Match Edition
=====================================================================
Evaluates actual coverage of existing LAPD police stations using the 
exact routing logic, boundary clipping, and metric calculations as the optimizer.
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
from shapely.geometry import Point, Polygon

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION (Mirrored from Optimizer)
# ─────────────────────────────────────────────
CRIME_DATA_PATH    = '../../resources/cleaned_data.csv'
BOUNDARY_FILE_PATH = '../../resources/LA_AREA.geojson' 
STATIONS_FILE_PATH = '../../resources/LAPD_Police_Stations.csv' 

# User-defined sub-area boundary coordinates (Identical to Optimizer)
USER_POLYGON_COORDS = [[-118.753967, 34.354774], [-118.096161, 34.354774], [-118.122253, 33.694638], [-118.850098, 33.680925], [-118.753967, 34.354774]]

SERVICE_MI   = 2.0  # Set to 2.0 to match your optimizer configuration
SERVICE_M    = SERVICE_MI * 1_609.34

OUTPUT_IMG   = '../../outputs/beats/existing_coverage_analysis.png'
OUTPUT_CSV   = '../../outputs/beats/existing_coverage_summary.csv'
OSM_CACHE    = '../../resources/la_drive_network.graphml'

# ── EXACT MIRROR EXPORT PATHS ────────────────────────────────────────────────
OUTPUT_STATIONS    = '../../outputs/beats/existing_stations.csv'
OUTPUT_OPT_SUMMARY = '../../outputs/beats/existing_coverage_summary.csv'

# ─────────────────────────────────────────────
# 1. HELPERS (Mirrored from Optimizer)
# ─────────────────────────────────────────────

def load_or_download_graph(boundary_gdf: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    if os.path.exists(OSM_CACHE):
        print("  Loading cached OSM road network ...")
        return ox.load_graphml(OSM_CACHE)
    print("  Downloading OSM road network ...")
    poly = boundary_gdf.to_crs(epsg=4326).unary_union.convex_hull
    G = ox.graph_from_polygon(poly, network_type='drive')
    os.makedirs(os.path.dirname(OSM_CACHE), exist_ok=True)
    ox.save_graphml(G, OSM_CACHE)
    return G

def snap_to_nodes(G_4326: nx.MultiDiGraph, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    return np.array(ox.nearest_nodes(G_4326, X=lons, Y=lats))

def build_existing_coverage_sets(G_metric: nx.MultiDiGraph, 
                                 hq_nodes: np.ndarray, 
                                 inc_nodes: np.ndarray, 
                                 radius_m: float) -> list:
    """Computes exact coverage maps using identical dictionary matching optimization engine."""
    node_to_incs = {}
    for idx, node in enumerate(inc_nodes):
        node_to_incs.setdefault(int(node), []).append(idx)

    coverage_sets = []
    for hq_node in hq_nodes:
        covered = set()
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G_metric, int(hq_node), cutoff=radius_m, weight='length'
            )
            for node in lengths:
                for inc_idx in node_to_incs.get(node, []):
                    covered.add(inc_idx)
        except Exception:
            pass
        coverage_sets.append(covered)
    return coverage_sets

def evaluate_coverage(coverage_sets: list, weights: np.ndarray, n_inc: int) -> tuple:
    coverage_counts = np.zeros(n_inc, dtype=int)
    for cs in coverage_sets:
        for inc_idx in cs:
            coverage_counts[inc_idx] += 1

    covered_mask = coverage_counts >= 1
    O = weights[covered_mask].sum()
    B = (weights * coverage_counts).sum()
    return coverage_counts, int(covered_mask.sum()), float(O), float(B)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MATCHED EXPORT ROUTINES
# ─────────────────────────────────────────────────────────────────────────────

def export_stations_csv(gdf_stations: gpd.GeoDataFrame,
                        coverage_sets: list,
                        weights: np.ndarray,
                        output_path: str):
    n_inc = len(weights)
    total_weight = weights.sum()
    
    stations_4326 = gdf_stations.to_crs(4326)
    hq_lons = stations_4326.geometry.x.values
    hq_lats = stations_4326.geometry.y.values
    x_3857 = gdf_stations.geometry.x.values
    y_3857 = gdf_stations.geometry.y.values

    rows = []
    for rank in range(len(gdf_stations)):
        covered_incs = list(coverage_sets[rank])
        n_covered = len(covered_incs)
        weighted_cov = weights[covered_incs].sum() if covered_incs else 0.0

        backup_count = 0
        for inc_idx in covered_incs:
            coverers = sum(1 for cs in coverage_sets if inc_idx in cs)
            if coverers > 1:
                backup_count += 1

        rows.append({
            'station_id': rank,
            'candidate_idx': int(rank),  # In baseline, candidate index aligns directly with rank
            'lat': float(hq_lats[rank]),
            'lon': float(hq_lons[rank]),
            'x_3857': float(x_3857[rank]),
            'y_3857': float(y_3857[rank]),
            'incidents_covered': n_covered,
            'weighted_coverage': round(float(weighted_cov), 3),
            'backup_incidents': backup_count,
            'beats_assigned': 1,         # Baseline placeholder value for structural parity
            'coverage_pct': round(100.0 * n_covered / n_inc, 2),
            'weighted_coverage_pct': round(100.0 * float(weighted_cov) / total_weight, 2),
        })

    df_stations = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_stations.to_csv(output_path, index=False)
    print(f"   Stations CSV exported -> {output_path}  ({len(df_stations)} rows)")
    return df_stations


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


# ─────────────────────────────────────────────
# 3. MAIN EVALUATION PIPELINE
# ─────────────────────────────────────────────

def evaluate_existing_coverage():
    t0 = time.time()

    # ── 3.1 Spatial Boundary and Intersection Filtering ──────────────────────
    print("[1/5] Loading geometry boundaries and generating intersection...")
    if not os.path.exists(BOUNDARY_FILE_PATH) or not os.path.exists(CRIME_DATA_PATH) or not os.path.exists(STATIONS_FILE_PATH):
        print("ERROR: Missing data input files.")
        return

    la_geojson = gpd.read_file(BOUNDARY_FILE_PATH).to_crs(epsg=4326)
    user_poly  = Polygon(USER_POLYGON_COORDS)
    user_gdf   = gpd.GeoDataFrame({'geometry': [user_poly]}, crs='EPSG:4326')

    intersection_poly = la_geojson.unary_union.intersection(user_gdf.unary_union)

    if intersection_poly.is_empty:
        print("ERROR: Bounding intersection zone is empty.")
        return

    city_boundary = gpd.GeoDataFrame({'geometry': [intersection_poly]}, crs='EPSG:4326').to_crs(epsg=3857)
    print("   Successfully locked bounding zone intersection.")

    # ── 3.2 Load and Process Existing Police Stations ───────────────────────
    df_stations = pd.read_csv(STATIONS_FILE_PATH)
    geometry_stations = [Point(xy) for xy in zip(df_stations['x'], df_stations['y'])]
    
    gdf_stations = gpd.GeoDataFrame(df_stations, geometry=geometry_stations, crs="EPSG:2229").to_crs(epsg=3857)
    gdf_stations = gpd.sjoin(gdf_stations, city_boundary, how='inner', predicate='within').copy()
    print(f"   {len(gdf_stations)} existing LAPD stations matched inside region footprint.")

    # ── 3.3 Load and Filter Crime Data ──────────────────────────────────────
    df_crime = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT', 'LON', 'crime_weight'])
    df_crime = df_crime[(df_crime['LAT'] != 0) & (df_crime['LON'] != 0)]
    
    gdf_crime = gpd.GeoDataFrame(
        df_crime, geometry=[Point(xy) for xy in zip(df_crime['LON'], df_crime['LAT'])], crs="EPSG:4326"
    ).to_crs(epsg=3857)
    
    gdf_crime = gpd.sjoin(gdf_crime, city_boundary, how='inner', predicate='within').copy()
    weights = gdf_crime['crime_weight'].values
    n_incidents = len(gdf_crime)
    print(f"   {n_incidents:,} incidents matched inside region.")

    if n_incidents == 0:
        print("ERROR: No crime incidents inside the boundary intersection.")
        return

    # ── 3.4 Load OSM Graph ───────────────────────────────────────────────────
    print("[2/5] Acquiring OSM road network ...")
    G = load_or_download_graph(city_boundary)
    G_4326  = ox.project_graph(G, to_crs='EPSG:4326')
    G_metric = ox.project_graph(G)

    # ── 3.5 Snapping Coordinates to Road Network ──────────────────────────────
    print("[3/5] Snapping stations and incidents to nodes ...")
    
    stations_4326 = gdf_stations.to_crs(epsg=4326)
    hq_nodes = snap_to_nodes(G_4326, stations_4326.geometry.x.values, stations_4326.geometry.y.values)
    
    crime_4326 = gdf_crime.to_crs(epsg=4326)
    inc_nodes = snap_to_nodes(G_4326, crime_4326.geometry.x.values, crime_4326.geometry.y.values)

    # ── 3.6 Coverage Matrix Calculations ─────────────────────────────────────
    print("[4/5] Computing identical coverage metrics ...")
    existing_coverage_sets = build_existing_coverage_sets(G_metric, hq_nodes, inc_nodes, SERVICE_M)
    
    coverage_counts, covered_count, max_cov, max_backup = evaluate_coverage(
        existing_coverage_sets, weights, n_incidents
    )

    total_weighted = weights.sum()
    pct_count      = 100 * covered_count   / n_incidents
    pct_weighted   = 100 * max_cov          / total_weighted

    print(f"\n  ── Existing Infrastructure Coverage Evaluation (S = {SERVICE_MI} mi) ──")
    print(f"   Active Stations Filtered     : {len(gdf_stations)}")
    print(f"   Incident coverage (count)    : {covered_count:,} / {n_incidents:,}  ({pct_count:.1f} %)")
    print(f"   Maximal Covering Obj (O)     : {max_cov:,.1f} / {total_weighted:,.1f}  ({pct_weighted:.1f} %)")
    print(f"   Maximal Backup Obj (B)       : {max_backup:,.1f}\n")

    # ── 3.7 Exact Schema Data File Exports ───────────────────────────────────
    print("[UI Export] Writing baseline data files matching optimization schemas...")
    
    # 1. Export stations matching stations.csv structural logic
    export_stations_csv(
        gdf_stations=gdf_stations,
        coverage_sets=existing_coverage_sets,
        weights=weights,
        output_path=OUTPUT_STATIONS
    )

    # 2. Export summary matching optimization_summary.csv structural logic
    export_optimization_summary(
        n_inc=n_incidents,
        covered_count=covered_count,
        pct_count=pct_count,
        pct_weight=pct_weighted,
        Z_ip=max_cov,  # Primary objective aligns to maximum covered weight
        O=max_cov,
        B=max_backup,
        total_weight=total_weighted,
        gap=0.0,
        status="N/A (Existing Baseline Setup)",
        SERVICE_MI=SERVICE_MI,
        SERVICE_M=SERVICE_M,
        NUM_BEATS=len(gdf_stations),   # Candidate facilities equal active nodes
        NUM_SECTORS=len(gdf_stations), # P assets equal active nodes
        runtime_s=time.time() - t0,
        output_path=OUTPUT_OPT_SUMMARY
    )

    # ── 3.8 Map Visualization ────────────────────────────────────────────────
    print("[5/5] Rendering matching layout map ...")
    fig, ax = plt.subplots(figsize=(16, 13))
    
    sectors_clipped = gpd.overlay(gpd.read_file(BOUNDARY_FILE_PATH).to_crs(epsg=3857), city_boundary, how='intersection')
    sectors_clipped.plot(ax=ax, column=sectors_clipped.index, cmap='tab20', alpha=0.45, edgecolor='royalblue', linewidth=1.5)
    
    gdf_stations.plot(ax=ax, color='red', marker='H', markersize=220, edgecolor='white', zorder=10)

    cx.add_basemap(ax, crs=3857, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.3, zoom=11)

    if len(gdf_stations) > 0:
        example_hq = gdf_stations.geometry.iloc[0]
        ax.add_patch(patches.Circle(
            (example_hq.x, example_hq.y), SERVICE_M,
            linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.10, zorder=9
        ))
        ax.annotate(
            f"{SERVICE_MI} mi Road-Network Service Radius ($S$)",
            xy=(example_hq.x, example_hq.y + SERVICE_M + 600),
            ha='center', weight='bold', color='blue', fontsize=9
        )

    stats_text = (
        f"Existing Infrastructure Baseline Performance\n"
        f"S = {SERVICE_MI} mi  | Filtered Stations: {len(gdf_stations)}\n"
        f"Coverage (Unweighted): {pct_count:.1f}%\n"
        f"Maximal Covering Obj (O): {max_cov:,.0f}\n"
        f"Maximal Backup Obj   (B): {max_backup:,.0f}"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.title("Existing LAPD Infrastructure Profile — Clipped Boundaries\n(Optimizer Control Study Formulation Verification)", fontsize=14)
    ax.set_axis_off()
    
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Map saved -> {OUTPUT_IMG}")
    print(f"Total Runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    evaluate_existing_coverage()