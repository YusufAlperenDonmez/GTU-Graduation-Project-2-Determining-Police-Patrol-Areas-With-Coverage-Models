"""
Existing Police Patrol Coverage Evaluator — Road-Network Edition
=====================================================================
This script evaluates the actual coverage of existing LAPD police stations
against historical crime density using a road-network distance model.

Key features:
1. Automatically handles California State Plane Zone V (EPSG:2229) coordinates.
2. Uses vectorized OSMnx node snapping for fast processing.
3. Computes ego-graph Dijkstra paths to measure true road distance coverage.
4. Generates a map overlaying exact sector boundaries and existing HQs.
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
from shapely.geometry import Point

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
CRIME_DATA_PATH    = '../../resources/cleaned_data.csv'
SECTORS_FILE_PATH  = '../../resources/LA_AREA.geojson' 
STATIONS_FILE_PATH = '../../resources/LAPD_police_stations.csv' # The CSV file with x,y columns

SERVICE_MI   = 2.5
SERVICE_M    = SERVICE_MI * 1_609.34

OUTPUT_IMG   = '../../outputs/beats/existing_coverage_analysis.png'
OUTPUT_CSV   = '../../outputs/beats/existing_coverage_summary.csv'
OSM_CACHE    = '../../resources/la_drive_network.graphml'

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
    return G

def snap_all_to_nodes(G: nx.MultiDiGraph, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Vectorized snap: maps geographic coordinates to the nearest OSM road nodes."""
    return np.array(ox.nearest_nodes(G, X=lons, Y=lats))

def coverage_via_ego_graphs(G: nx.MultiDiGraph, hq_nodes, inc_nodes, weights, radius_m):
    """Calculates maximal and backup coverage using actual road distances."""
    n_inc = len(inc_nodes)
    coverage_counts = np.zeros(n_inc, dtype=int)

    for hq_node in hq_nodes:
        try:
            # Single-source Dijkstra from the HQ node, cut off at radius_m
            lengths = nx.single_source_dijkstra_path_length(
                G, hq_node, cutoff=radius_m, weight='length'
            )
            reachable_nodes = np.array(list(lengths.keys()))
            
            # Check which incidents fall into these reachable nodes
            covered_by_hq = np.isin(inc_nodes, reachable_nodes)
            coverage_counts += covered_by_hq
        except Exception:
            continue

    covered_once_mask = coverage_counts >= 1
    covered_count = covered_once_mask.sum()
    maximal_covering_objective = weights[covered_once_mask].sum()
    maximal_backup_objective = (weights * coverage_counts).sum()

    return coverage_counts, covered_count, maximal_covering_objective, maximal_backup_objective


# ─────────────────────────────────────────────
# 2. MAIN EVALUATION PIPELINE
# ─────────────────────────────────────────────

def evaluate_existing_coverage():
    t0 = time.time()

    # ── 2.1 Load Spatial Boundary Data ───────────────────────────────────────
    print("[1/5] Loading Sectors and Stations …")
    if not os.path.exists(SECTORS_FILE_PATH) or not os.path.exists(STATIONS_FILE_PATH):
        print("ERROR: Missing geometry files.")
        return

    # Load sectors
    gdf_sectors = gpd.read_file(SECTORS_FILE_PATH).to_crs(epsg=3857)
    
    # Load and process police stations from CSV (EPSG:2229 -> EPSG:3857)
    df_stations = pd.read_csv(STATIONS_FILE_PATH)
    geometry_stations = [Point(xy) for xy in zip(df_stations['x'], df_stations['y'])]
    gdf_stations_raw = gpd.GeoDataFrame(df_stations, geometry=geometry_stations, crs="EPSG:2229")
    gdf_stations = gdf_stations_raw.to_crs(epsg=3857)

    # ── 2.2 Load Crime Data ──────────────────────────────────────────────────
    print("      Loading Crime Data …")
    df_crime = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT', 'LON', 'crime_weight'])
    df_crime = df_crime[(df_crime['LAT'] != 0) & (df_crime['LON'] != 0)]
    
    geometry_crime = [Point(xy) for xy in zip(df_crime['LON'], df_crime['LAT'])]
    gdf_crime_raw = gpd.GeoDataFrame(df_crime, geometry=geometry_crime, crs="EPSG:4326")
    gdf_crime = gdf_crime_raw.to_crs(epsg=3857)
    
    # Filter crime to stay strictly within the LAPD sectors area
    city_boundary = gdf_sectors.unary_union
    gdf_crime = gdf_crime[gdf_crime.geometry.within(city_boundary)]
    weights = gdf_crime['crime_weight'].values

    # ── 2.3 Load OSM Graph ───────────────────────────────────────────────────
    print("[2/5] Acquiring OSM road network …")
    G = load_or_download_graph(gdf_sectors)
    G_proj = ox.project_graph(G, to_crs='EPSG:4326')

    # ── 2.4 Vectorized Snapping ──────────────────────────────────────────────
    print("[3/5] Snapping stations and incidents to nodes …")
    
    # Snap Stations (must be in Lat/Lon 4326 for OSMnx)
    stations_4326 = gdf_stations.to_crs(epsg=4326)
    hq_nodes = snap_all_to_nodes(G_proj, stations_4326.geometry.x.values, stations_4326.geometry.y.values)
    
    # Snap Incidents
    crime_4326 = gdf_crime.to_crs(epsg=4326)
    inc_nodes = snap_all_to_nodes(G_proj, crime_4326.geometry.x.values, crime_4326.geometry.y.values)

    # ── 2.5 Coverage Analysis ────────────────────────────────────────────────
    print("[4/5] Computing coverage metrics …")
    coverage_counts, covered_count, max_cov, max_backup = coverage_via_ego_graphs(
        G_proj, hq_nodes, inc_nodes, weights, SERVICE_M
    )

    n_incidents      = len(inc_nodes)
    total_weighted   = weights.sum()
    pct_count        = 100 * covered_count    / n_incidents
    pct_weighted     = 100 * max_cov          / total_weighted

    print(f"\n   ── Existing Infrastructure Coverage (S = {SERVICE_MI} mi) ──")
    print(f"   Total Stations                : {len(gdf_stations)}")
    print(f"   Covered Incidents (count)     : {covered_count:,} / {n_incidents:,}  ({pct_count:.1f} %)")
    print(f"   Maximal Covering Objective (O): {max_cov:,.1f} / {total_weighted:,.1f}  ({pct_weighted:.1f} %)")
    print(f"   Maximal Backup Objective      : {max_backup:,.1f}\n")

    # Save metrics to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame({
        'metric': [
            'Total Stations', 'Total incidents', 'Service distance (mi)',
            'Incidents covered (count)', 'Coverage pct (count)',
            'Maximal Covering Objective', 'Coverage pct (weighted)',
            'Maximal Backup Objective'
        ],
        'value': [
            len(gdf_stations), n_incidents, SERVICE_MI,
            covered_count, round(pct_count, 2),
            round(max_cov, 2), round(pct_weighted, 2),
            round(max_backup, 2)
        ]
    }).to_csv(OUTPUT_CSV, index=False)


    # ── 2.6 Visualization ────────────────────────────────────────────────────
    print("[5/5] Rendering map …")
    fig, ax = plt.subplots(figsize=(16, 13))
    
    # Plot Sector Lines from GeoJSON (colored by index to distinguish them visually)
    gdf_sectors.plot(ax=ax, column=gdf_sectors.index, cmap='tab20', alpha=0.45, edgecolor='royalblue', linewidth=1.5)
    
    # Plot Police Stations
    gdf_stations.plot(ax=ax, color='red', marker='H', markersize=220, edgecolor='white', zorder=10, label='Police Stations')

    # Add Basemap
    cx.add_basemap(ax, crs=3857, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.3, zoom=11)

    # Add Reference Circle for Service Radius (centered on Central Division for scale)
    example_hq = gdf_stations.geometry.iloc[6] # Index 6 is Central Division in your CSV
    ax.add_patch(patches.Circle(
        (example_hq.x, example_hq.y), SERVICE_M,
        linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.15, zorder=9
    ))
    ax.annotate(
        f"{SERVICE_MI} mi Service Radius ($S$)",
        xy=(example_hq.x, example_hq.y + SERVICE_M + 600),
        ha='center', weight='bold', color='blue', fontsize=9
    )

    # Add Metrics Text Box
    stats_text = (
        f"Existing Infrastructure Performance\n"
        f"Service Radius: {SERVICE_MI} mi\n"
        f"Active Stations: {len(gdf_stations)}\n"
        f"Coverage (Weighted): {pct_weighted:.1f}%\n"
        f"Maximal Backup Obj: {max_backup:,.1f}"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title("Existing LAPD Police Stations & Sector Boundaries\n(Road-Network Coverage Evaluation)", fontsize=14)
    ax.set_axis_off()
    
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis complete. Map saved to {OUTPUT_IMG}")
    print(f"Total Runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    evaluate_existing_coverage()