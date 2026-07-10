"""
Optimized Police Patrol Beat Generator — Integer Linear Programming (PuLP) Edition
==================================================================================
Implements the exact Maximal Backup Covering Location Problem with the 
multiobjective constraint (O) as formulated in Curtin et al., 2010.
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
import pulp

from sklearn.cluster import MiniBatchKMeans
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
CRIME_DATA_PATH    = '../resources/cleaned_data.csv'
BOUNDARY_FILE_PATH = '../resources/LA_AREA.geojson'

# ILP Constraints
NUM_BEATS   = 300  # These serve as the Candidate Sites (J)
NUM_SECTORS = 21   # Number of HQs to select (P)
SERVICE_MI  = 3.0
SERVICE_M   = SERVICE_MI * 1_609.34

# THE MULTIOBJECTIVE PARAMETER (O in the paper)
# This forces the solver to maintain at least this much 'single coverage' weight.
# Lower this value to allow extreme clustering for maximal backup coverage.
MIN_SINGLE_COVERAGE_OBJ = 1_800_000 

OUTPUT_IMG  = '../outputs/beats/backup_1_800k.png'
OUTPUT_CSV  = '../outputs/beats/pulp_coverage_summary2.csv'
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
    return G

def snap_all_to_nodes(G: nx.MultiDiGraph, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    return np.array(ox.nearest_nodes(G, X=lons, Y=lats))

# ─────────────────────────────────────────────
# 2. MAIN PIPELINE
# ─────────────────────────────────────────────

def generate_patrol_map():
    t0 = time.time()

    # ── 2.1 Load Data ────────────────────────────────────────────────────────
    print("[1/7] Loading data …")
    city_boundary = gpd.read_file(BOUNDARY_FILE_PATH).to_crs(epsg=3857)
    df = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

    geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_filtered = gpd.sjoin(gdf, city_boundary, how='inner', predicate='within').copy()
    
    # ── 2.2 Generate Candidate Sites (Beats) ─────────────────────────────────
    print("[2/7] Generating 300 Candidate Sites (Beat Centroids) …")
    coords  = np.column_stack((gdf_filtered.geometry.x, gdf_filtered.geometry.y))
    weights = gdf_filtered['crime_weight'].values

    km_beats = MiniBatchKMeans(n_clusters=NUM_BEATS, n_init=5, batch_size=10_000, random_state=42)
    km_beats.fit(coords, sample_weight=weights)
    candidate_hqs = km_beats.cluster_centers_

    # ── 2.3 Load OSM and Snap Points ─────────────────────────────────────────
    print("[3/7] Snapping incidents and candidate sites to OSM network …")
    G = load_or_download_graph(city_boundary)
    G_proj = ox.project_graph(G, to_crs='EPSG:4326')

    inc_gdf = gdf_filtered.to_crs(4326)
    inc_nodes = snap_all_to_nodes(G_proj, inc_gdf.geometry.x.values, inc_gdf.geometry.y.values)

    cand_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(candidate_hqs[:,0], candidate_hqs[:,1]), crs=3857).to_crs(4326)
    cand_nodes = snap_all_to_nodes(G_proj, cand_gdf.geometry.x.values, cand_gdf.geometry.y.values)

    # ── 2.4 Data Aggregation (Crucial for ILP speed) ─────────────────────────
    print("[4/7] Aggregating incidents by unique road nodes …")
    # Instead of 997k incident variables, group them by their snapped node
    node_weights = {}
    for node, weight in zip(inc_nodes, weights):
        node_weights[node] = node_weights.get(node, 0) + weight
    
    I = list(node_weights.keys())          # Set of unique incident nodes
    J = list(range(len(cand_nodes)))       # Set of candidate sites (0 to 299)
    print(f"  Reduced constraints from {len(inc_nodes):,} to {len(I):,} unique demand nodes.")

    # ── 2.5 Compute Reachability (Set N_i) ───────────────────────────────────
    print("[5/7] Computing ego-graph reachability for all candidate sites …")
    # covers[j] = set of incident nodes reachable from candidate j within SERVICE_M
    covers = {j: set() for j in J}
    for j, hq_node in enumerate(cand_nodes):
        try:
            lengths = nx.single_source_dijkstra_path_length(G_proj, hq_node, cutoff=SERVICE_M, weight='length')
            covers[j] = set(lengths.keys()).intersection(I)
        except Exception:
            continue

    # N_i[i] = list of candidate sites (j) that can reach incident node i
    N_i = {i: [] for i in I}
    for j in J:
        for i in covers[j]:
            N_i[i].append(j)

    # ── 2.6 PuLP ILP Optimization ────────────────────────────────────────────
    print(f"[6/7] Building and solving Integer Linear Program (PuLP) …")
    t_ilp = time.time()
    
    prob = pulp.LpProblem("Maximal_Backup_Coverage", pulp.LpMaximize)

    # Variables
    # x_j: 1 if candidate j is selected as an HQ, 0 otherwise
    x = pulp.LpVariable.dicts("x", J, cat='Binary')
    
    # y_i: Number of HQs covering incident node i (Integer up to NUM_SECTORS)
    y = pulp.LpVariable.dicts("y", I, lowBound=0, upBound=NUM_SECTORS, cat='Integer')
    
    # w_i: 1 if incident node i is covered AT LEAST ONCE (Single Coverage), 0 otherwise
    w = pulp.LpVariable.dicts("w", I, cat='Binary')

    # Objective Function: Maximize total backup coverage weight
    prob += pulp.lpSum([node_weights[i] * y[i] for i in I]), "Maximize_Backup_Coverage"

    # Constraint 1: Exactly P facilities must be located
    prob += pulp.lpSum([x[j] for j in J]) == NUM_SECTORS, "Total_Facilities"

    for i in I:
        # Constraint 2: The coverage count y_i cannot exceed the number of open facilities in N_i
        # (This enforces sum(x_j) >= y_i as formulated in the paper)
        prob += y[i] <= pulp.lpSum([x[j] for j in N_i[i]])
        
        # Constraint 3: Single coverage flag w_i is bounded by open facilities
        prob += w[i] <= pulp.lpSum([x[j] for j in N_i[i]])

    # Constraint 4: Multiobjective standard coverage boundary (Equation 7 in paper)
    prob += pulp.lpSum([node_weights[i] * w[i] for i in I]) >= MIN_SINGLE_COVERAGE_OBJ, "Min_Standard_Coverage"

    print("  Solving ILP (This may take a few minutes) ...")
    # Using default CBC solver. Time limit set to 5 minutes to prevent infinite hangs.
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=300))
    
    print(f"  ILP Status: {pulp.LpStatus[prob.status]} ({time.time() - t_ilp:.1f}s)")

    if prob.status != pulp.LpStatusOptimal and prob.status != pulp.LpStatusNotSolved:
        print("WARNING: Optimal solution not found. Check objective constraints.")

    # Extract chosen HQs
    selected_j = [j for j in J if pulp.value(x[j]) == 1.0]
    optimal_hqs = candidate_hqs[selected_j]
    
    # Calculate final stats from the solver's variables
    final_backup_obj = sum([node_weights[i] * pulp.value(y[i]) for i in I])
    final_single_obj = sum([node_weights[i] * pulp.value(w[i]) for i in I])
    total_inc_weight = sum(node_weights.values())

    print(f"\n   ── PuLP Optimization Results (O = {MIN_SINGLE_COVERAGE_OBJ:,}) ──")
    print(f"   Maximal Single Coverage : {final_single_obj:,.1f} / {total_inc_weight:,.1f}")
    print(f"   Maximal Backup Objective: {final_backup_obj:,.1f}")
    print(f"   HQs Selected            : {len(optimal_hqs)}")

    # ── 2.7 Visualisation ────────────────────────────────────────────────────
    print("[7/7] Rendering optimized map …")
    fig, ax = plt.subplots(figsize=(16, 13))

    # Base candidate sites (Beats) Voronoi for visual background
    vor = Voronoi(candidate_hqs)
    lines = []
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            lines.append({'geometry': Polygon([vor.vertices[v] for v in region]), 'beat_id': i})

    gdf_voronoi = gpd.GeoDataFrame(lines, crs=3857)
    gdf_beats = gpd.overlay(gdf_voronoi, city_boundary, how='intersection')
    
    gdf_beats.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)

    # Plot Selected HQs
    gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(optimal_hqs[:, 0], optimal_hqs[:, 1]), crs=3857
    ).plot(ax=ax, color='red', marker='H', markersize=350, edgecolor='white', zorder=10)

    # Plot Unselected Candidates (for reference)
    unselected_hqs = np.delete(candidate_hqs, selected_j, axis=0)
    gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(unselected_hqs[:, 0], unselected_hqs[:, 1]), crs=3857
    ).plot(ax=ax, color='gray', marker='o', markersize=20, alpha=0.5, zorder=5)

    cx.add_basemap(ax, crs=3857, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.3, zoom=11)

    # Draw service areas around selected HQs
    for hq in optimal_hqs:
        ax.add_patch(patches.Circle((hq[0], hq[1]), SERVICE_M, linewidth=1.5, edgecolor='blue', facecolor='blue', alpha=0.08))

    ax.text(0.02, 0.02,
            f"ILP Maximal Backup Coverage Model\n"
            f"Service Radius (S) = {SERVICE_MI} mi\n"
            f"Min Single Coverage (O) = {MIN_SINGLE_COVERAGE_OBJ:,}\n"
            f"Achieved Single Coverage = {final_single_obj:,.1f}\n"
            f"Maximized Backup Score = {final_backup_obj:,.1f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title("ILP Optimized Police Patrol Geography\nRed Hexagons = Selected HQs | Blue Circles = Service Range", fontsize=14)
    ax.set_axis_off()

    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Map saved → {OUTPUT_IMG}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    generate_patrol_map()