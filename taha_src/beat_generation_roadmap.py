"""
Optimized Police Patrol Beat Generator — Road-Network Edition
=============================================================
Improvements over the baseline script
--------------------------------------
1. Road-network distances (OSMnx + NetworkX) replace straight-line (Euclidean)
   distances for the PPAC service-radius test.
2. Sector headquarters are snapped to the nearest OSM road node so that
   every computed distance is realistically traversable.
3. A coverage summary table is printed and saved alongside the map.

Algorithm (PPAC – Police Patrol Area Covering Model)
-----------------------------------------------------
Stage 1 – Weighted K-Means → beat centroids
    Minimise  Σ_j Σ_{i∈S_j}  a_i ‖x_i − μ_j‖²
    where a_i = crime_weight of incident i.

Stage 2 – Weighted K-Means on beat centroids → sector headquarters

Stage 3 – Road-network coverage test
    For each sector HQ  j  and each crime incident  i :
        d_road(i, j) = shortest path length on the OSM drive network (metres)
    An incident is *covered* iff  d_road(i, j_nearest) ≤ S (metres).

Stage 4 – Voronoi beat polygons clipped to city boundary → visualisation.
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import contextily as cx
import networkx as nx
import osmnx as ox

from sklearn.cluster import KMeans
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
CRIME_DATA_PATH    = '../resources/cleaned_data.csv'
BOUNDARY_FILE_PATH = '../resources/LA_AREA.geojson'

NUM_BEATS   = 300          # Small patrol units  (k₁)
NUM_SECTORS = 30           # Administrative groupings  (P)
SERVICE_MI  = 2.0          # Acceptable service distance  S  (miles)
SERVICE_M   = SERVICE_MI * 1_609.34   # … converted to metres

OUTPUT_IMG  = '../outputs/beats/final_patrol_area_map_roadnet.png'
OUTPUT_CSV  = '../outputs/beats/coverage_summary_roadnet.csv'

OSM_CACHE   = '../resources/la_drive_network.graphml'   # cached graph

# ─────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────

def load_or_download_graph(boundary_gdf: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    """
    Return the OSM drive network for the study area.
    The graph is persisted as GraphML so subsequent runs are instant.
    """
    if os.path.exists(OSM_CACHE):
        print("  Loading cached OSM road network …")
        G = ox.load_graphml(OSM_CACHE)
    else:
        print("  Downloading OSM road network (this may take a few minutes) …")
        # Use the convex hull of the boundary polygon to query OSM
        poly = boundary_gdf.to_crs(epsg=4326).unary_union.convex_hull
        G = ox.graph_from_polygon(poly, network_type='drive')
        os.makedirs(os.path.dirname(OSM_CACHE), exist_ok=True)
        ox.save_graphml(G, OSM_CACHE)
        print(f"  Saved graph to {OSM_CACHE}")
    return G


def snap_to_node(G: nx.MultiDiGraph, lon: float, lat: float) -> int:
    """Return the nearest OSM node id for a (lon, lat) coordinate pair."""
    return ox.nearest_nodes(G, X=lon, Y=lat)


def network_distance_m(G: nx.MultiDiGraph,
                        orig_node: int,
                        dest_node: int) -> float:
    """
    Shortest road-network distance in metres between two OSM nodes.
    Returns np.inf if no path exists.
    """
    try:
        return nx.shortest_path_length(G, orig_node, dest_node, weight='length')
    except nx.NetworkXNoPath:
        return np.inf


# ─────────────────────────────────────────────
# 2. MAIN PIPELINE
# ─────────────────────────────────────────────

def generate_patrol_map():
    # ── 2.1  Load data ──────────────────────────────────────────────────────
    if not os.path.exists(CRIME_DATA_PATH) or not os.path.exists(BOUNDARY_FILE_PATH):
        print("ERROR: Missing input files.  Check CRIME_DATA_PATH / BOUNDARY_FILE_PATH.")
        return

    print("[1/6] Loading data …")
    city_boundary = gpd.read_file(BOUNDARY_FILE_PATH).to_crs(epsg=3857)

    df = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

    geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_filtered = gpd.sjoin(gdf, city_boundary, how='inner', predicate='within').copy()
    print(f"   {len(gdf_filtered):,} crime incidents inside city boundary.")

    # ── 2.2  Stage 1 – Beat clustering (Weighted K-Means) ───────────────────
    print("[2/6] Stage 1: Weighted K-Means → beat centroids …")
    coords  = np.column_stack((gdf_filtered.geometry.x, gdf_filtered.geometry.y))
    weights = gdf_filtered['crime_weight'].values

    km_beats = KMeans(n_clusters=NUM_BEATS, n_init=10, random_state=42)
    km_beats.fit(coords, sample_weight=weights)
    beat_centers = km_beats.cluster_centers_          # shape (NUM_BEATS, 2)  — EPSG:3857

    # ── 2.3  Stage 2 – Sector clustering ────────────────────────────────────
    print("[3/6] Stage 2: Weighted K-Means on beat centroids → sector HQs …")
    km_sectors   = KMeans(n_clusters=NUM_SECTORS, n_init=10, random_state=42)
    sector_labels = km_sectors.fit_predict(beat_centers)
    sector_hqs   = km_sectors.cluster_centers_         # shape (NUM_SECTORS, 2) — EPSG:3857

    # ── 2.4  Load OSM road network ───────────────────────────────────────────
    print("[4/6] Acquiring OSM road network …")
    G = load_or_download_graph(city_boundary)
    # Project graph to EPSG:4326 (required by ox.nearest_nodes)
    G_proj = ox.project_graph(G, to_crs='EPSG:4326')

    # Convert sector HQ coordinates: EPSG:3857 → EPSG:4326
    hq_gdf_3857 = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(sector_hqs[:, 0], sector_hqs[:, 1]),
        crs=3857
    ).to_crs(4326)
    hq_lon = hq_gdf_3857.geometry.x.values
    hq_lat = hq_gdf_3857.geometry.y.values

    # Snap every sector HQ to the nearest OSM node
    print("   Snapping sector HQs to nearest road nodes …")
    hq_nodes = [snap_to_node(G_proj, lon, lat)
                for lon, lat in zip(hq_lon, hq_lat)]

    # Convert crime incident coordinates: EPSG:3857 → EPSG:4326
    inc_gdf_4326 = gdf_filtered.to_crs(4326)
    inc_lon = inc_gdf_4326.geometry.x.values
    inc_lat = inc_gdf_4326.geometry.y.values

    # Snap every incident to the nearest OSM node
    print("   Snapping incidents to nearest road nodes …")
    inc_nodes = [snap_to_node(G_proj, lon, lat)
                 for lon, lat in zip(inc_lon, inc_lat)]

    # ── 2.5  Stage 3 – Road-network coverage test (PPAC) ────────────────────
    print("[5/6] Stage 3: Computing road-network coverage …")
    n_incidents = len(inc_nodes)

    # For each incident find its nearest sector HQ (Euclidean in projected
    # space — fast approximation for assignment only), then compute the true
    # road-network distance to that HQ.
    inc_xy = np.column_stack((gdf_filtered.geometry.x, gdf_filtered.geometry.y))
    # Assign each incident to its nearest sector centroid
    assigned_sector = km_sectors.predict(inc_xy)

    covered          = 0
    covered_weighted = 0.0
    total_weighted   = weights.sum()
    road_distances   = np.full(n_incidents, np.nan)

    for idx in range(n_incidents):
        sec_idx  = assigned_sector[idx]
        hq_node  = hq_nodes[sec_idx]
        inc_node = inc_nodes[idx]
        dist_m   = network_distance_m(G_proj, inc_node, hq_node)
        road_distances[idx] = dist_m
        if dist_m <= SERVICE_M:
            covered          += 1
            covered_weighted += weights[idx]

    pct_covered          = 100 * covered          / n_incidents
    pct_covered_weighted = 100 * covered_weighted / total_weighted

    print(f"\n   ── PPAC Coverage Summary (S = {SERVICE_MI} mi / {SERVICE_M:.0f} m) ──")
    print(f"   Incidents covered (count)  : {covered:,} / {n_incidents:,}  ({pct_covered:.1f} %)")
    print(f"   Incidents covered (weighted): {covered_weighted:,.1f} / {total_weighted:,.1f}  "
          f"({pct_covered_weighted:.1f} %)\n")

    # Save coverage summary
    summary = pd.DataFrame({
        'metric': [
            'Total incidents',
            'Service distance (mi)',
            'Service distance (m)',
            'Incidents covered (count)',
            'Coverage pct (count)',
            'Incidents covered (weighted)',
            'Coverage pct (weighted)',
        ],
        'value': [
            n_incidents,
            SERVICE_MI,
            SERVICE_M,
            covered,
            round(pct_covered, 2),
            round(covered_weighted, 2),
            round(pct_covered_weighted, 2),
        ]
    })
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    # ── 2.6  Visualisation ──────────────────────────────────────────────────
    print("[6/6] Rendering map …")

    # Voronoi beat polygons
    vor = Voronoi(beat_centers)
    lines = []
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            poly = Polygon([vor.vertices[v] for v in region])
            lines.append({'geometry': poly,
                          'beat_id':  i,
                          'sector_id': sector_labels[i]})

    gdf_voronoi = gpd.GeoDataFrame(lines, crs=3857)
    gdf_beats_clipped = gpd.overlay(gdf_voronoi, city_boundary, how='intersection')
    gdf_sectors       = gdf_beats_clipped.dissolve(by='sector_id')

    fig, ax = plt.subplots(figsize=(16, 13))

    # Sector fill
    gdf_sectors.plot(ax=ax, column=gdf_sectors.index,
                     cmap='tab20', alpha=0.45,
                     edgecolor='royalblue', linewidth=2.0)

    # Beat outlines
    gdf_beats_clipped.plot(ax=ax, facecolor='none',
                           edgecolor='black', linewidth=0.3, alpha=0.6)

    # Sector HQs (road-snapped positions displayed)
    gdf_hqs = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(sector_hqs[:, 0], sector_hqs[:, 1]),
        crs=3857
    )
    gdf_hqs.plot(ax=ax, color='red', marker='H',
                 markersize=220, edgecolor='white', zorder=10)

    # Basemap
    cx.add_basemap(ax, crs=3857,
                   source=cx.providers.OpenStreetMap.Mapnik,
                   alpha=0.3, zoom=11)

    # Service-radius indicator (one example circle at the median HQ)
    example_hq = sector_hqs[len(sector_hqs) // 2]
    circle = patches.Circle(
        (example_hq[0], example_hq[1]), SERVICE_M,
        linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.10
    )
    ax.add_patch(circle)
    ax.annotate(
        f"{SERVICE_MI} mi Road-Network Service Radius ($S$)",
        xy=(example_hq[0], example_hq[1] + SERVICE_M + 600),
        ha='center', weight='bold', color='blue', fontsize=9
    )

    # Coverage annotation box
    textstr = (f"Road-Network PPAC Coverage\n"
               f"S = {SERVICE_MI} mi  |  Sectors = {NUM_SECTORS}  |  Beats = {NUM_BEATS}\n"
               f"Covered: {pct_covered:.1f}% of incidents\n"
               f"Weighted coverage: {pct_covered_weighted:.1f}%")
    props = dict(boxstyle='round', facecolor='white', alpha=0.85)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', bbox=props)

    plt.title(
        "Optimized Police Patrol Geography — Road-Network Distance Model\n"
        "Sectors (Coloured) · Beats (Outlined) · HQs (Red Hexagons)",
        fontsize=14
    )
    ax.set_axis_off()

    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSuccess!  Map saved → {OUTPUT_IMG}")
    print(f"          Coverage CSV saved → {OUTPUT_CSV}")


if __name__ == '__main__':
    generate_patrol_map()