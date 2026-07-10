import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd
from sklearn.cluster import KMeans
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
import contextily as cx
import os

# --- 1. CONFIGURATION ---
CRIME_DATA_PATH = '../resources/cleaned_data.csv' # Your trimmed dataset
BOUNDARY_FILE_PATH = '../resources/LA_AREA.geojson' #

NUM_BEATS = 300   # Total small patrol units
NUM_SECTORS = 30  # Total administrative groupings (P)
SERVICE_S = 2    # Service distance indicator in miles

def generate_final_patrol_map():
    if not os.path.exists(CRIME_DATA_PATH) or not os.path.exists(BOUNDARY_FILE_PATH):
        print("Error: Missing files. Ensure CSV and GeoJSON are in the directory.")
        return

    # Step 1: Load and Project
    city_boundary = gpd.read_file(BOUNDARY_FILE_PATH).to_crs(epsg=3857)
    df = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]
    
    geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_filtered = gpd.sjoin(gdf, city_boundary, how='inner', predicate='within').copy()

    # Step 2: Weighted Clustering (PPAC Logic)
    coords = np.column_stack((gdf_filtered.geometry.x, gdf_filtered.geometry.y))
    weights = gdf_filtered['crime_weight'].values
    
    kmeans_beats = KMeans(n_clusters=NUM_BEATS, n_init=10, random_state=42)
    kmeans_beats.fit(coords, sample_weight=weights)
    beat_centers = kmeans_beats.cluster_centers_

    kmeans_sectors = KMeans(n_clusters=NUM_SECTORS, n_init=10, random_state=42)
    sector_labels = kmeans_sectors.fit_predict(beat_centers)
    sector_hqs = kmeans_sectors.cluster_centers_

    # Step 3: Generate Beat Areas (Voronoi Polygons)
    # We create a large bounding box to ensure Voronoi covers the whole city
    vor = Voronoi(beat_centers)
    lines = []
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not -1 in region and len(region) > 0:
            poly = Polygon([vor.vertices[v] for v in region])
            lines.append({'geometry': poly, 'beat_id': i, 'sector_id': sector_labels[i]})
    
    gdf_voronoi = gpd.GeoDataFrame(lines, crs=3857)
    # CLIP TO CITY LIMITS: This creates the "Beats" limited to LA boundaries
    gdf_beats_clipped = gpd.overlay(gdf_voronoi, city_boundary, how='intersection')

    # Step 4: Visualization
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # 1. Draw Sector Areas (Unique colors)
    # We dissolve beat boundaries to get clean Sector Limits
    gdf_sectors = gdf_beats_clipped.dissolve(by='sector_id')
    gdf_sectors.plot(ax=ax, column=gdf_sectors.index, cmap='tab20', alpha=0.5, edgecolor='blue', linewidth=2)

    # 2. Draw Beat Limits (Thin gray lines)
    gdf_beats_clipped.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.6)

    # 3. Draw Sector HQ (Red Hexagons)
    gdf_hqs = gpd.GeoDataFrame(geometry=gpd.points_from_xy(sector_hqs[:,0], sector_hqs[:,1]), crs=3857)
    gdf_hqs.plot(ax=ax, color='red', marker='H', markersize=200, edgecolor='white', zorder=10)

    # 4. Basemap and Indicator
    cx.add_basemap(ax, crs=3857, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.3, zoom=11)
    
    # 2-mile Radius (S) indicator
    RADIUS_M = SERVICE_S * 1609.34
    example_loc = sector_hqs[len(sector_hqs)//2]
    circle = patches.Circle((example_loc[0], example_loc[1]), RADIUS_M, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.1)
    ax.add_patch(circle)
    ax.annotate(f"{SERVICE_S} Mile Service Radius ($S$)", xy=(example_loc[0], example_loc[1] + RADIUS_M + 500), ha='center', weight='bold', color='blue')

    plt.title("Optimized Police Patrol Geography\nSectors (Colored) and Beats (Outlined)", fontsize=16)
    ax.set_axis_off()
    
    # SAVE THE IMAGE
    plt.savefig('../outputs/beats/final_patrol_area_map.png', dpi=300, bbox_inches='tight')
    print("Success: Image saved as 'final_patrol_area_map.png'")

generate_final_patrol_map()