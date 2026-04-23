import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd
from sklearn.cluster import KMeans
from shapely.geometry import Point
import contextily as cx
import os

# --- 1. CONFIGURATION ---
CRIME_DATA_PATH = '../resources/cleaned_data.csv' # Your trimmed dataset
BOUNDARY_FILE_PATH = '../resources/LA_AREA.geojson' # Download from LA Controller


# Police Patrol Parameters (Based on PDF concepts)
NUM_BEATS_GOAL = 300   # "Reporting Districts" / Smallest units 
NUM_SECTORS_GOAL = 15  # "Stations" or Sector HQs (P) [cite: 131]
SERVICE_DISTANCE_MILES = 2 # surrogate for response time [cite: 124, 186]

def generate_realistic_patrol_map():
    # Validation
    if not os.path.exists(CRIME_DATA_PATH) or not os.path.exists(BOUNDARY_FILE_PATH):
        print("Error: Files not found. Check your file paths.")
        return

    print("Step 1: Loading and Geocoding Data...")
    city_boundary = gpd.read_file(BOUNDARY_FILE_PATH)
    
    df = pd.read_csv(CRIME_DATA_PATH)
    df = df.dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]
    
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Step 2: Project and Filter (LA City Limits)
    # Using California State Plane Zone V (Feet) for accurate distance
    target_crs = 2229 
    city_boundary = city_boundary.to_crs(epsg=target_crs)
    gdf = gdf.to_crs(epsg=target_crs)

    print("Step 2: Filtering incidents inside LA boundaries...")
    gdf_filtered = gpd.sjoin(gdf, city_boundary, how='inner', predicate='within')

    # --- STEP 3: STAGE 1 - GENERATING BEATS (FIXED HERE) ---
    print("Step 3: Stage 1 - Generating Weighted Beats...")
    
    # FIX: Stack x and y into a (N, 2) matrix for KMeans
    coords = np.column_stack((gdf_filtered.geometry.x, gdf_filtered.geometry.y))
    weights = gdf_filtered['crime_weight'].values

    kmeans_beats = KMeans(n_clusters=NUM_BEATS_GOAL, n_init=10, random_state=42)
    # Fit using coordinates and weights (ai)
    kmeans_beats.fit(coords, sample_weight=weights)
    
    # These centroids represent potential command centers (J) [cite: 124]
    beat_centers = pd.DataFrame(kmeans_beats.cluster_centers_, columns=['x', 'y'])
    gdf_beats = gpd.GeoDataFrame(beat_centers, geometry=gpd.points_from_xy(beat_centers.x, beat_centers.y), crs=target_crs)

    # Step 4: Stage 2 - Grouping Beats into Sectors
    print("Step 4: Stage 2 - Grouping Beats into Sectors...")
    kmeans_sectors = KMeans(n_clusters=NUM_SECTORS_GOAL, n_init=10, random_state=42)
    sector_labels = kmeans_sectors.fit_predict(beat_centers[['x', 'y']])
    
    sector_centers = pd.DataFrame(kmeans_sectors.cluster_centers_, columns=['x', 'y'])
    gdf_sectors = gpd.GeoDataFrame(sector_centers, geometry=gpd.points_from_xy(sector_centers.x, sector_centers.y), crs=target_crs)

    # Step 5: Visualization
    print("Step 5: Drawing Map with Basemap and 2-Mile Indicator...")
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Basemap (OpenStreetMap)
    cx.add_basemap(ax, crs=target_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.4)

    # Plot Sector HQ (Stations)
    gdf_sectors.plot(ax=ax, color='red', marker='H', s=200, edgecolor='white', zorder=10, label='Sector HQ')

    # Color the background by sector grouping
    gdf_filtered['sector_id'] = kmeans_sectors.predict(kmeans_beats.predict(coords).reshape(-1, 1))
    gdf_filtered.plot(ax=ax, column='sector_id', cmap='tab20', markersize=1, alpha=0.1, zorder=1)

    # DRAW 2-MILE INDICATOR
    # One mile = 5280 feet
    RADIUS_FEET = SERVICE_DISTANCE_MILES * 5280
    example_st = gdf_sectors.iloc[0].geometry # Example station for scale
    circle = patches.Circle((example_st.x, example_st.y), RADIUS_FEET, 
                             linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.1, zorder=5)
    ax.add_patch(circle)
    ax.annotate(f"{SERVICE_DISTANCE_MILES} Mile Radius (S)", xy=(example_st.x, example_st.y + RADIUS_FEET + 1000),
                ha='center', weight='bold', color='blue')

    plt.title("Optimized Police Geography: Beats and Sectors (LA)", fontsize=16)
    ax.set_axis_off()
    plt.show()

generate_realistic_patrol_map()