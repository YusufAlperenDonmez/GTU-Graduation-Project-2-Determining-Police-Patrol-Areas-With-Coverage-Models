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

NUM_BEATS_GOAL = 300   # "Reporting Districts" [cite: 40]
NUM_SECTORS_GOAL = 15  # "Stations" or Sector HQs (P) [cite: 131, 153]
SERVICE_DISTANCE_MILES = 2 # surrogate for response time (S) [cite: 124, 186]

def generate_realistic_patrol_map():
    if not os.path.exists(CRIME_DATA_PATH) or not os.path.exists(BOUNDARY_FILE_PATH):
        print("Error: Files not found. Check your file paths.")
        return

    print("Step 1: Loading Data...")
    city_boundary = gpd.read_file(BOUNDARY_FILE_PATH)
    df = pd.read_csv(CRIME_DATA_PATH)
    df = df.dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]
    
    # Geocoding
    geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Step 2: Project to Web Mercator (EPSG:3857) for accurate mapping
    target_crs = 3857 
    city_boundary = city_boundary.to_crs(epsg=target_crs)
    gdf = gdf.to_crs(epsg=target_crs)

    print("Step 2: Filtering incidents inside LA boundaries...")
    gdf_filtered = gpd.sjoin(gdf, city_boundary, how='inner', predicate='within').copy()

    # --- STEP 3: STAGE 1 - GENERATING BEATS ---
    print("Step 3: Stage 1 - Generating Weighted Beats...")
    coords = np.column_stack((gdf_filtered.geometry.x, gdf_filtered.geometry.y))
    weights = gdf_filtered['crime_weight'].values

    # Objective: Maximize coverage Z = sum(ai * yi) [cite: 105, 134]
    kmeans_beats = KMeans(n_clusters=NUM_BEATS_GOAL, n_init=10, random_state=42)
    beat_assignments = kmeans_beats.fit_predict(coords, sample_weight=weights)
    
    # Store the beat centers as potential facility sites J [cite: 124, 204]
    beat_centers = kmeans_beats.cluster_centers_
    gdf_beats = gpd.GeoDataFrame(geometry=gpd.points_from_xy(beat_centers[:,0], beat_centers[:,1]), crs=target_crs)

    # --- STEP 4: STAGE 2 - GROUPING BEATS INTO SECTORS ---
    print("Step 4: Stage 2 - Grouping Beats into Sectors...")
    kmeans_sectors = KMeans(n_clusters=NUM_SECTORS_GOAL, n_init=10, random_state=42)
    
    # sector_labels tells us which sector each beat belongs to 
    sector_labels = kmeans_sectors.fit_predict(beat_centers)
    
    sector_hqs = kmeans_sectors.cluster_centers_
    gdf_sectors = gpd.GeoDataFrame(geometry=gpd.points_from_xy(sector_hqs[:,0], sector_hqs[:,1]), crs=target_crs)

    # --- STEP 5: VISUALIZATION ---
    print("Step 5: Drawing Map...")
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # FIX: Hierarchical Mapping
    # Every incident belongs to a beat, and every beat belongs to a sector 
    gdf_filtered['sector_id'] = sector_labels[beat_assignments]
    
    # Plotting
    gdf_filtered.plot(ax=ax, column='sector_id', cmap='tab20', markersize=1, alpha=0.1, zorder=1)
    city_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, alpha=0.5)
    gdf_sectors.plot(ax=ax, color='red', marker='H', markersize=200, edgecolor='white', zorder=10, label='Sector HQ')

    # Basemap with fixed zoom for LA
    cx.add_basemap(ax, crs=target_crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5, zoom=11)

    # --- 2-MILE DISTANCE INDICATOR ---
    # S = acceptable service distance [cite: 124, 186]
    RADIUS_METERS = SERVICE_DISTANCE_MILES * 1609.34 
    example_st = gdf_sectors.iloc[len(gdf_sectors)//2].geometry 
    circle = patches.Circle((example_st.x, example_st.y), RADIUS_METERS, 
                             linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.15, zorder=5)
    ax.add_patch(circle)
    ax.annotate(f"{SERVICE_DISTANCE_MILES} Mile Radius (S)", 
                xy=(example_st.x, example_st.y + RADIUS_METERS + 500),
                ha='center', weight='bold', color='blue')

    plt.title("Optimized Police Geography: Beats and Sectors (Los Angeles)", fontsize=16)
    ax.set_axis_off()
    plt.show()

generate_realistic_patrol_map()