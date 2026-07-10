import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull

def generate_patrol_outlines(file_path):
    # 1. Load your trimmed dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("CSV file not found. Please check the path.")
        return

    # Clean data: Remove invalid coordinates
    df = df.dropna(subset=['LAT', 'LON', 'crime_weight'])
    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]
    
    # 2. Stage 1: Generate Beats (Smallest Units)
    # We create 100 beats to form a detailed 'mosaic'
    num_beats = 100
    kmeans_beats = KMeans(n_clusters=num_beats, n_init=10, random_state=42)
    # Weights influence the centers toward high-priority crimes
    kmeans_beats.fit(df[['LON', 'LAT']], sample_weight=df['crime_weight'])
    beat_centers = kmeans_beats.cluster_centers_

    # 3. Stage 2: Generate Sectors (Hierarchical Grouping)
    num_sectors = 8 # Total number of 'Stations' (P)
    kmeans_sectors = KMeans(n_clusters=num_sectors, n_init=10, random_state=42)
    sector_labels = kmeans_sectors.fit_predict(beat_centers)
    sector_hqs = kmeans_sectors.cluster_centers_

    # Create the Plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for i, ax in enumerate(axes):
        # A. Draw the City/Area Boundary (Convex Hull)
        hull = ConvexHull(df[['LON', 'LAT']])
        for simplex in hull.simplices:
            ax.plot(df[['LON', 'LAT']].iloc[simplex, 0], 
                    df[['LON', 'LAT']].iloc[simplex, 1], 'k-', lw=3, zorder=5)
        
        # B. Draw the Beat Outlines using Voronoi
        # This creates the 'grid' of small patrol areas
        vor = Voronoi(beat_centers)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', 
                        line_width=0.5, line_alpha=0.5, show_points=False)

        # C. Color the Beats by their Sector
        for j, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not -1 in region and len(region) > 0:
                polygon = [vor.vertices[i] for i in region]
                # Match the beat point index to its sector label
                color = plt.cm.tab10(sector_labels[j] % 10)
                ax.fill(*zip(*polygon), color=color, alpha=0.3)

        # D. Draw Sector HQ (Stations)
        ax.scatter(sector_hqs[:, 0], sector_hqs[:, 1], c='red', marker='H', 
                   s=250, edgecolors='white', linewidth=2, label='Sector HQ (Station)', zorder=10)

        if i == 0:
            ax.set_title("Explanatory Map: Beat & Sector Hierarchy", fontsize=15)
            # Add labels for explanation
            ax.annotate('City Boundary', xy=(df['LON'].mean(), df['LAT'].max()), 
                        xytext=(0, 20), textcoords='offset points', ha='center', weight='bold')
        else:
            ax.set_title("Clean Map: Final Boundaries", fontsize=15)

        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

# Run the code
generate_patrol_outlines('../resources/cleaned_data.csv')