
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def generate_citywide_patrol_model(file_path):
    try:
        # 1. Load your trimmed dataset
        df = pd.read_csv(file_path)
        
        # Data Cleaning: Essential for mapping
        # Remove incidents with no coordinates or zeroed-out values
        df = df.dropna(subset=['LAT', 'LON', 'crime_weight'])
        df = df[(df['LAT'] != 0) & (df['LON'] != 0)]
        
        # Optional: If the dataset is massive (e.g., >100k rows), 
        # we take a large sample to keep the visualization responsive.
        if len(df) > 50000:
            df = df.sample(n=50000, random_state=42)

        print(f"Analyzing {len(df)} total incidents across the entire map...")

        # 2. Stage 1: Generate Optimized Beats (Small Units)
        # We generate a high number of beats to represent the 'reporting districts' 
        # described in the PDF[cite: 124, 187].
        num_beats = 300 
        kmeans_beats = KMeans(n_clusters=num_beats, n_init=10, random_state=42)
        
        # We apply your 'crime_weight' to pull beat centers toward high-crime zones
        df['beat_label'] = kmeans_beats.fit_predict(
            df[['LAT', 'LON']], 
            sample_weight=df['crime_weight']
        )
        
        # These centroids represent the 'Potential Facility Sites' (j) [cite: 124, 204]
        beat_centers = kmeans_beats.cluster_centers_

        # 3. Stage 2: Group Beats into Sectors (User Input)
        # This decides how many 'Command Centers' (P) the city has[cite: 131, 137].
        try:
            p_sectors = int(input("Enter the total number of SECTORS (P) for the whole map: "))
        except ValueError:
            p_sectors = 15
            print(f"Invalid input. Defaulting to {p_sectors} sectors.")

        # Cluster the beat centers to find the optimal Sector Headquarters
        kmeans_sectors = KMeans(n_clusters=p_sectors, n_init=10, random_state=42)
        sector_labels = kmeans_sectors.fit_predict(beat_centers)
        sector_hqs = kmeans_sectors.cluster_centers_

        # 4. Visualization
        plt.figure(figsize=(15, 10))
        
        # Map sector IDs back to the original incidents for coloring the map
        beat_to_sector_map = {i: sector_labels[i] for i in range(num_beats)}
        df['sector_id'] = df['beat_label'].map(beat_to_sector_map)

        # Plot A: All incidents across the city (low alpha to see density)
        # The colors represent the different Sectors
        plt.scatter(df['LON'], df['LAT'], c=df['sector_id'], 
                    cmap='tab20', alpha=0.05, s=1, label='City-Wide Incidents')

        # Plot B: Individual Beat Centers
        plt.scatter(beat_centers[:, 1], beat_centers[:, 0], c='black', 
                    marker='.', s=5, alpha=0.5, label='Optimized Beats')

        # Plot C: Sector Headquarters (The 'Stations' or Sector HQ)
        # These are the P optimal locations from the PPAC model[cite: 153, 209].
        plt.scatter(sector_hqs[:, 1], sector_hqs[:, 0], c='red', 
                    marker='H', s=200, edgecolors='white', linewidth=2, 
                    label='Optimal Sector HQ (Stations)')

        plt.title(f"City-Wide Police Patrol Optimization\n"
                  f"Hierarchy: {num_beats} Beats grouped into {p_sectors} Sectors", fontsize=16)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        # Clean up legend and grid
        lgnd = plt.legend(loc='upper right', scatterpoints=1, fontsize=10)
        for handle in lgnd.legend_handles:
            handle.set_alpha(1.0)
            handle.set_sizes([50])
            
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

# Usage: Run with your local path
generate_citywide_patrol_model('../resources/cleaned_data.csv')