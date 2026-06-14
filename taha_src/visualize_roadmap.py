"""
Road-Network Path Visualizer
=============================
Generates a publication-quality image showing shortest road paths
from 3 sector HQs to a handful of nearby incidents each.

Works even without your full dataset: if the CSV/GeoJSON are missing
it falls back to a built-in LA demo with synthetic incidents so you
can always see the road-drawing logic.
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
import osmnx as ox
from sklearn.cluster import MiniBatchKMeans
from shapely.geometry import Point

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CRIME_DATA_PATH    = '../resources/cleaned_data.csv'
BOUNDARY_FILE_PATH = '../resources/LA_AREA.geojson'
OSM_CACHE          = '../resources/la_drive_network.graphml'
OUTPUT_PATH        = '../outputs/beats/roadnet_paths_sample.png'

NUM_SECTORS        = 30        # must match your main script
SAMPLE_HQS         = 10         # how many sector HQs to show paths from
INCIDENTS_PER_HQ   = 50         # how many incidents per HQ to trace
MAX_PATH_DIST_M    = 2 * 1609  # only draw paths within 2-mile service radius

# Colour palette – one per HQ
HQ_COLORS = ['#E63946', '#2196F3', '#4CAF50']
# ─────────────────────────────────────────────────────────────────────────────


def load_or_download_graph(boundary_gdf):
    if os.path.exists(OSM_CACHE):
        print("  Loading cached OSM graph …")
        return ox.load_graphml(OSM_CACHE)
    print("  Downloading OSM graph (first run only) …")
    poly = boundary_gdf.to_crs(4326).unary_union.convex_hull
    G = ox.graph_from_polygon(poly, network_type='drive')
    os.makedirs(os.path.dirname(OSM_CACHE), exist_ok=True)
    ox.save_graphml(G, OSM_CACHE)
    return G


def load_data():
    """Load real data if available, else return a small synthetic LA dataset."""
    if os.path.exists(CRIME_DATA_PATH) and os.path.exists(BOUNDARY_FILE_PATH):
        print("  Using real dataset …")
        boundary = gpd.read_file(BOUNDARY_FILE_PATH).to_crs(3857)
        df = pd.read_csv(CRIME_DATA_PATH).dropna(subset=['LAT','LON','crime_weight'])
        df = df[(df['LAT'] != 0) & (df['LON'] != 0)]
        gdf = gpd.GeoDataFrame(df,
                               geometry=[Point(xy) for xy in zip(df['LON'], df['LAT'])],
                               crs=4326).to_crs(3857)
        gdf = gpd.sjoin(gdf, boundary, how='inner', predicate='within').copy()
        return gdf, boundary
    else:
        print("  Real data not found — using synthetic LA demo …")
        rng = np.random.default_rng(42)
        # Small grid of random incidents around downtown / mid-city LA
        lons = rng.uniform(-118.45, -118.22, 400)
        lats = rng.uniform(33.95,    34.10,  400)
        df = pd.DataFrame({'LON': lons, 'LAT': lats,
                           'crime_weight': rng.integers(1, 6, 400).astype(float)})
        gdf = gpd.GeoDataFrame(df,
                               geometry=[Point(xy) for xy in zip(df['LON'], df['LAT'])],
                               crs=4326).to_crs(3857)
        # Fake boundary = bounding box
        from shapely.geometry import box
        bbox = box(*gdf.total_bounds)
        boundary = gpd.GeoDataFrame(geometry=[bbox], crs=3857)
        return gdf, boundary


def get_node_coords(G, node_id):
    """Return (x, y) in the graph's CRS for a given node."""
    d = G.nodes[node_id]
    return d['x'], d['y']


def draw_path_on_ax(ax, G, path_nodes, color, lw=2.0, alpha=0.85, zorder=5):
    """Draw each edge of a shortest-path route as a line on the axes."""
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        # Get edge geometry if it exists, else straight line between nodes
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            continue
        # Take the first parallel edge
        edata = list(edge_data.values())[0]
        if 'geometry' in edata:
            xs, ys = edata['geometry'].xy
        else:
            xu, yu = get_node_coords(G, u)
            xv, yv = get_node_coords(G, v)
            xs, ys = [xu, xv], [yu, yv]
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                solid_capstyle='round', zorder=zorder)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[1/5] Loading data …")
    gdf, boundary = load_data()
    print(f"   {len(gdf):,} incidents loaded.")

    # ── 2. Cluster into sectors to get HQ locations ───────────────────────────
    print("[2/5] Clustering incidents into sectors …")
    coords  = np.column_stack((gdf.geometry.x, gdf.geometry.y))
    weights = gdf['crime_weight'].values

    km = MiniBatchKMeans(n_clusters=NUM_SECTORS, n_init=5,
                         batch_size=min(10_000, len(gdf)),
                         random_state=42)
    km.fit(coords, sample_weight=weights)
    sector_hqs  = km.cluster_centers_          # shape (NUM_SECTORS, 2) EPSG:3857
    sec_labels  = km.labels_

    # Pick SAMPLE_HQS HQs that are well-spread (just take evenly-spaced indices)
    chosen_sec  = np.linspace(0, NUM_SECTORS-1, SAMPLE_HQS, dtype=int)

    # ── 3. Load OSM graph ─────────────────────────────────────────────────────
    print("[3/5] Loading OSM road network …")
    G_raw = load_or_download_graph(boundary)
    # Project to the same CRS as the graph's stored coordinates (usually UTM or 4326)
    G = ox.project_graph(G_raw, to_crs='EPSG:4326')

    # Helper: convert EPSG:3857 point → lon/lat for OSM snapping
    def to_lonlat(x3857, y3857):
        pt = gpd.GeoDataFrame(geometry=[Point(x3857, y3857)], crs=3857).to_crs(4326)
        return pt.geometry.x[0], pt.geometry.y[0]

    # ── 4. Compute paths ──────────────────────────────────────────────────────
    print("[4/5] Computing shortest road paths …")

    path_data = []   # list of dicts: {hq_idx, inc_idx, path_nodes, dist, color}

    for color, sec_idx in zip(HQ_COLORS, chosen_sec):
        hq_xy   = sector_hqs[sec_idx]            # EPSG:3857
        hq_lon, hq_lat = to_lonlat(*hq_xy)
        hq_node = ox.nearest_nodes(G, X=hq_lon,  Y=hq_lat)

        # Get all incidents in this sector, sorted by Euclidean distance to HQ
        mask     = sec_labels == sec_idx
        sec_gdf  = gdf[mask].copy()
        if len(sec_gdf) == 0:
            continue

        # Euclidean distance from each incident to HQ (quick pre-filter)
        dx = sec_gdf.geometry.x.values - hq_xy[0]
        dy = sec_gdf.geometry.y.values - hq_xy[1]
        euc_dist = np.sqrt(dx**2 + dy**2)
        # Take the closest incidents (slightly more than needed — some paths may fail)
        n_try    = min(INCIDENTS_PER_HQ * 3, len(sec_gdf))
        nearest_idx = np.argsort(euc_dist)[:n_try]
        candidates  = sec_gdf.iloc[nearest_idx]

        count = 0
        for _, row in candidates.iterrows():
            if count >= INCIDENTS_PER_HQ:
                break
            lon, lat = to_lonlat(row.geometry.x, row.geometry.y)
            inc_node = ox.nearest_nodes(G, X=lon, Y=lat)

            try:
                dist = nx.shortest_path_length(G, hq_node, inc_node, weight='length')
                if dist > MAX_PATH_DIST_M:
                    continue
                path = nx.shortest_path(G, hq_node, inc_node, weight='length')
                path_data.append({
                    'sec_idx':    sec_idx,
                    'hq_node':    hq_node,
                    'hq_xy':      hq_xy,
                    'inc_xy':     (row.geometry.x, row.geometry.y),
                    'path':       path,
                    'dist_m':     dist,
                    'dist_mi':    dist / 1609.34,
                    'color':      color,
                })
                count += 1
            except nx.NetworkXNoPath:
                continue

        print(f"   Sector {sec_idx}: {count} paths computed.")

    # ── 5. Visualise ──────────────────────────────────────────────────────────
    print("[5/5] Rendering …")

    # Get node positions for rendering (G is in EPSG:4326 — lon/lat)
    # We need to convert everything to a consistent plotting space.
    # Easiest: render the graph edges in lon/lat, and convert HQ/incident
    # markers from 3857 → 4326 as well.

    fig, axes = plt.subplots(1, SAMPLE_HQS,
                             figsize=(7 * SAMPLE_HQS, 9),
                             facecolor='#0d1117')

    if SAMPLE_HQS == 1:
        axes = [axes]

    for ax_idx, (ax, sec_idx, color) in enumerate(
            zip(axes, chosen_sec, HQ_COLORS)):

        ax.set_facecolor('#0d1117')

        # Filter paths for this sector
        sector_paths = [p for p in path_data if p['sec_idx'] == sec_idx]
        if not sector_paths:
            ax.text(0.5, 0.5, 'No paths found', color='white',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        # Collect all nodes in all paths to build a subgraph for background
        all_nodes = set()
        for p in sector_paths:
            all_nodes.update(p['path'])
        # Ego graph for visual context: background street network
        try:
            ego = nx.ego_graph(G, sector_paths[0]['hq_node'],
                               radius=MAX_PATH_DIST_M * 1.2, distance='length')
        except Exception:
            ego = G.subgraph(all_nodes)

        # Draw background street network (faint)
        for u, v, edata in ego.edges(data=True):
            if 'geometry' in edata:
                xs, ys = edata['geometry'].xy
            else:
                xu = G.nodes[u]['x']; yu = G.nodes[u]['y']
                xv = G.nodes[v]['x']; yv = G.nodes[v]['y']
                xs, ys = [xu, xv], [yu, yv]
            ax.plot(xs, ys, color='#2a2f3a', linewidth=0.5,
                    alpha=0.7, zorder=1)

        # Draw each path
        for p in sector_paths:
            draw_path_on_ax(ax, G, p['path'], color=color,
                            lw=2.5, alpha=0.9, zorder=5)

            # Incident marker
            inc_ll = gpd.GeoDataFrame(
                geometry=[Point(p['inc_xy'])], crs=3857).to_crs(4326)
            ax.scatter(inc_ll.geometry.x, inc_ll.geometry.y,
                       c='white', s=60, zorder=8, edgecolors=color,
                       linewidths=1.5)

            # Distance label midpoint of path
            mid_node = p['path'][len(p['path'])//2]
            mx = G.nodes[mid_node]['x']
            my = G.nodes[mid_node]['y']
            ax.text(mx, my, f"{p['dist_mi']:.2f} mi",
                    fontsize=6.5, color='white', zorder=9,
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='#0d1117', alpha=0.7,
                              edgecolor='none'))

        # HQ marker
        hq_ll = gpd.GeoDataFrame(
            geometry=[Point(sector_paths[0]['hq_xy'])], crs=3857).to_crs(4326)
        ax.scatter(hq_ll.geometry.x, hq_ll.geometry.y,
                   c=color, s=350, marker='H', zorder=10,
                   edgecolors='white', linewidths=2)
        ax.text(hq_ll.geometry.x[0],
                hq_ll.geometry.y[0] + 0.002,
                f"Sector {sec_idx}\nHQ",
                fontsize=8, color=color, zorder=11,
                ha='center', va='bottom', fontweight='bold')

        # Service radius circle (approximate in lon/lat degrees)
        deg_radius = MAX_PATH_DIST_M / 111_320   # rough metres→degrees
        circle = plt.Circle(
            (hq_ll.geometry.x[0], hq_ll.geometry.y[0]),
            deg_radius,
            color=color, fill=False,
            linewidth=1.2, linestyle='--', alpha=0.5, zorder=4
        )
        ax.add_patch(circle)

        # Axis limits: zoom to ego graph extent with padding
        all_x = [G.nodes[n]['x'] for n in ego.nodes]
        all_y = [G.nodes[n]['y'] for n in ego.nodes]
        pad   = deg_radius * 0.15
        ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
        ax.set_ylim(min(all_y)-pad, max(all_y)+pad)

        ax.set_title(
            f"Sector {sec_idx}  ·  {len(sector_paths)} sample paths",
            color='white', fontsize=11, pad=10, fontweight='bold'
        )
        ax.tick_params(colors='#555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2f3a')
        ax.set_xlabel("Longitude", color='#555', fontsize=7)
        ax.set_ylabel("Latitude",  color='#555', fontsize=7)

    # Shared legend
    legend_elements = [
        mlines.Line2D([0],[0], color=c, lw=2.5,
                      label=f"Sector {s} paths")
        for c, s in zip(HQ_COLORS, chosen_sec)
    ] + [
        mlines.Line2D([0],[0], color='white', marker='H',
                      markersize=10, linestyle='None', label='Sector HQ'),
        mlines.Line2D([0],[0], color='white', marker='o',
                      markersize=7, linestyle='None',
                      markeredgecolor=HQ_COLORS[0], label='Incident'),
        mlines.Line2D([0],[0], color='white', lw=1.2, linestyle='--',
                      alpha=0.5, label=f'{MAX_PATH_DIST_M/1609:.0f}-mi service radius'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=len(legend_elements), fontsize=8,
               facecolor='#1a1f2e', edgecolor='#2a2f3a',
               labelcolor='white', framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Road-Network Shortest Paths: Sector HQs → Sample Incidents",
        color='white', fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print(f"\n  Saved → {OUTPUT_PATH}")


if __name__ == '__main__':
    main()