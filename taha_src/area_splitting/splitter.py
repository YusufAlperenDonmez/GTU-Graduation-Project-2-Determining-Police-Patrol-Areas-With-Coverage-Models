import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import json
from shapely.geometry import shape

st.set_page_config(page_title="LAPD Manual Sector Splitter", layout="wide")
st.title("LAPD Patrol: Manual Boundary Drawing")

st.sidebar.header("1. Upload Data")
uploaded_boundary = st.sidebar.file_uploader("Upload Master Boundary (GeoJSON)", type=['geojson'])

if uploaded_boundary:
    # 1. Load the master boundary
    boundary_gdf = gpd.read_file(uploaded_boundary)
    
    # FIX: Convert all non-geometry columns to strings so Folium's JSON parser doesn't crash
    for col in boundary_gdf.columns:
        if col != 'geometry':
            boundary_gdf[col] = boundary_gdf[col].astype(str)
    # Calculate map center dynamically based on the uploaded GeoJSON
    bounds = boundary_gdf.total_bounds # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Instructions")
        st.markdown("""
        1. Use the **Polygon Tool (⬟)** on the left side of the map.
        2. Click on the map to trace your custom sector boundaries.
        3. Double-click to close a shape. 
        4. Draw all 3 sectors. The app will automatically capture them.
        """)

    with col2:
        st.subheader("Interactive Map Area")
        # 2. Setup the map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")
        
        # Add the master boundary as an empty red outline so you know where to draw
        folium.GeoJson(
            boundary_gdf, 
            name="Master Boundary",
            style_function=lambda x: {'fillColor': 'none', 'color': 'red', 'weight': 2}
        ).add_to(m)
        
        # 3. Add the Drawing Tool
        # We disable markers and lines because we only want closed Polygons for patrol areas
        draw = Draw(
            export=False,
            draw_options={
                'polyline': False,
                'rectangle': False,
                'circle': False,
                'circlemarker': False,
                'marker': False,
                'polygon': True # Only allow polygons
            }
        )
        draw.add_to(m)
        
        # 4. Render the map and capture user interactions
        # st_folium returns a dictionary of the map state, including user drawings
        output = st_folium(m, width=800, height=600)
    
    # 5. Process the drawn shapes
    # If the user has drawn anything on the map, it appears in "all_drawings"
    if output and output.get("all_drawings"):
        drawings = output["all_drawings"]
        
        if len(drawings) > 0:
            with col1:
                st.success(f"Successfully captured {len(drawings)} custom sector(s)!")
                
                # Convert the raw JSON output from Folium into a GeoDataFrame
                features = []
                for i, draw_obj in enumerate(drawings):
                    geom = shape(draw_obj['geometry'])
                    features.append({'sector_name': f'Custom Sector {i+1}', 'geometry': geom})
                
                drawn_gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
                
                # Show a download button for the new boundaries
                geojson_data = drawn_gdf.to_json()
                st.download_button(
                    label="📥 Download Custom Sectors as GeoJSON",
                    data=geojson_data,
                    file_name="manual_custom_sectors.geojson",
                    mime="application/geo+json"
                )
                
                if st.button("🚀 Run Optimizer on these Sectors"):
                    st.info("Optimizer initiated! Passing the drawn GeoJSON to the PuLP solver...")
                    # Pass `drawn_gdf` to your IP code here.
                    
else:
    st.info("Please upload your Master Boundary GeoJSON in the sidebar to start drawing.")