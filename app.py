import requests
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from flask_restful import Resource, Api
import geopandas as gpd
from shapely.geometry import LineString, Polygon, Point
from pysheds.grid import Grid
from rasterio.transform import from_bounds
from shapely import geometry, ops
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.shutil import copy
from rasterio.io import MemoryFile
import json
import zipfile
import os
import pandas as pd
from PIL import Image
from geopy.geocoders import Nominatim
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak, Image, Paragraph, Spacer, Frame, PageTemplate, BaseDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime
import contextily as ctx
import matplotlib
matplotlib.use('Agg')

def basin_length(basin):
    gdf = basin
    gdf = gdf.to_crs(3857)
    max_distance = 0

    for geometry in gdf.geometry:
        if geometry.geom_type == 'Polygon':
            coords = np.array(geometry.exterior.coords)
        elif geometry.geom_type == 'MultiPolygon':
            # Flatten all coordinates from all polygons in the multipolygon
            coords = np.concatenate([np.array(polygon.exterior.coords) for polygon in geometry.geoms])
        
        # Compute the convex hull and find the maximum distance
        convex_hull = geometry.convex_hull
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance > max_distance:
                    max_distance = distance

    return max_distance / 1000 

#STREAM ORDER EXTRACTION
def calculate_strahler_numbers(gdf):
    """
    Calculate Strahler numbers for each LineString in a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame containing LineString geometries.

    Returns:
    GeoDataFrame: A GeoDataFrame with an additional 'Strahler' column.
    """
    def get_coords(line, index):
        return line.coords[index]

    def find_head_lines(lines):
        head_idx = []
        first_points = [get_coords(line, 0) for line in lines]
        last_points = [get_coords(line, -1) for line in lines]

        for i, first_point in enumerate(first_points):
            if first_point not in last_points:
                head_idx.append(i)
        return head_idx

    def find_next_line(curr_idx, lines, first_points):
        last_point = get_coords(lines[curr_idx], -1)
        try:
            next_idx = first_points.index(last_point)
        except ValueError:
            next_idx = None
        return next_idx

    def find_sibling_line(curr_idx, lines, last_points):
        last_point = get_coords(lines[curr_idx], -1)
        sibling_idx = last_points.index(last_point) if last_point in last_points else None
        return sibling_idx

    # Initialize Strahler number to 0
    gdf['stream_order'] = 0
    lines = list(gdf.geometry)
    first_points = [get_coords(line, 0) for line in lines]
    last_points = [get_coords(line, -1) for line in lines]

    # Find head lines
    head_idx = find_head_lines(lines)

    # Calculate Strahler numbers
    for idx in head_idx:
        curr_idx = idx
        curr_ord = 1
        gdf.at[curr_idx, 'stream_order'] = curr_ord  # head line always 1

        while True:
            next_idx = find_next_line(curr_idx, lines, first_points)
            if next_idx is None:
                break

            next_ord = gdf.at[next_idx, 'stream_order']
            sibl_idx = find_sibling_line(curr_idx, lines, last_points)
            if sibl_idx is not None:
                sibl_ord = gdf.at[sibl_idx, 'stream_order']

                if sibl_ord:
                    if sibl_ord > curr_ord:
                        break
                    elif sibl_ord < curr_ord:
                        if next_ord == curr_ord:
                            break
                    else:
                        curr_ord += 1

            gdf.at[next_idx, 'stream_order'] = curr_ord
            curr_idx = next_idx

    return gdf

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5000"])  # Enable CORS for all routes
api = Api(app)

@app.route('/')
def index():
    return render_template('index.html')

class Watershed(Resource):
    def post(self):
        data = request.get_json()
        x = data.get('x')
        y = data.get('y')

        if x is None or y is None:
            return {'message': 'Missing coordinates'}, 400

        try:
            x = float(x)
            y = float(y)
        except ValueError:
            return {'message': 'Invalid coordinates'}, 400
        

        bbox_size = 7.0  # Adjust this size as necessary

        # Define the GeoServer WMS service URL and the layer name
        wms_url = 'http://localhost:8080/geoserver/INDIA_DEM/wms'
        wcs_url = 'http://localhost:8080/geoserver/INDIA_DEM/wcs'
        layer_name = 'FDIR'  # Make sure the layer name is fully qualified with workspace

        # Construct the bounding box around the coordinates
        minx = x - bbox_size
        miny = y - bbox_size
        maxx = x + 1
        maxy = y + bbox_size
        # Define bbox size (degrees)
        x2 = round(x,3)
        y2 = round(y,3)
        point = Point(x2, y2)

        # Construct the WCS request parameters
        wcs_params = {
            'service': 'WCS',
            'version': '2.0.1',
            'request': 'GetCoverage',
            'coverageId': layer_name,
            'format': 'image/tiff',
            'subset': [
                f'Long({minx},{maxx})',
                f'Lat({miny},{maxy})'
            ]
        }

        # Print the WCS request URL for debugging
        wcs_request_url = f"{wcs_url}?service=WCS&version=2.0.1&request=GetCoverage&coverageId={layer_name}&format=image/tiff&subset=Long({minx},{maxx})&subset=Lat({miny},{maxy})"
        print(f"WCS Request URL: {wcs_request_url}")

        # Request DEM data from WCS service
        wcs_response = requests.get(wcs_url, params=wcs_params)

        # Check if the response is valid
        if wcs_response.status_code != 200 or 'image/tiff' not in wcs_response.headers.get('content-type', ''):
            print(f"Response Status Code: {wcs_response.status_code}")
            print(f"Response Headers: {wcs_response.headers}")
            return {'message': 'Error: Failed to retrieve valid TIFF data from the WCS service.'}, 500

         # Verify the file format and save as a Cloud Optimized GeoTIFF
        try:
            with MemoryFile(wcs_response.content) as memfile:
                with memfile.open() as src:
                    dem_data = src.read(1)
                    profile = src.profile

                cog_profile = profile.copy()
                cog_profile.update({
                    'driver': 'GTiff',
                    'tiled': True,
                    'blockxsize': 512,
                    'blockysize': 512,
                    'compress': 'deflate',
                    'interleave': 'band'
                 })

                with MemoryFile() as cog_memfile:
                    with cog_memfile.open(**cog_profile) as cog:
                        cog.write(dem_data, 1)
                        copy(cog, 'data/cloud_optimized_dem.tiff', **cog_profile)

            print("Cloud Optimized GeoTIFF saved as 'cloud_optimized_dem.tiff'")

        except rasterio.errors.RasterioIOError:
            return 'Error: The downloaded file is not a valid TIFF format.'

        # Load the DEM data using pysheds
        grid = Grid.from_raster('data/cloud_optimized_dem.tiff')
        fdir = grid.read_raster('data/cloud_optimized_dem.tiff')

        # Specify directional mapping
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        acc = grid.accumulation(fdir, dirmap=dirmap)

        # Snap pour point to high accumulation cell
        x_snap, y_snap = grid.snap_to_mask(acc > 10000, (x, y))

        # Delineate the catchment
        catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')

        # Clip and plot the catchment
        grid.clip_to(catch)
        clipped_catch = grid.view(catch)

        # Polygonize catchment
        shapes = grid.polygonize()
        catchment_polygon = ops.unary_union([geometry.shape(shape) for shape, value in shapes])
        try:

            if catchment_polygon.geom_type == 'GeometryCollection':
                for geom in catchment_polygon.geoms:
                    if geom.geom_type == 'Polygon':
                        geom_simplified = geom.simplify(tolerance=0.0017, preserve_topology=True)
                        x, y = geom_simplified.exterior.xy
            elif catchment_polygon.geom_type == 'MultiPolygon':
                for polygon in catchment_polygon.geoms:
                    polygon_simplified = polygon.simplify(tolerance=0.0017, preserve_topology=True)
                    x, y = polygon_simplified.exterior.xy
            else:
                catchment_polygon_simplified = catchment_polygon.simplify(tolerance=0.0017, preserve_topology=True)
                x, y = catchment_polygon_simplified.exterior.xy
            gdf_basin = gpd.GeoDataFrame(geometry=[catchment_polygon_simplified], crs=4326)
        except:
            if catchment_polygon.geom_type == 'GeometryCollection':
                for geom in catchment_polygon.geoms:
                    if geom.geom_type == 'Polygon':
                        x, y = geom.exterior.xy
            elif catchment_polygon.geom_type == 'MultiPolygon':
                for polygon in catchment_polygon.geoms:
                    x, y = polygon.exterior.xy
            else:
                x, y = catchment_polygon.exterior.xy
            gdf_basin = gpd.GeoDataFrame(geometry=[catchment_polygon], crs=4326)
        gdf_basin = gdf_basin.to_crs(3857)
        # Extract river network
        branches = grid.extract_river_network(fdir, acc > 500    , dirmap=dirmap)

        lines = [LineString(branch['geometry']['coordinates']) for branch in branches['features']]

        # Create a GeoDataFrame for the stream network
        stream_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
        stream_gdf = calculate_strahler_numbers(stream_gdf)
        stream_gdf = stream_gdf.to_crs(3857)
        stream_gdf['length'] = round(stream_gdf.geometry.length/1000,2)
        stream_gdf = stream_gdf.to_crs(4326)
        


        gdf_basin['perimeter'] = round(gdf_basin.geometry.length/1000,2)
        gdf_basin['area'] = round(gdf_basin.geometry.area/1000000,2)
        gdf_basin['basin_length'] = round(basin_length(gdf_basin),5)
        order = list(stream_gdf['stream_order'].unique())
        # Morphometry Analysis

        #Drainage Density Dd
        
        gdf_basin['Dd'] = round((sum(stream_gdf['length']))/gdf_basin['area'],4)

        #Form Factor F
        gdf_basin['F'] = round(gdf_basin['area']/(gdf_basin['basin_length']**2),4)

        #Circulatory Index Rc
        constant = 4 * 3.142
        gdf_basin['Rc'] = round((constant*gdf_basin['area'])/(gdf_basin['perimeter']**2),4)

        #Elongatio Ratio Re
        nominator = 1.129*(gdf_basin['area']**(1/2))
        gdf_basin['Re'] = round(nominator / gdf_basin['basin_length'],4)
        gdf_basin = gdf_basin.to_crs(4326)
        outlet_point = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
     # Save the GeoDataFrame to a shapefile
        outlet = outlet_point.to_json()
        basin = gdf_basin.to_json()
        stream = stream_gdf.to_json()

        return jsonify([json.loads(basin), json.loads(stream), json.loads(outlet)])
    
@app.route('/download', methods=['POST'])
def download():
    data = request.json
    basin_data = data['basin']
    stream_data = data['streams']
    
    # Convert GeoJSON to GeoDataFrames
    basin_gdf = gpd.GeoDataFrame.from_features(basin_data['features'], crs=4326)
    streams_gdf = gpd.GeoDataFrame.from_features(stream_data['features'], crs=4326)
    
    temp_dir = 'data'
    # Save GeoDataFrames to shapefiles
    basin_gdf.to_file(os.path.join(temp_dir, 'basin.shp'))
    streams_gdf.to_file(os.path.join(temp_dir, 'streams.shp'))
    
    # Create a zip file containing the shapefiles
    zip_filename = 'watershed_shapefiles.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))
    
    return jsonify({'download_url': f'/{zip_filename}'})

@app.route('/watershed_shapefiles.zip', methods=['GET'])
def download_file():
    return send_file('watershed_shapefiles.zip', as_attachment=True)



def fetch_places_within_bounds(gdf):
    places = []

    for index, row in gdf.iterrows():
        geometry = row['geometry']
        area = row['area']
        coords = []

        if geometry.geom_type == 'Polygon':
            coords.extend(list(geometry.exterior.coords))
        elif geometry.geom_type == 'MultiPolygon':
            for poly in geometry.geoms:
                coords.extend(list(poly.exterior.coords))

        # Separate latitude and longitude and round to 1 decimal place
        latitudes = set(round(coord[1], 1) for coord in coords)
        longitudes = set(round(coord[0], 1) for coord in coords)

        # Determine downsampling based on area
        if area > 10000:
            latitudes = list(latitudes)[0:len(latitudes):5]
            longitudes = list(longitudes)[0:len(longitudes):5]
        elif 5000 < area < 1000:
            latitudes = list(latitudes)[0:len(latitudes):4]
            longitudes = list(longitudes)[0:len(longitudes):4]
        elif 1000 < area < 500:
            latitudes = list(latitudes)[0:len(latitudes):3]
            longitudes = list(longitudes)[0:len(longitudes):3]
        elif 500 < area < 250:
            latitudes = list(latitudes)[0:len(latitudes):2]
            longitudes = list(longitudes)[0:len(longitudes):2]
        else:
            latitudes = list(latitudes)[0:len(latitudes):1]
            longitudes = list(longitudes)[0:len(longitudes):1]

        user_agent = "HydroMorphoLabs (www.example1234@gmail.com)"
        geolocator = Nominatim(user_agent=user_agent)

        # Fetch places for corners of the bounding box
        for lat in latitudes:
            for lon in longitudes:
                try:
                    location = geolocator.reverse((lat, lon), exactly_one=True)
                    if location:
                        place = {'Places': location.address}
                        places.append(place)
                except Exception as e:
                    print(f"Error fetching location for ({lat}, {lon}): {e}")

    places_df = pd.DataFrame(places)
    return places_df

#Morphometry analysis
#Morphometry analysis
def morphometry_calculation(streams_gdf, basin_gdf):
    bounds = basin_gdf.total_bounds
    min_lon, min_lat, max_lon, max_lat = bounds
    basin_gdf['Centroid'] = basin_gdf.geometry.centroid.apply(lambda p: (p.x, p.y))
    

    basin_gdf = basin_gdf.to_crs(epsg = 3857)
    streams_gdf = streams_gdf.to_crs(epsg = 3857)
    # Calculate parameters and create DataFrame
    gdf1 = streams_gdf  # Use streams_gdf as gdf1
    df = pd.DataFrame()
    
    for i in range(1, len(gdf1['stream_order'].unique()) + 1):
        df[f'Number of stream orders (Nu) {i}'] = [len(gdf1[gdf1['stream_order'] == i])]
        df[f'Length of stream orders (Lu)(Kms) {i}'] = [sum(gdf1[gdf1['stream_order'] == i]['length']) / 1000]
        if i < len(gdf1['stream_order'].unique()):
            df[f'Bifurcation Ratio (Rb){i}:{i+1}'] = (i)/(i+1)

    df['Total Stream Order Nu'] = len(gdf1['stream_order'])

    # Calculate mean bifurcation ratio
    bifurcation_columns = df.filter(like='Bifurcation Ratio (Rb)')
    if not bifurcation_columns.empty:
        df['Mean bifurcation ratio (Rbm)'] = bifurcation_columns.mean().mean()
    else:
        df['Mean bifurcation ratio (Rbm)'] = None

    # Morphometry Parameters
    # Study Area
    # Extent
    df['xmin'] = min_lon
    df['xmax'] =  max_lon
    df['ymin'] = min_lat 
    df['ymax'] = max_lat
    df['centroid'] = basin_gdf['Centroid']
    #imp_places = []
    df['Total Stream Length'] = sum(gdf1['length']) / 1000
    df['Basin area (A) (Sq Km)'] = basin_gdf['area'].mean()
    df['Basin perimeter (P) (Kms)'] = basin_gdf['perimeter'].mean()
    df['Basin Length (Lb) (Kms)'] = basin_gdf['basin_length'].mean()
    df['Form factor (Ff)'] = basin_gdf['F'].mean()
    df['Shape Factor (Sf)'] = 1 / basin_gdf['F'].mean()
    df['Relative perimeter (Rp)'] = basin_gdf['area'].mean() / basin_gdf['perimeter'].mean()
    df['Length Area Relation (Lar)'] = 1.4 * (basin_gdf['area'].mean() ** 0.6)
    df['Rotundity coefficient (Rc)'] = ((basin_gdf['basin_length'].mean() ** 2) * 3.142) / (4 * basin_gdf['area'].mean())
    df['Mean Basin Width (Wb)'] = basin_gdf['area'].mean() / basin_gdf['basin_length'].mean()
    df['Compactness Coefficient (Cc)'] = 0.282 * basin_gdf['perimeter'].mean() / (basin_gdf['area'].mean() ** 0.5)
    df['Circularity ratio (Rc)'] = (4 * 3.142 * basin_gdf['area'].mean()) / (basin_gdf['perimeter'].mean() ** 2)
    nominator = 1.129 * (basin_gdf['area'].mean() ** 0.5)
    df['Elongation ratio (Re)'] = nominator / df['Basin Length (Lb) (Kms)']
    df['Drainage density (Dd) (km/km2)'] = df['Total Stream Length'] / basin_gdf['area'].mean()
    df['Stream frequency; (F) (number/km2)'] = len(gdf1['stream_order'].unique()) / basin_gdf['area'].mean()
    df['Ccm (km2/km)'] = 1 / df['Drainage density (Dd) (km/km2)']
    df['Infiltration Number (Ifn)'] = df['Stream frequency; (F) (number/km2)'] * df['Drainage density (Dd) (km/km2)']
    df['Drainage Intensity (Di)'] = df['Stream frequency; (F) (number/km2)'] / df['Drainage density (Dd) (km/km2)']
    df['Average Length of Overland Flow (Lg) (Kms)'] = 0.5 * df['Drainage density (Dd) (km/km2)']
    df['Drainage Texture (Dt)'] = len(gdf1['stream_order']) / basin_gdf['perimeter'].mean()

    return df

#map plotting
def plotting(streams_gdf,basin_gdf, outlet_gdf ):
     # Plotting
    fig, ax = plt.subplots(figsize=(20, 22))
    streams_gdf.plot(ax=ax, column='stream_order', legend=True, categorical=True, cmap='rainbow')
    basin_gdf.plot(ax=ax, alpha=0.5, color='#c1fc5b', edgecolor = 'k', linewidth = 1)
    outlet_gdf.plot(ax = ax, marker='*', color='black',markersize=100)
    leg = ax.get_legend()
    leg.set_bbox_to_anchor((1.2, 1))  # (x, y) position outside the plot
    leg.set_frame_on(False)
    leg.set_title('Stream Order', prop={'size': 20, 'weight': 'bold'})

    plt.title('Stream Order Map',fontsize=28, fontweight='bold')
    # Save GeoDataFrame plot as an image
    plt.savefig('data/gdf_plot.png', format='png', bbox_inches='tight')
    plt.close()

    #Dark theme map
    # Plotting
    fig, ax = plt.subplots(figsize=(20, 22))
    streams_gdf.plot(ax=ax, column = 'stream_order', legend=True, categorical=True, cmap = 'cool')
    basin_gdf.plot(ax=ax, color='none', edgecolor='#83fa50',  linewidth = 1)
    outlet_gdf.plot(ax =ax ,marker='*', color='#ff425b',markersize=100)
    leg = ax.get_legend()
    leg.set_bbox_to_anchor((1.2, 1))  # (x, y) position outside the plot
    leg.set_frame_on(False)
    leg.set_title('Stream Order', prop={'size': 20, 'weight': 'bold'})
    
    ctx.add_basemap(ax, crs=basin_gdf.crs, source=ctx.providers.CartoDB.DarkMatter)
        
    plt.title('Dark-Theme Stream Order Map', fontsize=28, fontweight='bold')


    # Save GeoDataFrame plot as an image
    plt.savefig('data/darkmap_plot.png', format='png', bbox_inches='tight')
    plt.close()

    #Satellite imagery map
    fig, ax = plt.subplots(figsize=(20, 22))
    streams_gdf.plot(ax=ax, color='yellow')
    basin_gdf.plot(ax=ax, color='none', edgecolor='y',  linewidth = 3)
    outlet_gdf.plot(ax =ax ,marker='*', color='blue',markersize=100)

    ctx.add_basemap(ax, crs=basin_gdf.crs, source=ctx.providers.Esri.WorldImagery)
    
    plt.title('Basin Satellite Imagery Map',fontsize=28, fontweight='bold')
    plt.savefig('data/satellitemap.png', format='png', bbox_inches='tight')
    plt.close()
    
    #TOPOMAP
    fig, ax = plt.subplots(figsize=(20, 22))
    streams_gdf.plot(ax=ax, column = 'stream_order', legend=True, categorical=True, cmap = 'plasma')
    basin_gdf.plot(ax=ax, color='none', edgecolor='k',  linewidth = 3)
    outlet_gdf.plot(ax =ax ,marker='*', color='blue',markersize=100)
    leg = ax.get_legend()
    leg.set_bbox_to_anchor((1.2, 1))  # (x, y) position outside the plot
    leg.set_frame_on(False)
    leg.set_title('Stream Order',prop={'size': 20, 'weight': 'bold'})
    
    ctx.add_basemap(ax, crs=basin_gdf.crs, source=ctx.providers.Esri.WorldTopoMap)
    plt.title('Basin Topography Map', fontsize=28, fontweight='bold')
    plt.savefig('data/topomap.png', format='png', bbox_inches='tight')
    plt.close()

    #LOCATION MAP
    fig, ax = plt.subplots(figsize=(20, 22))
    basin_gdf.plot(ax=ax, color='blue', alpha = 1, edgecolor='k', linewidth=2)
    
    # Set the scaling factor (1.5 for 50% zoom out)
    scaling_factor = 25
    
    # Get current bounds
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # Calculate the new bounds
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    x_range = (xmax - xmin) * scaling_factor / 2
    y_range = (ymax - ymin) * scaling_factor / 2
    
    # Set new limits
    ax.set_xlim([x_center - x_range, x_center + x_range])
    ax.set_ylim([y_center - y_range, y_center + y_range])
    
    # Add the basemap
    ctx.add_basemap(ax, crs=basin_gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)
    plt.title('Loaction Map of Basin', fontsize=28, fontweight='bold')
    plt.savefig('data/locatiommap.png', format='png', bbox_inches='tight')
    plt.close()


@app.route('/download_csv', methods=['POST'])
def download_csv():
    data = request.json
    basin_data = data['basin']
    stream_data = data['streams']
    
    # Convert GeoJSON to GeoDataFrames
    basin_gdf = gpd.GeoDataFrame.from_features(basin_data['features'], crs=4326)
    streams_gdf = gpd.GeoDataFrame.from_features(stream_data['features'], crs=4326)

    df = morphometry_calculation(streams_gdf=streams_gdf, basin_gdf=basin_gdf)
    sorted_columns = sorted(df.columns)
    df = df[sorted_columns]
    df = df.T
    df = df.reset_index()
    df = df.sort_values(by=['index'])
    df = df.reset_index()
    df.drop('level_0', axis=1, inplace=True)
    df.to_csv("data/temp.csv", index=True, header=False)

    return send_file('data/temp.csv', as_attachment=True, download_name='morphometry_report.csv')


#save as pdf
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    data = request.json
    basin_data = data['basin']
    stream_data = data['streams']
    outlet_data = data['outlet']
    
    # Convert GeoJSON to GeoDataFrames
    basin_gdf = gpd.GeoDataFrame.from_features(basin_data['features'], crs=4326)
    streams_gdf = gpd.GeoDataFrame.from_features(stream_data['features'], crs=4326)
    outlet_gdf = gpd.GeoDataFrame.from_features(outlet_data['features'], crs=4326)

    df = morphometry_calculation(streams_gdf=streams_gdf, basin_gdf=basin_gdf)
    plotting(streams_gdf=streams_gdf, basin_gdf=basin_gdf, outlet_gdf=outlet_gdf)
    places_df = fetch_places_within_bounds(basin_gdf)

    df = df.T
    df = df.reset_index()
    df = df.sort_values(by=['index'])
    df = df.reset_index()
    df.drop('level_0', axis=1, inplace=True)

    # Convert all DataFrame values to strings to avoid issues with floats
    df = df.astype(str)
    places_df = places_df.astype(str)


    # Define constants
    PAGE_WIDTH, PAGE_HEIGHT = A4
    MARGIN = 1 * inch
    TABLE_WIDTH = PAGE_WIDTH - 2 * MARGIN
    MAX_ROWS_PER_PAGE = 35  # Adjust this based on your layout

    # Create a PDF document
    pdf_file = "data/output.pdf"
    doc = BaseDocTemplate(pdf_file, pagesize=A4, rightMargin=MARGIN, leftMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)

    # Create a frame for the page template
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height - 0.5 * inch, id='normal')

    # Define a function for the footer with timestamp and page numbers
    def footer(canvas, doc):
        canvas.saveState()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_text = f"Generated on {timestamp}  |  Page {doc.page}"

        canvas.setFont('Times-Roman', 10)
        canvas.drawString(MARGIN, 0.75 * inch, footer_text)  # Positioning the footer
        canvas.restoreState()

    # Add the page template with the footer function
    doc.addPageTemplates([PageTemplate(id='default', frames=frame, onPage=footer)])

    # Load the CSV data
    data1 = df  # Ensure `df` is your first DataFrame
    data2 = places_df  # Load the second CSV file

    # Prepare table data for both DataFrames
    styles = getSampleStyleSheet()
    styleN = styles['BodyText']

    def prepare_table_data(data):
        # Convert each cell into a Paragraph, except for header rows
        return [[Paragraph(str(cell), styleN) for cell in row] for row in [data.columns.tolist()] + data.values.tolist()]

    def split_table(data, max_rows_per_page):
        num_pages = (len(data) // max_rows_per_page) + 1
        return [data[i*max_rows_per_page:(i+1)*max_rows_per_page] for i in range(num_pages)]

    # Prepare table pages for df
    table_data1 = prepare_table_data(data1)
    table_pages1 = split_table(table_data1, MAX_ROWS_PER_PAGE)

    # Prepare table pages for places_df
    table_data2 = prepare_table_data(data2)
    table_pages2 = split_table(table_data2, MAX_ROWS_PER_PAGE)

    # Create a list to store PDF elements
    elements = []

    # Add a title page with custom font and style
    title_style = ParagraphStyle(
        name='TitleStyle',
        fontName='Times-Bold',  # Font family and weight
        fontSize=36,
        alignment=1,  # Center alignment
        spaceAfter=36  # Space below the title
    )

    subtitle_style = ParagraphStyle(
        name='SubTitle',
        fontName='Times-Bold',  # Font family and weight
        fontSize=22,
        alignment=1,  # Center alignment
        leading=20,
        spaceAfter=16  # Space below the title
    )

    # Calculate the center position for the title (considering margins)
    center_y = (PAGE_HEIGHT - MARGIN * 2) / 2
    title_height = 36  # Approximate height of the title in points
    subtitle_height = 4 * 14 + 3 * 6  # Approximate height of the subtitle in points (4 lines, 3 gaps)

    # Spacer to center the title
    elements.append(Spacer(1, center_y - title_height / 2 - subtitle_height / 2))

    title = Paragraph("River Morphometry Report<br/><br/>", title_style)
    subtitle = Paragraph(
        "Module Developed By Sumanth M<br/><br/>"
        "M.Sc in GIS<br/><br/>"
        "Centre for Geoinformatics Technology<br/><br/>"
        "DOS Geography, Manasagangotri Campus<br/><br/>"
        "University of Mysore",
        subtitle_style
    )
    elements.append(title)
    elements.append(subtitle)
    elements.append(PageBreak())

    # Define a function to scale the image to fit the page size
    def add_scaled_image(image_path, max_width, max_height):
        img = Image(image_path)
        aspect_ratio = img.imageWidth / float(img.imageHeight)

        # Determine whether to scale by width or height
        if aspect_ratio > 1:
            # Landscape orientation: fit to page width
            width = min(max_width, img.imageWidth)
            height = width / aspect_ratio
        else:
            # Portrait orientation: fit to page height
            height = min(max_height, img.imageHeight)
            width = height * aspect_ratio

        # Adjust dimensions if they exceed max values
        if width > max_width:
            width = max_width
            height = width / aspect_ratio
        if height > max_height:
            height = max_height
            width = height * aspect_ratio

        img.drawWidth = width
        img.drawHeight = height

        # Center the image on the page
        img.hAlign = 'CENTER'

        return img

    # Maximum dimensions for images
    max_width = PAGE_WIDTH - 2 * MARGIN
    max_height = PAGE_HEIGHT - 2 * MARGIN

    # Add the plot images to the document with aspect ratio preservation
    image_files = ["data/locatiommap.png", "data/gdf_plot.png", "data/satellitemap.png", "data/topomap.png", "data/darkmap_plot.png"]

    for image_path in image_files:
        image = add_scaled_image(image_path, max_width, max_height)
        elements.append(image)
        elements.append(PageBreak())

    # Function to add tables to the PDF elements
    def add_table_pages(table_pages, data, elements):
        for page_data in table_pages:
            col_widths = [TABLE_WIDTH / len(data.columns)] * len(data.columns)
            table = Table(page_data, colWidths=col_widths)

            # Adjust table styling
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Justify text to the left
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('WORDSPACE', (0, 0), (-1, -1), 0),
            ]))

            # Add the table to the elements
            elements.append(table)
            if page_data is not table_pages[-1]:  # Add page break if not the last page
                elements.append(PageBreak())

    # Add tables for df
    add_table_pages(table_pages1, data1, elements)

    # Add tables for places_df
    add_table_pages(table_pages2, data2, elements)

    # Add an end page with a custom title style
    end_title_style = ParagraphStyle(
        name='EndTitleStyle',
        fontName='Times-Bold',  # Font family and weight
        fontSize=20,
        alignment=1,  # Center alignment
        spaceAfter=12  # Space below the title
    )

    end_title = Paragraph("<br/><br/>End of Report", end_title_style)
    elements.append(end_title)
    elements.append(PageBreak())

    # Build the PDF document
    doc.build(elements)


    return send_file(pdf_file, as_attachment=True, download_name='morphometry_report.pdf')


api.add_resource(Watershed, '/')

# Main Driver Function
if __name__ == '__main__':
    app.run(debug=True)
