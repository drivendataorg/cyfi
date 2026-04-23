"""Geometry and polygon processing for CyFi."""
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point
from typing import Union, List, Tuple
import pandas as pd

def polygon_to_grid(
    polygon: Union[Polygon, str, gpd.GeoSeries],
    resolution_meters: int = 20,
    crs: str = "EPSG:4326"
) -> pd.DataFrame:
    """
    Convert a polygon to a grid of points at specified resolution.
    
    Args:
        polygon: Shapely Polygon, GeoJSON string, or path to GeoJSON file
        resolution_meters: Grid resolution in meters (Sentinel-2 is 10-20m)
        crs: Coordinate reference system
    
    Returns:
        DataFrame with columns: latitude, longitude, geometry
    """
    # Load polygon if path provided
    if isinstance(polygon, str):
        gdf = gpd.read_file(polygon)
        polygon = gdf.geometry.iloc[0]
    
    # Get bounds
    minx, miny, maxx, maxy = polygon.bounds
    
    # Calculate number of points based on resolution
    # Convert meters to degrees (approximate)
    meter_to_deg = 0.00001  # ~1m = 0.00001 degrees
    step = resolution_meters * meter_to_deg
    
    # Create grid
    x_coords = np.arange(minx, maxx, step)
    y_coords = np.arange(miny, maxy, step)
    
    points = []
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            if polygon.contains(point):
                points.append({
                    'latitude': y,
                    'longitude': x,
                    'geometry': point
                })
    
    return pd.DataFrame(points)

def generate_sample_csv(
    polygon_input: str,
    date: str,
    output_path: str,
    resolution_meters: int = 20
) -> None:
    """
    Generate a CSV of sample points from a polygon.
    
    Args:
        polygon_input: Path to GeoJSON or shapefile
        date: Date for prediction (YYYY-MM-DD)
        output_path: Output CSV path
        resolution_meters: Grid resolution
    """
    # Generate grid points
    grid_df = polygon_to_grid(polygon_input, resolution_meters)
    
    # Add date column
    grid_df['date'] = date
    
    # Save to CSV
    grid_df[['latitude', 'longitude', 'date']].to_csv(output_path, index=False)
    print(f"Generated {len(grid_df)} sample points at {output_path}")
