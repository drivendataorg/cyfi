"""Geometry and polygon processing for CyFi."""
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point
from typing import Union
import pandas as pd
from pathlib import Path

def polygon_to_grid(
    polygon: Union[Polygon, str],
    resolution_meters: int = 20
) -> pd.DataFrame:
    """Convert a polygon to a grid of points."""
    # Load polygon if path provided
    if isinstance(polygon, str):
        if Path(polygon).exists():
            gdf = gpd.read_file(polygon)
            polygon = gdf.geometry.iloc[0]
    
    # Get bounds
    minx, miny, maxx, maxy = polygon.bounds
    
    # Convert meters to degrees (approximate)
    meter_to_deg = 0.000008983
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
                })
    
    return pd.DataFrame(points)

def generate_sample_csv(
    polygon_input: str,
    date: str,
    output_path: str,
    resolution_meters: int = 20
) -> int:
    """Generate CSV of sample points from polygon."""
    grid_df = polygon_to_grid(polygon_input, resolution_meters)
    
    if len(grid_df) == 0:
        raise ValueError("No points generated within polygon. Check polygon and resolution.")
    
    # Add date column
    grid_df['date'] = date
    
    # Select and reorder columns
    output_df = grid_df[['latitude', 'longitude', 'date']].copy()
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    
    n_points = len(grid_df)
    print(f"Generated {n_points} sample points at {output_path}")
    return n_points
