"""Tests for polygon/grid functionality (Issue #132)."""
import pytest
import pandas as pd
from pathlib import Path
import json

def test_polygon_to_grid():
    """Test grid generation from polygon."""
    from cyfi.geometry import polygon_to_grid
    from shapely.geometry import Polygon
    
    # Create a small square polygon
    polygon = Polygon([
        (-73.206937, 41.424144),
        (-73.200000, 41.424144),
        (-73.200000, 41.430000),
        (-73.206937, 41.430000)
    ])
    
    # Generate grid at coarse resolution
    grid = polygon_to_grid(polygon, resolution_meters=100)
    
    assert len(grid) > 0
    assert 'latitude' in grid.columns
    assert 'longitude' in grid.columns

def test_generate_sample_csv(tmp_path):
    """Test CSV generation from polygon."""
    from cyfi.geometry import generate_sample_csv
    
    # Create temporary GeoJSON with proper polygon (closed ring)
    geojson_data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-73.206937, 41.424144],
                    [-73.200000, 41.424144],
                    [-73.200000, 41.430000],
                    [-73.206937, 41.430000],
                    [-73.206937, 41.424144]  # Close the ring
                ]]
            }
        }]
    }
    
    geojson_path = tmp_path / "test_polygon.geojson"
    with open(geojson_path, 'w') as f:
        json.dump(geojson_data, f)
    
    # Generate CSV
    output_csv = tmp_path / "samples.csv"
    n_points = generate_sample_csv(str(geojson_path), "2023-07-01", str(output_csv), resolution_meters=100)
    
    assert output_csv.exists()
    assert n_points > 0
    assert isinstance(n_points, int)
    
    df = pd.read_csv(output_csv)
    assert 'latitude' in df.columns
    assert 'longitude' in df.columns
    assert 'date' in df.columns
    assert len(df) == n_points

def test_no_points_in_polygon(tmp_path):
    """Test error when polygon contains no points."""
    from cyfi.geometry import generate_sample_csv
    
    # Create a tiny polygon that won't contain any grid points
    geojson_data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-73.206937, 41.424144],
                    [-73.206935, 41.424144],
                    [-73.206935, 41.424146],
                    [-73.206937, 41.424146],
                    [-73.206937, 41.424144]  # Close the ring
                ]]
            }
        }]
    }
    
    geojson_path = tmp_path / "tiny_polygon.geojson"
    with open(geojson_path, 'w') as f:
        json.dump(geojson_data, f)
    
    output_csv = tmp_path / "empty.csv"
    
    # This should raise ValueError
    with pytest.raises(ValueError, match="No points generated"):
        generate_sample_csv(str(geojson_path), "2023-07-01", str(output_csv), resolution_meters=100)
