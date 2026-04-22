import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import Point


def get_waterbody_mask(
    sample_lat: float,
    sample_lon: float,
    bbox: list,
    shape: tuple,
    mask_gdf: gpd.GeoDataFrame,
    buffer_m: int = 50,
) -> np.ndarray:
    """Identify the target waterbody and create a binary mask matching the
    satellite image extent.

    Args:
        sample_lat (float): Latitude of sample point
        sample_lon (float): Longitude of sample point
        bbox (list): Bounding box [min_lon, min_lat, max_lon, max_lat]
        shape (tuple): (height, width) of the target mask
        mask_gdf (gpd.GeoDataFrame): Waterbody polygons (must be EPSG:4326)
        buffer_m (int): Search radius in meters around sample point to identify the waterbody

    Returns:
        np.ndarray: Boolean mask of the same shape as requested
    """
    # 1. Identify the target polygon
    point = Point(sample_lon, sample_lat)

    # Find polygons that contain the point
    matches = mask_gdf[mask_gdf.contains(point)]

    if matches.empty:
        # Try buffering the point to handle points on the shore
        # 50m is approximately 0.00045 degrees
        buffer_deg = buffer_m / 111320.0
        matches = mask_gdf[mask_gdf.intersects(point.buffer(buffer_deg))]

    if matches.empty:
        # If no waterbody found, return all ones (no spatial filtering applied)
        return np.ones(shape, dtype=bool)

    # Pick the first matching polygon
    target_poly = matches.iloc[0].geometry

    # 2. Rasterize the polygon into the bounding box
    # bbox: [min_lon, min_lat, max_lon, max_lat]
    # from_bounds: (west, south, east, north, width, height)
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], shape[1], shape[0])

    mask = features.rasterize(
        [target_poly],
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )

    return mask.astype(bool)
