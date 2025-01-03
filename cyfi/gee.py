import ee
import pandas as pd


def create_feature_collection(df):
    """Convert the CSV into an Earth Engine FeatureCollection"""
    features = []
    for row in df.itertuples():
        lat = row["latitude"]
        lon = row["longitude"]
        date = row["date"]

        # Create an ee.Feature for each row
        feature = ee.Feature(ee.Geometry.Point([lon, lat]), {"date": date})
        features.append(feature)

    # Return an Earth Engine FeatureCollection
    return ee.FeatureCollection(features)


def calculate_features_from_gee(feature):
    date = ee.Date(feature.get("date"))

    # Define the time range: 30 days before the given date
    start_date = date.advance(-30, "day")
    end_date = date

    # Create a 2,000 meter buffer around the point (circular buffer)
    point = feature.geometry()
    buffer = point.buffer(2000)  # Buffer is in meters (2 km radius)

    s2_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    cloud_score_collection = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    latest_image = (
        s2_collection.filterBounds(buffer)
        .filterDate(start_date, end_date)
        .filter(
            "CLOUDY_PIXEL_PERCENTAGE < 15"
        )  # exclude images where more than 5% of pixels in image are clouds
        .sort("system:time_start", False)  # sort so we can take most recent image
        .linkCollection(cloud_score_collection, ["cs_cdf"])  # bring in cloud score
    ).first()

    feature = feature.set({"image_id": latest_image.get("system:id")})

    masked_image = latest_image.updateMask(
        # mask out pixels where cloud score is too high
        latest_image.select("cs_cdf").gte(0.6)
    ).updateMask(
        # filter to water area
        latest_image.select("SCL").eq(6)
    )

    # add NDVI bands
    for band in ["B4", "B5", "B6", "B7"]:
        masked_image = masked_image.addBands(
            masked_image.normalizedDifference(["B8", band]).rename(f"NDVI_{band}")
        )

    # add ratio bands
    green_red_ratio = (
        masked_image.select("B3").divide(masked_image.select("B4")).rename("green_red_ratio")
    )
    green_blue_ratio = (
        masked_image.select("B3").divide(masked_image.select("B2")).rename("green_blue_ratio")
    )
    red_blue_ratio = (
        masked_image.select("B4").divide(masked_image.select("B2")).rename("red_blue_ratio")
    )
    masked_image = masked_image.addBands([green_red_ratio, green_blue_ratio, red_blue_ratio])

    # band means
    means = masked_image.select("B.*", "WVP", "AOT", "NDVI.*", ".*_ratio").reduceRegion(
        reducer=ee.Reducer.mean(), geometry=buffer, scale=10, maxPixels=1e8
    )

    # green percentiles
    percentiles = masked_image.select("B3").reduceRegion(
        reducer=ee.Reducer.percentile([5, 95]), geometry=buffer, scale=10, maxPixels=1e8
    )

    # percent water (use unmasked image)
    mean_water = (
        latest_image.select("SCL")
        .eq(6)
        .reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=10, maxPixels=1e8)
        .rename(["SCL"], ["percent_water"])
    )

    # inputs for AOT range
    aot_minmax = masked_image.select("AOT").reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=buffer,
        scale=10,
        maxPixels=1e8,
    )

    stats = means.combine(percentiles).combine(mean_water).combine(aot_minmax)
    return feature.set(stats)
