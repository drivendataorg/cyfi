import ee
import pandas as pd


# TODO: filter based on clouds in bbox not image (should be recalculated)
# TODO: winsorize band values
# TODO: make bbox rectangular?
# TODO: parameterize things
# TODO: calculate averages first for ratios?


def create_feature_collection(df):
    """Convert the CSV into an Earth Engine FeatureCollection"""
    features = [
        ee.Feature(ee.Geometry.Point([row.longitude, row.latitude]), {"date": row.date})
        for row in df.itertuples()
    ]
    return ee.FeatureCollection(features)


def _calculate_gee_satellite_features_for_sample_item(feature):
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
            "CLOUDY_PIXEL_PERCENTAGE < 5"
        )  # exclude images where more than 5% of pixels in image are clouds
        .sort("system:time_start", False)  # sort so we can take most recent image
        .linkCollection(cloud_score_collection, ["cs_cdf"])  # bring in cloud score
    ).first()

    feature = feature.set(
        {
            "image_id": latest_image.get("system:id"),
            "image_date": ee.Date(latest_image.get("system:time_start")).format(
                "YYYY-MM-dd HH:mm:ss"
            ),
        }
    )

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


def calculate_satellite_features_gee(samples, features_config):
    points_fc = create_feature_collection(samples)
    results_fc = points_fc.map(_calculate_gee_satellite_features_for_sample_item)
    features = pd.DataFrame([r["properties"] for r in results_fc.getInfo()["features"]])
    features.index = samples.index
    features = features.rename(
        columns={
            "B1": "B01_mean",
            "B2": "B02_mean",
            "B3": "B03_mean",
            "B4": "B04_mean",
            "B5": "B05_mean",
            "B6": "B06_mean",
            "B7": "B07_mean",
            "B8": "B08_mean",
            "B8A": "B8A_mean",
            "B9": "B09_mean",
            "B10": "B10_mean",
            "B11": "B11_mean",
            "B12": "B12_mean",
            "WVP": "WVP_mean",
            "AOT": "AOT_mean",
            "B3_p5": "green5th",
            "B3_p95": "green95th",
            "NDVI_B4": "NDVI_B04",
            "NDVI_B5": "NDVI_B05",
            "NDVI_B6": "NDVI_B06",
            "NDVI_B7": "NDVI_B07",
            "image_id": "item_id",
        }
    )

    # create additional features
    features["green95th_blue_ratio"] = features.green95th / features.B02_mean
    features["green5th_blue_ratio"] = features.green5th / features.B02_mean
    features["AOT_range"] = features.AOT_max - features.AOT_min

    features["days_before_sample"] = (
        pd.to_datetime(samples.date) - pd.to_datetime(features.image_date)
    ).dt.days
    features["month"] = pd.to_datetime(features.image_date).dt.month
    return features
