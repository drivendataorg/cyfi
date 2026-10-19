import ee
import pandas as pd

# Initialize the Earth Engine API
ee.Initialize()


# Function to load CSV and create a FeatureCollection
def load_points_from_csv(csv_file_path):
    """Load points from a CSV and return as a FeatureCollection."""
    df = pd.read_csv(csv_file_path)
    features = [
        ee.Feature(ee.Geometry.Point([row["longitude"], row["latitude"]]), {"date": row["date"]})
        for _, row in df.iterrows()
    ]
    return ee.FeatureCollection(features)


# Function to filter Sentinel-2 and Cloud Score+ collections within a time range and spatial area
def filter_collections(feature, buffer, start_date, end_date):
    """Filter Sentinel-2 and Cloud Score+ collections by date and spatial region."""
    s2_sr_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(buffer)
        .filterDate(start_date, end_date)
    )

    cloud_score_collection = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filterBounds(buffer)
        .filterDate(start_date, end_date)
    )

    # Link the collections based on time and spatial proximity
    linked_collection = s2_sr_collection.linkCollection(cloud_score_collection)
    return linked_collection.sort("system:time_start", False).first()


# Function to calculate cloud percentage in an image
def calculate_cloud_percentage(image, buffer):
    """Calculate the percentage of cloud-covered pixels in the image."""
    cloud_score = image.select("cloud_score")
    cloud_mask = cloud_score.gt(50)  # Define cloud mask based on Cloud Score threshold

    # Calculate the total number of cloud pixels
    cloud_pixels = cloud_mask.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=buffer, scale=10, maxPixels=1e8
    ).get("cloud_score")

    # Calculate total pixels in the buffer area (in 10m resolution)
    total_pixels = buffer.area().divide(
        10 * 10
    )  # Convert area to pixel count (10m x 10m resolution)

    # Compute cloud percentage
    cloud_percentage = ee.Number(cloud_pixels).divide(total_pixels).multiply(100)
    return cloud_percentage


# Function to get the most recent cloud-free image within 30 days
def get_most_recent_image_with_cloud_check(feature):
    """Get the most recent Sentinel-2 image with cloud coverage < 5%."""
    date = ee.Date(feature.get("date"))
    start_date = date.advance(-30, "day")
    end_date = date

    # Define a 2000m circular buffer around the point
    point = feature.geometry()
    buffer = point.buffer(2000)

    # Filter and link the collections
    linked_image = filter_collections(feature, buffer, start_date, end_date)

    if linked_image is None:
        return feature.set({"image_id": None, "image_date": None, "cloud_percentage": None})

    # Calculate cloud percentage for the linked image
    cloud_percentage = calculate_cloud_percentage(linked_image, buffer)

    # Only keep the image if cloud percentage is less than 5%
    cloud_free_image = ee.Algorithms.If(
        cloud_percentage.lt(5),
        linked_image.select("B.*").updateMask(linked_image.select("cloud_score").lt(50)),
        None,
    )

    return feature.set(
        {
            "image_id": ee.Algorithms.If(cloud_free_image, linked_image.get("system:id"), None),
            "image_date": ee.Algorithms.If(
                cloud_free_image, linked_image.get("system:time_start"), None
            ),
            "cloud_percentage": cloud_percentage,
        }
    )


# Main function to process all points
def process_points(csv_file_path, output_csv_path):
    """Process all points and save results to a CSV."""
    points_fc = load_points_from_csv(csv_file_path)

    # Apply the function to all points
    results_fc = points_fc.map(get_most_recent_image_with_cloud_check)

    # Convert the results FeatureCollection to a pandas DataFrame
    results_list = results_fc.getInfo()["features"]
    results = [
        {
            "latitude": feature["geometry"]["coordinates"][1],
            "longitude": feature["geometry"]["coordinates"][0],
            "date": feature["properties"]["date"],
            "image_id": feature["properties"].get("image_id", None),
            "image_date": feature["properties"].get("image_date", None),
            "cloud_percentage": feature["properties"].get("cloud_percentage", None),
        }
        for feature in results_list
    ]

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)

    print(f"Results saved to {output_csv_path}")


# Example usage:
csv_file_path = "your_file.csv"  # Path to your CSV file with latitude, longitude, and date
output_csv_path = "most_recent_images_with_less_than_5_percent_cloud.csv"
process_points(csv_file_path, output_csv_path)
