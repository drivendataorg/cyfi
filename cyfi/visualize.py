from pathlib import Path

import gradio as gr
import geopy.distance as distance
import pandas as pd
import rioxarray
import planetary_computer as pc
import matplotlib.pyplot as plt


def get_bounding_box(latitude: float, longitude: float, meters_window: int):
    """
    Given a latitude, longitude, and buffer in meters, returns a bounding
    box around the point with the buffer on the left, right, top, and bottom.

    Returns a list of [minx, miny, maxx, maxy]
    """
    distance_search = distance.distance(meters=meters_window)

    # calculate the lat/long bounds based on ground distance
    # bearings are cardinal directions to move (south, west, north, and east)
    min_lat = distance_search.destination((latitude, longitude), bearing=180)[0]
    min_long = distance_search.destination((latitude, longitude), bearing=270)[1]
    max_lat = distance_search.destination((latitude, longitude), bearing=0)[0]
    max_long = distance_search.destination((latitude, longitude), bearing=90)[1]

    return [min_long, min_lat, max_long, max_lat]


def plot_image(
    num,
    # df,
    # sample_id=None,
    # degree_buffer=0.01,
    # severity_pred_col="severity_pred",
    # severity_actual_col="severity",
):
    # if sample_id is not None:
        # sample = df[df.sample_id == sample_id]
    # else:
        # sample = df.sample(1)

    sample_crs="EPSG:4326"

    DIR = Path("experiments/results/2000_30days_filter_cloud_water_winsorize_labels")
    preds = pd.read_csv(DIR / "preds.csv")
    meta = pd.read_csv(DIR / "sentinel_metadata_test.csv")

    df = preds.merge(meta, on="sample_id")
    sample = df.iloc[num]
    sample = sample.squeeze()

    distance_search = distance.distance(meters=2000)

    # calculate the lat/long bounds based on ground distance
    # bearings are cardinal directions to move (south, west, north, and east)
    (minx, miny, maxx, maxy) = get_bounding_box(
        sample.latitude,
        sample.longitude,
        2000,
    )

    # crop image and reproject
    cropped_img_array = (
        rioxarray.open_rasterio(pc.sign(sample.visual_href))
        .rio.clip_box(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            crs=sample_crs,
        )
        .rio.reproject(sample_crs)
    )

    fig, ax = plt.subplots()

    # cropped_img_array.plot.imshow(ax=ax)
    # ax.plot(sample.longitude, sample.latitude, "ro", markersize=4)

    return cropped_img_array.to_numpy().transpose(1, 2, 0), sample.density_cells_per_ml
    # plt.title(
    #     f"Predicted cyanobacteria density: {sample[severity_pred_col]:,.0f}\nActual advisory: {sample[severity_actual_col]}"
    # )

    # print(f"Sample ID: {sample.sample_id}")
    # print(f"Date: {sample.date}")
    # print(
    #     f"Number of days before sampling date image is taken: {sample.days_before_sample}"
    # )
    # print(f"Water body: {sample.Water_Body_Name}")
    # print(f"Advisory detail description: {sample.Advisory_Detail_Description}")


# demo = gr.Interface(
#     fn=plot_image,
#     inputs=gr.Slider(0, 100),
#     outputs=["image", gr.Label()],
# )
# demo.launch()



import plotly.graph_objects as go



def make_map():
    # set column order
    DIR = Path("experiments/results/2000_30days_filter_cloud_water_winsorize_labels")
    preds = pd.read_csv(DIR / "preds.csv")
    meta = pd.read_csv(DIR / "sentinel_metadata_test.csv")
    df = preds.merge(meta, on="sample_id")[["date", "latitude", "longitude", "density_cells_per_ml", "severity"]]

    fig = go.Figure(go.Scattermapbox(
                customdata=df,
                lat=df['latitude'].tolist(),
                lon=df['longitude'].tolist(),
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=6
                ),
                hoverinfo="text",
                hovertemplate='<b>Date</b>: %{customdata[0]}<br><b>Latitude</b>: %{customdata[1]}<br><b>Longitude</b>: %{customdata[2]}<br><b>Predicted density</b>: %{customdata[3]}<br><b>Predicted severity</b>: %{customdata[4]}'
            ))

    sample = preds.iloc[1].squeeze()

    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=sample.latitude,
                lon=sample.longitude,
            ),
            pitch=0,
            zoom=9
        ),
    )
    return fig

with gr.Blocks() as demo:
    map = gr.Plot()
    demo.load(make_map, [], map)

demo.launch()