from pathlib import Path
import tempfile

import gradio as gr
import geopy.distance as distance
import pandas as pd
import rioxarray
import planetary_computer as pc
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import typer


cyfi_examples_dir = Path(tempfile.gettempdir()) / "cyfi_explorer"
cyfi_examples_dir.mkdir(exist_ok=True, parents=True)


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
    sample_id,
    date,
    lat,
    lon,
    density,
):
    df = pd.read_csv(cyfi_examples_dir / "log.csv")
    sample = df.set_index("sample_id").loc[sample_id].squeeze()

    distance_search = distance.distance(meters=2000)
    sample_crs = "EPSG:4326"

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

    return (
        cropped_img_array.to_numpy().transpose(1, 2, 0),
        sample.density_cells_per_ml,
        sample.severity,
        sample.date,
        sample.latitude,
        sample.longitude,
    )


def make_map():
    # set column order
    df = pd.read_csv(cyfi_examples_dir / "log.csv")
    map_df = df[["date", "latitude", "longitude", "density_cells_per_ml", "severity"]]

    fig = go.Figure(
        go.Scattermapbox(
            customdata=map_df,
            lat=map_df["latitude"].tolist(),
            lon=map_df["longitude"].tolist(),
            mode="markers",
            marker=go.scattermapbox.Marker(size=6),
            hoverinfo="text",
            hovertemplate="<b>Date</b>: %{customdata[0]}<br><b>Latitude</b>: %{customdata[1]}<br><b>Longitude</b>: %{customdata[2]}<br><b>Predicted density</b>: %{customdata[3]}<br><b>Predicted severity</b>: %{customdata[4]}",
        )
    )

    sample = map_df.iloc[1].squeeze()

    fig.update_layout(
        mapbox_style="carto-positron",
        hovermode="closest",
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=sample.latitude,
                lon=sample.longitude,
            ),
            pitch=0,
            zoom=9,
        ),
    )
    return fig


def visualize(
    output_directory: Path = typer.Argument(
        Path.cwd(),
        help="CyFi output directory containing preds.csv and sentinel_metadata.csv from a prior prediction run.",
    )
):
    """Launch CyFi Explorer to see Sentinel-2 imagery alongside predictions."""

    # merge preds and sentinel metadata into single csv and write out to temp directory
    preds = pd.read_csv(output_directory / "preds.csv")
    preds = preds[preds.severity.notnull()]
    preds["density_cells_per_ml"] = preds.density_cells_per_ml.astype(int)
    meta = pd.read_csv(output_directory / "sentinel_metadata.csv")
    df = preds.merge(meta, on="sample_id")

    # save out as log.csv (expected filename for gradio examples)
    df.to_csv(cyfi_examples_dir / "log.csv", index=False)

    with gr.Blocks() as demo:
        gr.Markdown("CyFi Explorer")

        with gr.Tab("Imagery Explorer"):
            with gr.Row():
                image = gr.Image(label="Sentinel-2 imagery")
                with gr.Column():
                    density = gr.Textbox(label="Estimated cyanobacteria density (cells/ml)")
                    severity = gr.Textbox(label="Severity level")
                    date = gr.Textbox(label="date")
                    lat = gr.Number(label="latitude")
                    lon = gr.Number(label="longitude")

            input_sample = gr.Textbox(label="sample_id", visible=False)
            input_date = gr.Textbox(label="date", visible=False)
            input_lat = gr.Number(label="latitude", visible=False)
            input_lon = gr.Number(label="longitude", visible=False)
            input_density = gr.Textbox(label="density", visible=False)

            gr.Markdown(
                "Click on a row to see the Sentinel-2 imagery used to generate the cyanobacteria estimate."
            )
            gr.Examples(
                examples=str(cyfi_examples_dir),
                inputs=[input_sample, input_date, input_lat, input_lon, input_density],
                label="Sample points",
                fn=plot_image,
                run_on_click=True,
                outputs=[image, density, severity, date, lat, lon],
            )

        with gr.Tab("Map Explorer"):
            map = gr.Plot()
            demo.load(make_map, [], map)

    demo.launch()
