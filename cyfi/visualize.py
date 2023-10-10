import io
from pathlib import Path
import tempfile
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import planetary_computer as pc
import plotly.graph_objects as go
import typer
import rioxarray

from cyfi.data.satellite_data import get_bounding_box


def visualize(
    output_directory: Path = typer.Argument(
        Path.cwd(),
        exists=True,
        help="CyFi output directory containing preds.csv and sentinel_metadata.csv from a prior prediction run.",
    )
):
    """Launch CyFi Explorer to see Sentinel-2 imagery alongside predictions."""

    if not (output_directory / "preds.csv").exists():
        raise ValueError(
            f"Output directory {output_directory} does not contain a preds.csv file. Be sure to run `cyfi predict` first and make sure to use the flag `--keep-metadata`."
        )

    if not (output_directory / "sentinel_metadata.csv").exists():
        raise ValueError(
            f"Output directory {output_directory} does not contain a sentinel_metadata.csv file. Make sure to specify `--keep-metadata` when running `cyfi predict`."
        )

    # merge preds and sentinel metadata into single csv and write out to temp directory
    preds = pd.read_csv(output_directory / "preds.csv")
    preds = preds[preds.severity.notnull()].copy()
    preds["density_cells_per_ml"] = preds.density_cells_per_ml.astype(int)
    meta = pd.read_csv(output_directory / "sentinel_metadata.csv")
    df = preds.merge(meta, on="sample_id")

    # Add date of satellite imagery
    df["imagery_date"] = (
        pd.to_datetime(df.date) - pd.to_timedelta(df.days_before_sample, unit="d")
    ).dt.date

    cyfi_examples_dir = Path(tempfile.gettempdir()) / "cyfi_explorer"
    cyfi_examples_dir.mkdir(exist_ok=True, parents=True)

    # save out as log.csv (expected filename for gradio examples)
    df.to_csv(cyfi_examples_dir / "log.csv")

    def plot_image(
        evt: gr.SelectData,
    ):
        if evt.index[1] != 0:
            raise gr.Error("Please click on the sample_id for the row")

        sample = df[df.sample_id == evt.value].squeeze()
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

        # plot imagery with point on it
        fig, ax = plt.subplots(frameon=False)
        cropped_img_array.plot.imshow(ax=ax)
        ax.plot(sample.longitude, sample.latitude, "ro", markersize=4, markerfacecolor="None")
        ax.axis("equal")
        ax.set_axis_off()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        # convert to PIL image for rendering
        img_buf = io.BytesIO()
        ax.figure.savefig(img_buf, format="png", bbox_inches="tight")
        im = Image.open(img_buf)

        return (
            im,
            f"{sample.density_cells_per_ml:,.0f}",
            sample.severity,
            f"({sample.longitude}, {sample.latitude})",
            sample.date,
            sample.imagery_date,
            sample.item_id,
        )

    def make_map():
        # set column order
        map_df = df[
            ["sample_id", "date", "latitude", "longitude", "density_cells_per_ml", "severity"]
        ].copy()

        color_map = {"low": "teal", "moderate": "gold", "high": "red"}
        map_df["color"] = map_df.severity.map(color_map)
        fig = go.Figure(
            go.Scattermapbox(
                customdata=map_df,
                lat=map_df["latitude"].tolist(),
                lon=map_df["longitude"].tolist(),
                mode="markers",
                marker=dict(size=6, color=map_df.color),
                hoverinfo="text",
                hovertemplate="<b>Sample ID</b>: %{customdata[0]}<br>Date</b>: %{customdata[1]}<br><b>Latitude</b>: %{customdata[2]}<br><b>Longitude</b>: %{customdata[3]}<br><b>Predicted density</b>: %{customdata[4]:,.0f}<br><b>Predicted severity</b>: %{customdata[5]}",
                name="",
            )
        )

        fig.update_layout(
            mapbox_style="carto-positron",
            hovermode="closest",
            mapbox=dict(
                bearing=0,
                center=go.layout.mapbox.Center(
                    lat=37.0902,
                    lon=-95.7129,
                ),
                pitch=0,
                zoom=3,
            ),
        )
        return fig

    with gr.Blocks(title="CyFi Explorer") as demo:
        with gr.Row():
            gr.Markdown(
                """
                # CyFi Explorer

                Click on the `sample_id` for a row in the table to see the Sentinel-2 imagery used to generate the cyanobacteria estimate.
                """
            )
        with gr.Row():
            gr.Markdown(
                """
                ### CyFi estimates
                """
            )
        with gr.Row():
            data = gr.DataFrame(
                df[
                    [
                        "sample_id",
                        "date",
                        "latitude",
                        "longitude",
                        "density_cells_per_ml",
                        "severity",
                    ]
                ].style.format({"density_cells_per_ml": "{:,.0f}"}),
                height=200,
            )

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    """
                    ### Sentinel-2 Imagery
                    """
                )

                image = gr.Image(type="pil", label="Sentinel-2 imagery", container=False)

            with gr.Column(scale=2):
                gr.Markdown(
                    """
                    ### Details on the selected sample
                    """
                )
                density = gr.Textbox(label="Estimated cyanobacteria density (cells/ml)")
                severity = gr.Textbox(label="Estimated severity level")
                loc = gr.Textbox(label="Location")
                date = gr.Textbox(label="Sampling date")
                days_before_sample = gr.Textbox(label="Satellite imagery date")
                tile = gr.Textbox(label="Sentinel-2 tile")

                data.select(
                    plot_image,
                    None,
                    [image, density, severity, loc, date, days_before_sample, tile],
                )

        with gr.Row():
            gr.Markdown(
                """
                ### Map of all estimates
                """
            )

        with gr.Row():
            map = gr.Plot()
            demo.load(make_map, [], map)

    demo.launch()
