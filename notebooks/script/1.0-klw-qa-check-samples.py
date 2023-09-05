#!/usr/bin/env python
# coding: utf-8

# Generate list of samples to use for QA checks for temporal consistency

get_ipython().run_line_magic('load_ext', 'lab_black')


from datetime import timedelta

from cloudpathlib import AnyPath
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

from cyano.data.utils import add_unique_identifier


STATES_SHAPEFILE = gpd.GeoDataFrame.from_file(
    "../../competition-nasa-cyanobacteria/data/raw/cb_2018_us_state_500k/cb_2018_us_state_500k.shp"
)


def plot_map(df, markersize=5, **kwargs):
    _, ax = plt.subplots()

    STATES_SHAPEFILE.plot(color="ghostwhite", edgecolor="lightgray", ax=ax)

    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.plot(ax=ax, markersize=markersize)

    ax.set_xlim([-125, -65])
    ax.set_ylim([25, 50])

    return ax


df = pd.read_csv(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/data/final/combined_final_release.csv"
    )
)
df["date"] = pd.to_datetime(df.date)
df.head(3)


# ### Select subset of samples

# subset to points in water bodies
subset = df[df.distance_to_water_m == 0]
print(f"{subset.shape[0]} samples in water")

# only have max 1 example at each location
subset = subset.sample(frac=1, random_state=3)
subset = subset.groupby(["latitude", "longitude"], as_index=False).first()
print(f"{subset.shape[0]} unique locations")

subset = subset.sample(n=5, random_state=3)
subset.head(1)


subset.region.value_counts()


subset.severity.value_counts().sort_index()


# see on a map where these samples are
plot_map(subset)
plt.show()


# ### Add temporal checks

predict_df = []

for sample in subset.itertuples():
    predict_df.append(
        pd.DataFrame(
            {
                "latitude": sample.latitude,
                "longitude": sample.longitude,
                # 4 week range for each sample
                # 7 days between samples
                "date": pd.date_range(
                    start=sample.date - timedelta(days=14),
                    end=sample.date + timedelta(days=14),
                    freq="7d",
                    inclusive="both",
                ),
            }
        )
    )

predict_df = pd.concat(predict_df)
predict_df.shape


predict_df


save_to = AnyPath(
    "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/temporal_check_samples.csv"
)

with save_to.open("w+") as f:
    predict_df.to_csv(f, index=False)

print(f"Samples for prediction saved to {save_to}")


from pathlib import Path


predict_df.to_csv("../temporal_check_samples.csv", index=False)


# `python cyano/cli.py predict temporal_check_samples.csv --output-path temporal_check_preds.csv`












