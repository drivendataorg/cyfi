#!/usr/bin/env python
# coding: utf-8

# Generate list of samples to use for QA checks for temporal consistency

get_ipython().run_line_magic('load_ext', 'lab_black')


from datetime import timedelta

from cloudpathlib import AnyPath
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from shapely.geometry import Point

from cyano.data.utils import add_unique_identifier


EXPERIMENT_DIR = Path("../experiments/results/temporal_qa_checks/")
EXPERIMENT_DIR.mkdir(exist_ok=True, parents=True)


df = pd.read_csv(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/competition/test.csv"
    )
)
df["date"] = pd.to_datetime(df.date)
df.head(3)


# ### Select subset of samples

# subset to points in water bodies
subset = df[df.distance_to_water_m == 0]
print(f"Narrowed to {subset.shape[0]:,} samples in water")

# only have max 1 example at each location
subset = subset.sample(frac=1, random_state=3)
subset = subset.groupby(["latitude", "longitude"], as_index=False).first()
print(f"Narrowed to {subset.shape[0]:,} unique locations")

# take some from each region
subset = pd.concat(
    [
        subset[subset.region == region].sample(n=7, random_state=2)
        for region in subset.region.unique()
    ]
)
subset = add_unique_identifier(subset)

subset.head(3)


# see on a map where these samples are
_, ax = plt.subplots()

STATES_SHAPEFILE = gpd.GeoDataFrame.from_file(
    "../../competition-nasa-cyanobacteria/data/raw/cb_2018_us_state_500k/cb_2018_us_state_500k.shp"
)
STATES_SHAPEFILE.plot(color="ghostwhite", edgecolor="lightgray", ax=ax)

geometry = [Point(xy) for xy in zip(subset.longitude, subset.latitude)]
gdf = gpd.GeoDataFrame(subset, geometry=geometry)
gdf.plot(ax=ax, markersize=5)

ax.set_xlim([-125, -65])
ax.set_ylim([25, 50])
plt.show()


subset.severity.value_counts().sort_index()


# ### Add temporal checks
# 
# Add additional rows around each sample spanning 4 weeks

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
                "region": sample.region,
                "original_sample_id": sample.Index,
            }
        )
    )

predict_df = pd.concat(predict_df)
predict_df.shape


predict_df


save_to = EXPERIMENT_DIR / "samples.csv"
save_to.parent.mkdir(exist_ok=True, parents=True)

predict_df.to_csv(save_to, index=False)

print(f"Samples for prediction saved to {save_to}")


# To generate predictions:
# 
# `python cyano/cli.py predict experiments/results/temporal_qa_checks/samples.csv --output-path experiments/results/temporal_qa_checks/preds.csv`












