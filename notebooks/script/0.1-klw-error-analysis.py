#!/usr/bin/env python
# coding: utf-8

# Look into cases we get very wrong, ie. severity 1 but predicted severity 4.
# 
# We'll look at the predictions from our best model - third place sentinel and land cover features, predicting log density, with folds and a higher number of boosted rounds.

get_ipython().run_line_magic('load_ext', 'lab_black')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from cloudpathlib import AnyPath
import geopandas as gpd
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from shapely.geometry import Point

from cyano.data.utils import add_unique_identifier
from cyano.settings import REPO_ROOT


# ## Load data

best_exp_dir = AnyPath(
    "s3://drivendata-competition-nasa-cyanobacteria/experiments/results/third_sentinel_with_folds"
)
cache_dir = REPO_ROOT.parent / "experiments/cache"


# load all metadata for reference
meta = pd.read_csv(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/data/final/combined_final_release.csv"
    )
)
meta.head(3)


# load our satellite metadata
sat_meta = pd.read_csv(cache_dir / "satellite_metadata_test.csv")
sat_meta.shape


sat_meta.head(2)


# load best predictions
preds = pd.read_csv(best_exp_dir / "preds.csv", index_col=0)
preds.head()


# load actual
true = pd.read_csv(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/competition/test.csv"
    )
)
true = add_unique_identifier(true)
true.head(2)


true["pred_severity"] = preds.loc[true.index].severity
true["pred_log_density"] = preds.loc[true.index].log_density


# check samples with actual severity 1 but predicted severity 4
check = true[(true.severity == 1) & (true.pred_severity == 4)]
check.shape


# ## Sample metadata
# 
# What region are these from? What providers? 

# almost all are in the south
check.region.value_counts()


# almost all are north carolina
# could these be routine sites with inaccurate gps data?
check.data_provider.value_counts()


# pull in original NC data
logger.info("Loading raw NC data")
nc_raw = pd.read_excel(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/data/raw/nc/New Use This NCDWR phyto 2013-2021 All Data.xlsx"
    ),
    sheet_name="Cyanobacteria Density",
)
logger.info(f"Loaded {nc_raw.shape[0]:,} rows of raw NC data")

nc_raw = nc_raw.rename(columns={"Lat": "latitude", "Long": "longitude"})
nc_raw["date"] = pd.to_datetime(nc_raw.Date)
nc_raw.head(2)


# Get the subset of NC data that has low severity, high predicted severity
raw_subset = check[["latitude", "longitude"]].merge(
    nc_raw, how="inner", on=["latitude", "longitude"]
)
logger.info(
    f"{raw_subset.shape[0]:,} rows of raw NC are from locations with low severity, high predicted severity"
)


# What waterbodies is this subset from?
raw_subset.Waterbody.value_counts(dropna=False)


raw_subset.groupby(["Waterbody", raw_subset.date.dt.year]).size().sort_index()


# In an email from Elizabeth Fensin in NC:
# > I’m also surprised I sent data from the Cape Fear River 2020-2021 study since it was unusual.  We usually assign one taxonomist to particular waterbodies to ensure uniform results.  In the Cape Fear study, one taxonomist did the 2020 and another did the 2021 samples.  This created confusing results and some samples were recounted.
# 
# She also noted that routinely monitored sites were more likely to have inaccurate GPS data. Pamlico is one of their ambient sites.
# 
# One option is to drop both the Cape Fear and Pamlico River data, since it seems suspect.

# how much of our final data is from cape fear river?
cape_fear_coords = nc_raw[nc_raw.Waterbody == "Cape Fear River"][
    ["latitude", "longitude"]
].drop_duplicates()
logger.info(f"Cape fear has {cape_fear_coords.shape[0]} locations")

cape_fear_final = cape_fear_coords.merge(
    meta, on=["latitude", "longitude"], how="inner"
)
assert (
    cape_fear_final.data_provider
    == "N.C. Division of Water Resources N.C. Department of Environmental Quality"
).all()
logger.info(f"{cape_fear_final.shape[0]} final data points are from cape fear")


# would we also want to drop pamlico? how much data is from pamlico?
pamlico_coords = nc_raw[nc_raw.Waterbody == "Pamlico River"][
    ["latitude", "longitude"]
].drop_duplicates()
logger.info(f"Pamlico river fear has {pamlico_coords.shape[0]} locations")

pamlico_final = pamlico_coords.merge(meta, on=["latitude", "longitude"], how="inner")
assert (
    pamlico_final.data_provider
    == "N.C. Division of Water Resources N.C. Department of Environmental Quality"
).all()
logger.info(f"{pamlico_final.shape[0]} final data points are from pamlico river")


states_shapefile = gpd.GeoDataFrame.from_file(
    "s3://drivendata-competition-nasa-cyanobacteria/data/raw/cb_2018_us_state_500k/cb_2018_us_state_500k.shp"
)


ax = states_shapefile.loc[:, "geometry"].plot(color="ghostwhite", edgecolor="lightgray")

geo = [Point(xy) for xy in zip(pamlico_coords.longitude, pamlico_coords.latitude)]
gdf = gpd.GeoDataFrame(pamlico_coords, geometry=geo)
gdf.plot(ax=ax, markersize=3, label="pamlico")

geo = [Point(xy) for xy in zip(cape_fear_coords.longitude, cape_fear_coords.latitude)]
gdf = gpd.GeoDataFrame(cape_fear_coords, geometry=geo)
gdf.plot(ax=ax, markersize=3, color="darkorange", label="cape fear")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_xlim([-84, -75])
ax.set_ylim([34, 37])
plt.legend()


# The pamlico river points are generally very far from the pamlico river. Besides the few farther east, the rest of the points are all much farther south than the tar or the neuse river. They are a little west of the Jordan Lake, so not in either the Tar-Pamlico or Neuse water basin. The Neuse river is a separate waterbody in the data, as is Jordan Lake.
# 
# ![image.png](attachment:b4e25680-604f-4268-8bea-ee0567765128.png)
# 
# The cape fear points are generally consistent with where the cape fear river is

pd.concat(
    [
        cape_fear_final.distance_to_water_m.describe().rename("cape_fear"),
        pamlico_final.distance_to_water_m.describe().rename("pamlico"),
        meta.distance_to_water_m.describe().rename("all_data"),
    ],
    axis=1,
)


# **Takeaway**
# 
# The odd behavior of the model on these cases could be related to inaccurate lat / longs or inaccurate ground truth measurements. 
# 
# - The cape fear data in particular was flagged by our NC contact as being potentially confusing
# - Ambient sites like pamlico were flagged for potentially inaccurate GPS locations. The points in the NC dataset that say they are for the pamlico river are generally not anywhere near the pamlico river
# - We can also see that both cape fear and pamlico data points tend to be farther from water than the rest of the data, with pamlico being worse
# - There are 82 final data points from cape fear, and 683 from pamlico
# 
# **I recommend removing both pamlico river and cape fear river data from our train / test sets**

# ## Satellite metadata
# 
# Based on the satellite metadata, are the images for the subset of samples with low severity but high predicted severity different from other images?

# get one row per image,
# including indication of whether that image is used for our subset to check
sat_meta["low_actual_high_pred"] = sat_meta.sample_id.isin(check.index).astype(str)

sat_meta = sat_meta.drop(columns=["sample_id"]).groupby("item_id").max()
sat_meta["low_actual_high_pred"].value_counts()


show_cols = [
    "eo:cloud_cover",
    "s2:high_proba_clouds_percentage",
    "s2:medium_proba_clouds_percentage",
    "s2:thin_cirrus_percentage",
    "s2:water_percentage",
    "s2:nodata_pixel_percentage",
    "s2:dark_features_percentage",
]

fig, axes = plt.subplots(len(show_cols), 1, sharey=True, figsize=(6, 7))

for ax, col in zip(axes, show_cols):
    sns.boxplot(data=sat_meta, x=col, y="low_actual_high_pred", ax=ax)
    ax.set_ylabel("")

axes[int(len(show_cols) / 2)].set_ylabel(
    "Image is used for low severity, high prediction sample"
)
plt.tight_layout()


# **Takeaway**
# 
# Based on the satellite metadata, the images for these samples are slightly more cloudy, but not drastically so. They also have a wide range of labeled cloud cover, and are not all highly clouded. They also don't have significantly more no data pixels, more dark pixels, or fewer water pixels. They do tend to have slightly lower percentage of water pixels.
# 
# The similarity in satellite metadata suggests that the oddities with these samples are more likely a result of upstream data issues.









