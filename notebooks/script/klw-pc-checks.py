#!/usr/bin/env python
# coding: utf-8

# Issue [19](https://github.com/drivendataorg/cyanobacteria-prediction/issues/19)
# 
# Check that our planetary computer results are correct

import json

from cloudpathlib import AnyPath
import pandas as pd
from tqdm import tqdm

from cyano.config import FeaturesConfig
from cyano.data.utils import add_unique_identifier
from cyano.data.satellite_data import (
    select_items,
    search_planetary_computer,
    get_items_metadata
)


# ### Load data

test = pd.read_csv(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/competition/test.csv"
    )
)
test = add_unique_identifier(test)
test.shape


test.head(2)


# Take a subset to run PC search on
samples = test.sample(n=100, random_state=3) #(n=100, random_state=3)
samples.region.value_counts()


# ### Define functions

# use the same geographic window as our original full PC search
config = FeaturesConfig(pc_meters_search_window=10000)
config.pc_meters_search_window, config.pc_days_search_window


# Note that our original PC search includes a time buffer both before *and* after the sample, while our current package code only adds a time buffer *before* the sample. 
# 
# Rather than comparing the raw results of the PC search, we'll also complete the next step of identifying relevant satellite data (`identify_satellite_data`) to apply the same date range to both, and *then* we'll compare results.

def identify_satellite_data_mod(samples: pd.DataFrame,
                           config: FeaturesConfig,
                            candidate_sentinel_meta: pd.DataFrame,
                            sample_item_map: dict
                           ) -> pd.DataFrame:
    """Modified version of 
    cyano.data.utils.satellite_data.identify_satellite_data
    that allows us to change the item map and candidate metadata
    """
    selected_meta = []
    for sample in tqdm(samples.itertuples(), total=len(samples)):
        sample_item_ids = sample_item_map[sample.Index]['sentinel_item_ids']
        if len(sample_item_ids) == 0:
            continue

        sample_items_meta = candidate_sentinel_meta[
            candidate_sentinel_meta.item_id.isin(sample_item_ids)
        ].copy()
        selected_ids = select_items(sample_items_meta, sample.date, config)

        sample_items_meta = sample_items_meta[
            sample_items_meta.item_id.isin(selected_ids)
        ]
        sample_items_meta['sample_id'] = sample.Index

        selected_meta.append(sample_items_meta)

    selected_meta = pd.concat(selected_meta).reset_index(drop=True)
    
    return selected_meta


# ## Search planetary computer

regenerated_candidate_meta = []
regenerated_map = {}

for sample in tqdm(samples.itertuples(), total=len(samples)):
    # Search planetary computer
    search_results = search_planetary_computer(
        sample.date,
        sample.latitude,
        sample.longitude,
        collections=["sentinel-2-l2a"],
        days_search_window=config.pc_days_search_window,
        meters_search_window=config.pc_meters_search_window,
    )
    
    # Get satellite metadata
    sample_items_meta = get_items_metadata(
        search_results, sample.latitude, sample.longitude, config
    )
    
    regenerated_map[sample.Index]  = {
        "sentinel_item_ids": sample_items_meta.item_id.tolist()
        if len(sample_items_meta) > 0
        else []
    }
    regenerated_candidate_meta.append(sample_items_meta)
    
regenerated_candidate_meta = (
    pd.concat(regenerated_candidate_meta).groupby("item_id", as_index=False)
    .first().reset_index(drop=True)
)
print(f"Generated metadata for {regenerated_candidate_meta.shape[0]:,} Sentinel item candidates")


regenerated_meta = identify_satellite_data_mod(
    samples = samples,
    config = config,
    candidate_sentinel_meta = regenerated_candidate_meta,
    sample_item_map = regenerated_map
)
regenerated_meta.shape


regenerated_meta.head(2)


# ## Load results from past PC search

pc_results_dir = (
    AnyPath("s3://drivendata-competition-nasa-cyanobacteria")
    / "data/interim/full_pc_search"
)
loaded_candidate_meta = pd.read_csv(
    pc_results_dir / "sentinel_metadata.csv"
)
assert loaded_candidate_meta.item_id.is_unique
print(
    f"Loaded {loaded_candidate_meta.shape[0]:,} rows of Sentinel candidate metadata"
)

with open(pc_results_dir / "sample_item_map.json", "r") as fp:
    loaded_map = json.load(fp)
print(f"Loaded sample item map for {len(loaded_map):,} samples")


loaded_meta = identify_satellite_data_mod(
    samples = samples,
    config=config,
    candidate_sentinel_meta = loaded_candidate_meta,
    sample_item_map = loaded_map
)
loaded_meta.shape


loaded_meta.head(2)


# ## Compare new results with past results

loaded_meta.shape, regenerated_meta.shape


include_cols = ['item_id', 'sample_id', 
                'datetime', 'platform', 'min_long', 'max_long', 'min_lat',
       'max_lat', 'eo:cloud_cover', 's2:nodata_pixel_percentage',
       'days_before_sample']


compare_loaded_meta = (
    loaded_meta.sort_values(by=['sample_id', 'item_id'])[include_cols].round(5)
)
compare_regenerated_meta = (
    regenerated_meta.sort_values(by=['sample_id', 'item_id'])[include_cols].round(5)
)


(compare_loaded_meta == compare_regenerated_meta).all()







