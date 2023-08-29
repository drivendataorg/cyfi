#!/usr/bin/env python
# coding: utf-8

# Look into cases we get very wrong
# 
# We'll look at the predictions from our best model - third place sentinel and land cover features, predicting log density, with folds and a higher number of boosted rounds.

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from cloudpathlib import AnyPath
import pandas as pd

from cyano.data.utils import add_unique_identifier


# load all metadata for reference
meta = pd.read_csv(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/data/final/combined_final_release.csv"
    )
)
meta.head(3)


best_exp_dir = AnyPath(
    "s3://drivendata-competition-nasa-cyanobacteria/experiments/results/third_sentinel_with_folds"
)


# load best predictions
preds = pd.read_csv(best_exp_dir / 'preds.csv', index_col=0)
preds.head()


# load actual
true = pd.read_csv(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/competition/test.csv"
    )
)
true = add_unique_identifier(true)
true.head(2)


true['pred_severity']  = preds.loc[true.index].severity
true['pred_log_density'] = preds.loc[true.index].log_density


# check samples with actual severity 1 but predicted severity 4
check = true[(true.severity == 1) & (true.pred_severity == 4)]
check.shape


# ## Check metadata
# 
# What region are these from? What providers? 

# almost all are in the south
check.region.value_counts()


# almost all are north carolina
# could these be routine sites with inaccurate gps data?
check.data_provider.value_counts()


# Check some lat / longs
check[check.region == 'south'][['latitude', 'longitude']].drop_duplicates().head()


# pull in original NC data
nc_raw = pd.read_excel(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/data/raw/nc/New Use This NCDWR phyto 2013-2021 All Data.xlsx"
    ),
    sheet_name="Cyanobacteria Density"
)
nc_raw.shape 


nc_raw = nc_raw.rename(columns={'Lat':'latitude', 'Long':'longitude'})
nc_raw['date'] = pd.to_datetime(nc_raw.Date)
nc_raw.head(2)


raw_subset = check[['latitude', 'longitude']].merge(nc_raw, how='inner', on=['latitude', 'longitude'])
raw_subset.shape


raw_subset.Waterbody.value_counts(dropna=False)


raw_subset.groupby(['Waterbody', raw_subset.date.dt.year]).size().sort_index()


raw_subset.StationDesc.value_counts(dropna=False)


# In an email from Elizabeth Fensin in NC:
# > I’m also surprised I sent data from the Cape Fear River 2020-2021 study since it was unusual.  We usually assign one taxonomist to particular waterbodies to ensure uniform results.  In the Cape Fear study, one taxonomist did the 2020 and another did the 2021 samples.  This created confusing results and some samples were recounted.
# 
# She also noted that routinely monitored sites were more likely to have inaccurate GPS data. Pamlico is one of their ambient sites.
# 
# One option is to drop the Cape Fear River data altogether.

# how much of our final data is from cape fear river?
(cape_fear_coords := nc_raw[nc_raw.Waterbody == 'Cape Fear River'][['latitude', 'longitude']].drop_duplicates())


cape_fear_final = cape_fear_coords.merge(meta, on=['latitude', 'longitude'], how='inner')
cape_fear_final.shape


cape_fear_final.groupby(['latitude', 'longitude', 'data_provider']).size()


pd.concat(
    [final_from_cape_fear.distance_to_water_m.describe().rename('cape_fear'),
     meta.distance_to_water_m.describe().rename('all_data')
    ], axis=1
)


# **Takeaway**
# 
# I recommend removing the Cape Fear data from NC from our training set. It was flagged by our NC contact as being potentially confusing, and the odd behavior of the model on these cases could be related to inaccurate lat / longs or inaccurate ground truth measurements. We can also see that cape fear data points tend to be farther from water than the rest of the data

# would we also want to drop pamlico? how much data is from pamlico?
# pamlico has a lot more different lat / longs
pamlico_coords = nc_raw[nc_raw.Waterbody == 'Pamlico River'][['latitude', 'longitude']].drop_duplicates()

# how much of our data is pamlico?
pamlico_final = pamlico_coords.merge(meta, on=['latitude', 'longitude'], how='inner')
pamlico_final.shape


pamlico_final.data_provider.value_counts()


# The majority of the Pamlico data (630 samples) has severity 1 but is predicted severity 4. It's very possible that similarly, this is just because the data is noisy.

pd.concat(
    [cape_fear_final.distance_to_water_m.describe().rename('cape_fear'),
        pamlico_final.distance_to_water_m.describe().rename('pamlico'),
     meta.distance_to_water_m.describe().rename('all_data')
    ], axis=1
)


# ## Example images












