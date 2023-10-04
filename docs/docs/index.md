# CyFi: Cyanobacteria Finder

CyFi is a command line tool that uses satellite imagery and machine learning to estimate cyanobacteria levels, one type of harmful algal bloom. The goal of CyFi is to help water quality managers better allocate resources for in situ sampling, and make more informed decisions around public health warnings for critical resources like drinking water reservoirs. Ultimately, more accurate and more timely detection of algal blooms helps keep both the human and marine life that rely on these water bodies safe and healthy.

------

## Quickstart

### Install

Install CyFi with pip:

```
pip install cyfi
```

For detailed instructions for those installing python for the first time, see the [Installation](https://cyfi.drivendata.org/stable/installation) docs.

### Generate batch predictions

Generate batch predictions at the command line with `cyfi predict`.

First, specify your sample points in a csv with the following columns:

* latitude
* longitude
* date

For example,

```
# sample_points.csv
latitude,longitude,date
41.424144,-73.206937,2023-06-22
36.045,-79.0919415,2023-07-01
35.884524,-78.953997,2023-08-04
```

Then run:
```
cyfi predict sample_points.csv
```

This will output a `preds.csv` that contains a column for cyanobacteria density and a column for the associated severity level based on WHO thresholds.
```
# preds.csv
sample_id,date,latitude,longitude,density_cells_per_ml,severity
7ff4b4a56965d80f6aa501cc25aa1883,2023-06-22,41.424144,-73.206937,34173.0,moderate
882b9804a3e28d8805f98432a1a9d9af,2023-07-01,36.045,-79.0919415,7701.0,low
10468e709dcb6133d19a230419efbb24,2023-08-04,35.884524,-78.953997,4053.0,low
```

To see all of the available options, run `cyfi predict --help`.

### Generate prediction for a single point

Or, generate a cyanobacteria estimate for a single point on a single date using `cyfi predict-point`.

Just specify the latitude, longitude, and date as arguments at the command line.

```
cyfi predict-point --lat 41.2 --lon -73.2 --date 2023-09-01
```

This will print out the estimated cyanobacteria density and associated severity level based on WHO thresholds.

```
2023-10-03 13:27:22.565 | SUCCESS  | cyfi.cli:predict_point:154 - Estimate generated:
date                    2023-09-01
latitude                      41.2
longitude                    -73.2
density_cells_per_ml        38,262
severity                  moderate
```

To see all of the available options, run `cyfi predict-point --help`.

------

## About the model

CyFi was born out of the [Tick Tick BLoom](https://www.drivendata.org/competitions/143/tick-tick-bloom/) machine learning competition, hosted by DrivenData. The goal in that challenge was to detect and classify the severity of cyanobacteria blooms in small, inland water bodies using publicly available satellite, climate, and elevation data. Labels were based on "in situ" samples that were collected manually by [many organizations](https://www.drivendata.org/competitions/143/tick-tick-bloom/page/651/#about-the-project-team) across the U.S. The model in CyFi is based on the [winning solutions](https://github.com/drivendataorg/tick-tick-bloom) from that challenge, and has been optimized for generalizability and efficiency.

### Why use machine learning

Machine learning is particularly well-suited to this task because indicators of cyanobacteria are visible from free, routinely collected data sources. Whereas manual water sampling is time and resource intensive, machine learning models can generate estimates in seconds. This allows water managers to prioritize where water sampling will be most beneficial, and can provide a birds-eye view of water conditions across the state.

### Data sources

CyFi relies on two data sources as input:

**Sentinel-2 satellite imagery**

*  Sentinel-2 is a wide-swath, high-resolution, multi-spectral imaging mission. It supports the monitoring of vegetation, soil and water cover, as well as observation of inland waterways and coastal areas. The Sentinel-2 Multispectral Instrument (MSI) samples [13 spectral bands](https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#available-bands-and-data): four bands at 10 metres, six bands at 20 metres and three bands at 60 metres spatial resolution. The mission provides a global coverage of the Earth's land surface every 5 days. Sentinel-2 data is accessed through Microsoft's [Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a).

**Land cover map**

* The Climate Research Data Package (CRDP) Land Cover Gridded Map (2020) classifies land surface into 22 classes, which have been defined using the United Nations Food and Agriculture Organization's Land Cover Classification System (LCCS). This map is based on data from the Medium Resolution Imaging Spectrometer (MERIS) sensor on board the polar-orbiting Envisat-1 environmental research satellite by the European Space Agency. This data comes from the CCI-LC database hosted by the ESA Climate Change Initiative's Land Cover project.

### Overview of the model

Each observation (or "sampling point") is a unique combination of date, latitude, and longitude.

```
# example input csv row
latitude,longitude,date
41.424144,-73.206937,2023-06-22
```

The feature generation for each observation is as follows:

- identify relevant Sentinel-2 tiles based on
    - a bounding box of 2,000m around the sampling point
    - a time range of 30 days prior to (and including) the sampling date
- select the most recent image that has a bouding box containing fewer than 5% of cloud pixels
- filter the pixels in the bounding box to the water area using the [scene classification (SCL) band]()
- generate summary statistics (e.g., mean, max, min) and ratios (e.g, NDVI) using the 15 Sentinel-2 bands

The land cover value for each sampling point is looked up from the static land cover map and add that to the satellite features.

```
# example features csv row
B01_mean,B02_mean,B03_mean,B04_mean,B05_mean,B06_mean,B07_mean,B08_mean,B09_mean,B11_mean,B12_mean,B8A_mean,WVP_mean,AOT_mean,percent_water,green95th,green5th,green_red_ratio,green_blue_ratio,red_blue_ratio,green95th_blue_ratio,green5th_blue_ratio,NDVI_B04,NDVI_B05,NDVI_B06,NDVI_B07,AOT_range,month,days_before_sample,land_cover
548.1428571428571,1341.6052631578948,1607.3355263157894,1613.8026315789473,234.0,287.6666666666667,265.3333333333333,2929.2960526315787,3316.714285714286,362.6666666666667,153.33333333333334,171.66666666666666,1742.828947368421,76.0,7.138607971445568e-05,3918.9999999999973,711.55,0.9959926293732521,1.1980688884094073,1.2028893117043604,2.921127478864673,0.5303720994095839,0.2895586278203927,0.8520530509274761,0.8211563566211183,0.8338878778871611,0.0,5,6,130
```

Cyanobacteria estimates are then generated by a [LightGBM model](https://github.com/microsoft/LightGBM), a gradient-boosted decision tree algorithm. Density values are discretized into severity buckets using the WHO guidlines.

**Severity buckets**  

- Low: 0 - 20,000 cells/ml
- Moderate: 20,000 - 100,000 cells/ml
- High: > 100,000 cells/ml

```
# example predictions csv row
date,latitude,longitude,density_cells_per_ml,severity
2019-08-26,38.9725,-94.67293,426593.0,high
```
