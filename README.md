CyFi: Cyanobacteria Finder
==============================

Cyfi is a command line tool that uses satellite imagery and machine learning to estimate cyanobacteria levels, one type of harmful algal bloom. The goal of CyFi is to help water quality managers better allocate resources for in situ sampling, and make more informed decisions around public health warnings for critical resources like drinking water reservoirs.

## Quickstart

### Install

Install CyFi with pip:

```
pip install cyfi
```

For detailed instructions for those installing python for the first time, see the [Installation](installation.md) page.

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
