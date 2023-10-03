## Quickstart

#### Install
Install CyFi with pip

```
pip install cyfi
```

#### Generate batch predictions

Generate batch predictions at the command line with `predict` by specifying a csv with columns: latitude, longitude, and date.

```
cyfi predict sample_points.csv
```

Where your sample_points.csv looks like:

```
latitude,longitude,date
41.424144,-73.206937,2023-06-22
36.045,-79.0919415,2023-07-01
35.884524,-78.953997,2023-08-04
```

#### Generate prediction for a single point

Or, generate a cyanobacteria estimate for a single point on a single date using `predict-point`.

```
cyfi predict-point --lat 41.2 --lon -73.2 --date 2023-09-01
```