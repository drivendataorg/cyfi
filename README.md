CyFi
==============================

Cyanobacteria Finder

> Estimate cyanobacteria density based on satellite imagery.

The goal of CyFi is to help water quality managers better allocate resources for in situ sampling, and make more informed decisions around public health warnings for critical resources like drinking water reservoirs.

## Installation

CyFi is available from PyPI. To install, run:

```
pip install cyfi
```

CyFi requires python 3.10 or greater.

## Quickstart

Generate batch predictions at the command line with `predict` by specifying a csv with columns: `latitude`, `longitude`, and `date`.

```
cyfi predict sample_points.csv
```

Where your input csv looks like:

```
latitude,longitude,date
41.424144,-73.206937,2023-06-22
36.045,-79.0919415,2023-07-01
35.884524,-78.953997,2023-08-04
```

Or, generate a cyanobacteria estimate for a single point on a single date using `predict-point`.

```
cyfi predict-point --lat 41.2 --lon -73.2 --date 2023-09-01
```

# TODO: insert screencast

### Experiment module

There is an unsupported `experiment` module for training new models.

```
$ python cyfi/experiment.py --help
Usage: experiment.py [OPTIONS] CONFIG_PATH

  Run an experiment

Arguments:
  CONFIG_PATH  Path to an experiment configuration  [required]

Options:
  --install-completion [bash|zsh|fish|powershell|pwsh]
            Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
            Show completion for the specified shell, to
            copy it or customize the installation.
  --help    Show this message and exit.
```