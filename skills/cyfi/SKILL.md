---
name: cyfi
description: Estimates cyanobacteria density and harmful algal bloom severity for inland lakes and reservoirs from Sentinel-2 satellite imagery, using the CyFi CLI. Use when someone wants cyanobacteria or algal bloom estimates for locations and dates, wants to compare satellite estimates with their own cell counts, or wants to view the imagery behind an estimate.
---

# Estimate cyanobacteria with CyFi

CyFi estimates cyanobacteria density (cells/ml) and a severity level for a location on a date, from Sentinel-2 imagery taken in the 30 days up to that date. Your job: turn the person's locations and dates into CyFi's input, run it, and help them read the results honestly. Assume they know their water bodies well and CyFi not at all.

Not installed? `pip install cyfi`, or `conda install -c conda-forge cyfi`. On a Mac, `pip` needs `brew install libomp` first; conda doesn't. `cyfi --version` checks the install.

## Open the run

Before any command, tell the person in a few sentences, in your own words:

- you're using CyFi, which estimates cyanobacteria from satellite imagery, and what you'll do: gather locations and dates, run the estimates, explain them, and optionally compare them with their measurements
- each estimate comes from the most recent clear satellite image in the 30 days up to the date, and some locations and dates won't have one
- estimates are a screening tool for deciding where to look or sample, not a replacement for measurements

## Steps

### 1. Gather locations and dates

CyFi reads a csv with columns `date` (YYYY-MM-DD), `latitude` and `longitude`, in decimal degrees (WGS-84, EPSG:4326). Other columns are ignored.

- Use the coordinates and dates the person gives. When they give a name, such as a lake or a sampling site, look it up and show them the coordinates you found before using them.
- Convert other coordinate systems to WGS-84, and say that you did.
- A point should be on or beside the water they mean. A dock or shoreline sampling site is fine.
- Dates can't be in the future.
- Never invent a location or date to fill a gap. Ask.

### 2. Run CyFi

- One location: `cyfi predict-point --lat <lat> --lon <lon> --date <YYYY-MM-DD>`. It prints the estimate and saves nothing.
- Several: `cyfi predict <samples.csv> -d <output dir> --keep-metadata`. `--keep-metadata` saves the image details that step 3 and step 5 use. Add `-o` to replace earlier results.

Before a run, tell the person it downloads satellite imagery and can take a few minutes for a handful of points, and much longer for hundreds. The first run also downloads a 2 GB land cover map. Downloaded imagery is cached, so rerunning the same points is fast. Run one CyFi job at a time: parallel jobs can hit the imagery service's rate limit and fail.

### 3. Explain the results

`preds.csv` has one row per location and date, with `density_cells_per_ml` and `severity`. Severity follows WHO recreational thresholds: low is under 20,000 cells/ml, moderate 20,000 to 100,000, and high over 100,000.

Relay the results as a short table. For each row, also say:

- how old the image was, from `days_before_sample` in `sentinel_metadata.csv`. An image from three weeks before the date may not reflect conditions on the date.
- when density and severity are empty: there was no usable image, because the 30 days before the date had no clear image of the area. Suggest a different date rather than guessing.

### 4. Compare with measurements (optional)

If the person has cyanobacteria cell counts, compare them with `cyfi evaluate <preds.csv> <measurements.csv> --save-dir <dir> -o`. The measurements csv needs `date`, `latitude`, `longitude` and `density_cells_per_ml`, plus `region` for results by region. Predictions are matched to measurements by the exact latitude, longitude and date text, so build the step 1 csv from the measurement file's own columns. Relay the accuracy and the plots it writes.

### 5. Show the imagery (optional)

`cyfi visualize <output dir>` opens the CyFi Explorer, which shows the satellite image behind each estimate next to a map of all estimates. It runs a local web server until stopped, so start it in the background and give the person the URL it prints.

## What an estimate means

- An estimate describes the water in a 4 km x 4 km box around the point, not the point alone. If other lakes, ponds or rivers are in the box, they count too. Points a few hundred meters apart in the same lake usually get nearly the same estimate, so don't present CyFi as mapping variation within a lake.
- The model was trained on cyanobacteria cell counts from US lakes and reservoirs, sampled from 2013 to 2021. Use it with care outside the US.
- Only cyanobacteria cell counts can confirm or contradict an estimate. Chlorophyll-a, microcystin and Secchi depth measure different things.
- Sentinel-2 imagery is sparse before 2017, so earlier dates often have no estimate.

## Rules

- Don't present an estimate as a measurement, a toxin level, or a basis for a health advisory on its own. It tells the person where a bloom is likely, so they can decide where to look or sample.
- Learn the tool from `cyfi <command> --help` and its output.

## Feedback

When CyFi does something wrong, or its output leaves you guessing, tell the person, and suggest they open an issue at <https://github.com/drivendataorg/cyfi/issues> with the command, its output and what they expected.
