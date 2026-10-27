---
name: cyfi
description: Estimates cyanobacteria density and harmful algal bloom severity for inland lakes and reservoirs from Sentinel-2 satellite imagery, using the CyFi CLI, and opens the CyFi Explorer to browse the imagery behind each estimate. Use when someone asks about current or past cyanobacteria or algal bloom conditions at lakes, or wants to look at CyFi's estimates on a map.
---

# Estimate cyanobacteria with CyFi

CyFi estimates cyanobacteria density (cells/ml) and a severity level for a location on a date, from the most recent clear Sentinel-2 image in the 30 days up to that date. Learn the commands from `cyfi <command> --help`. Install with `pip install cyfi` (on a Mac, `brew install libomp` first) or `conda install -c conda-forge cyfi`.

## Generate estimates

1. Write a csv with `date` (YYYY-MM-DD), `latitude` and `longitude` in decimal degrees (WGS-84), one row per location. For "current conditions", use today's date. For a period such as "last week", use its last day, since CyFi already uses the most recent clear image up to that date; add earlier dates only if the person wants to see change over time. When the person names a lake or site, look up its coordinates and show them before running. A point should be on the water. Never invent a location or date. Sentinel-2 imagery starts in mid-2015, so don't run earlier dates; CyFi fails on them with `KeyError: 'item_id'`.
2. Run `cyfi predict <csv> -d <dir> --keep-metadata`, even for one location. `--keep-metadata` saves which image each estimate used, which the next section and the Explorer need. Prefer it to `cyfi predict-point`, whose results the Explorer can't show.
3. Runs download imagery: a few minutes for a handful of locations, about 15 minutes per 100, plus a 2 GB land cover map the first time. Tell the person before a long run, and run it detached so a command time limit can't cut it off, for example `nohup cyfi predict ... > predict.log 2>&1 &`, then check the log until it finishes. Run one job at a time, since parallel jobs hit the imagery service's rate limit. Re-running the same locations and dates reuses the downloaded imagery and takes well under a minute, so if an earlier run skipped `--keep-metadata`, re-run it with `-o --keep-metadata`.

## Explain the results

Give a short table of the location, estimate, severity, and image date (the date minus `days_before_sample` in `sentinel_metadata.csv`). Then:

- An empty estimate means no clear image in the 30 days before the date. If no location had a clear image, CyFi stops with "Relevant satellite data is not available". Either way, suggest another date rather than guessing.
- Say how old the image is. An image from three weeks before may not reflect conditions now.
- Severity uses WHO thresholds: low is under 20,000 cells/ml, moderate 20,000 to 100,000, and high over 100,000.
- An estimate describes the water in a 4 km x 4 km box around the point. Nearby points in one lake get nearly the same estimate, so don't present CyFi as mapping variation within a lake. Separate lakes within a few kilometers share part of their box, so say so when their estimates are close.
- CyFi is a screening tool for deciding where to look or sample. Don't present an estimate as a measurement, a toxin level, or a basis for an advisory on its own.

## Open the CyFi Explorer

Offer this after any run. `cyfi visualize <dir>` opens a local web app with a map of every estimate and the Sentinel-2 image behind the selected one, so the person can see the water for themselves. It needs the `preds.csv` and `sentinel_metadata.csv` from a `--keep-metadata` run, and an internet connection to load imagery. It keeps running until stopped, so start it as a detached process that outlives your command, not as a background task with a time limit, for example `nohup cyfi visualize <dir> --port 7860 > explorer.log 2>&1 &`. Check that <http://127.0.0.1:7860> loads before giving it to the person; its startup message can be held back when output goes to a file. Locations without an estimate don't appear. Stop the process when the person is done. Each row of `sentinel_metadata.csv` also has a `visual_href` link to the full Sentinel-2 tile, a large file, if they'd rather not open the Explorer.

When CyFi does something wrong or confusing, suggest the person open an issue at <https://github.com/drivendataorg/cyfi/issues>.
