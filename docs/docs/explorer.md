# CyFi Explorer

The focus of CyFi making it easy to generate predictions using a trained machine learning model. That said, we know it can often be helpful to view predictions in context. That's why we've created CyFi Explorer!

CyFi Explorer lets you see the corresponding Sentinel-2 imagery for each cyanobacteria estimate. The explorer runs locally on your machine and is intended to help you get a sense of where the model is performing well and where it might be getting tripped up. It is not intended to replace more robust data analytics tools and decision-making workflows.

<img class="shadow p-1 mb-5 bg-white rounded" style="display: block; margin: 0 auto; max-width: 95%" src="../images/explorer_screenshot_1.jpg" alt="Screenshot of CyFi Explorer showing an image of a mostly green lake with a red dot in the middle on the left and a table with sampling point details on the right."/>

## Prerequisite: generating predictions

Before you can visualize predictions, you first have to generate them!

Per the [Quickstart](../#quickstart) page, you'll want to run `cyfi predict` and specify an input csv containing sampling points (where each row is a unique combination of latitude, longitude, and date). To use the explorer, you must add the `--keep-metadata` flag which will output a `sentinel_metadata.csv` file containing information about the Sentinel-2 image used for each point. You can choose to output both prediction and metadata files to the same directory by specifying `--output-directory`, or `-d`.

So your command might look something like this:
```
cyfi predict california/california_samples.csv --keep-metadata -d california/
```

After predictions have been generated, your output directory will look like:
```
california/
├── california_samples.csv
├── preds.csv
└── sentinel_metadata.csv
```

Now we're ready to launch the explorer!

## Lauching the explorer

Launching the explorer is easy! Simply run `cyfi visualize` and specify the folder containing your predictions and metadata files.

Per the above example, you would run:

```
cyfi visualize california/
```

And you'll see the following print out:

```
16:06 $ cyfi visualize california/
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

Clicking on the [`http://127.0.0.1:7860`](http://127.0.0.1:7860) link will open a browser page with the CyFi Explorer.

## Navigating the explorer

CyFi Explorer is easy to use! Simply click on the `sample_id` in the table of predictions. The Sentinel-2 image will be displayed along with information about the CyFi prediction and Sentinel-2 image. You can also sort the table by clicking on the column headers.

<img class="shadow p-1 mb-5 bg-white rounded" style="display: block; margin: 0 auto; max-width: 95%" src="../images/explorer_screenshot_2.jpg" alt="Screenshot of CyFi explorer showing the predictions table along with the Sentinel-2 imagery."/>

If you scroll down, you'll also find a map with sampling points colored by their estimated severity. Hover over points to see more details.

<img class="shadow p-1 mb-5 bg-white rounded"  style="display: block; margin: 0 auto; max-width: 80%" src="../images/explorer_screenshot_3.png" alt="Screenshot of CyFi explorer showing a couple colored points on a map along with a hoverover pop up."/>

Happy visualizing!
