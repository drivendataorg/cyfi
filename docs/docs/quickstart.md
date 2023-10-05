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

<div class="table-container-class">
    <table>
        <tr>
            <th>latitude</th>
            <th>longitude</th>
            <th>date</th>
        </tr>
        <tr>
            <td>41.424144</td><td>-73.206937</td><td>2023-06-22</td>
        </tr>
        <tr>
            <td>36.045</td><td>-79.0919415</td><td>2023-07-01</td>
        </tr>
        <tr>
            <td>35.884524</td><td>-78.953997</td><td>2023-08-04</td>
        </tr>
    </table>
</div>

#### Generate prediction for a single point

Or, generate a cyanobacteria estimate for a single point on a single date using `predict-point`.

```
cyfi predict-point --lat 41.2 --lon -73.2 --date 2023-09-01
```