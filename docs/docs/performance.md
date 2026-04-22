# Model Performance

CyFi is designed to provide accurate, high-resolution cyanobacteria estimates for small, inland water bodies that are often missed by other satellite-based monitoring tools.

## Accuracy

In evaluations against ground-truth "in situ" samples, CyFi achieves a **72% accuracy** in presence/absence detection of cyanobacteria blooms. This outperforms existing methods reliant on lower-resolution sensors (such as Sentinel-3), which typically achieve around 66% accuracy on the same types of water bodies.

## Resolution Advantage

Most existing harmful algal bloom (HAB) monitoring tools rely on **Sentinel-3** satellite imagery, which has a spatial resolution of **300 meters**. While effective for large lakes and oceans, this resolution is too coarse to monitor smaller reservoirs and ponds.

CyFi uses **Sentinel-2** imagery, which provides a significantly higher resolution:
- **10 to 60 meters** depending on the spectral band.
- This allows for much more granular analysis and the ability to detect blooms in water bodies that appear as only a few pixels in 300m imagery.

## Unprecedented Coverage

By leveraging high-resolution Sentinel-2 data, CyFi can monitor water bodies smaller than **250 acres**.
- **98% of lakes** in the contiguous United States are under 250 acres.
- CyFi provides **10x greater coverage** of U.S. water bodies compared to methods that rely solely on 300m resolution sensors.

This makes CyFi a unique and powerful tool for local water quality managers who need to monitor smaller, high-priority recreational and drinking water resources.

## More Information

For a full technical breakdown of the model architecture, training process, and detailed performance metrics, please refer to our reference paper:

> Dorne, E., Wetstone, K., Cerquera, T. B., & Gupta, S. (2024). Cyanobacteria detection in small, inland water bodies with CyFi. In Proceedings of the 23nd Python in Science Conference (pp. 154–173). [https://doi.org/10.25080/pdhk7238](https://doi.org/10.25080/pdhk7238)
