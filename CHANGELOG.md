# CyFi changelog

### v1.1.4 - 2025-03-10

- Set minimum versions for gradio and scikit-learn; resolve pandas warnings; docstring fix ([PR #152](https://github.com/drivendataorg/cyfi/pull/152), [PR #154](https://github.com/drivendataorg/cyfi/pull/154), [PR #146](https://github.com/drivendataorg/cyfi/pull/146))

### v1.1.3 - 2024-01-19

- Log error and exit gracefully when there is no valid satellite imagery for any point ([PR #136](https://github.com/drivendataorg/cyfi/pull/136))

### v1.1.2 - 2024-01-05

- Clean up runtime requirements ([PR #133](https://github.com/drivendataorg/cyfi/pull/133))

### v1.1.1 - 2023-11-27

- Use the `no-sign-request` flag to download the land cover map from s3 ([PR #130](https://github.com/drivendataorg/cyfi/pull/130))

### v1.1.0 - 2023-10-13

- Added support for calling the CLI with `python -m cyfi` ([PR #122](https://github.com/drivendataorg/cyfi/pull/122))
- Updated default feature fraction for LightGBM model ([PR #120](https://github.com/drivendataorg/cyfi/pull/120))
- Added Sentinel-2 tile information to CyFi explorer ([PR #116](https://github.com/drivendataorg/cyfi/pull/116))

### v1.0.0 - 2023-10-10

CyFi is a command line tool that uses satellite imagery and machine learning to estimate cyanobacteria levels in small, inland water bodies.

CyFi has its origins in the [Tick Tick Bloom](https://www.drivendata.org/competitions/143/tick-tick-bloom/) machine learning competition, hosted by DrivenData and created on behalf of [NASA](https://www.nasa.gov/). The goal in that challenge was to detect and classify the severity of cyanobacteria blooms in small, inland water bodies using publicly available satellite, climate, and elevation data. Labels were based on "in situ" samples that were collected manually by [many organizations](https://www.drivendata.org/competitions/143/tick-tick-bloom/page/651/#about-the-project-team) across the U.S. The model in CyFi is based on the [winning solutions](https://github.com/drivendataorg/tick-tick-bloom) from that challenge, and has been optimized for generalizability and efficiency.
