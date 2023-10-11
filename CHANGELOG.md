# CyFi changelog

### Unreleased

- Added support for calling the CLI with `python -m cyfi` ([PR #122](https://github.com/drivendataorg/cyfi/pull/122))

### v1.0.0 - 2023-10-10

CyFi is a command line tool that uses satellite imagery and machine learning to estimate cyanobacteria levels in small, inland water bodies.

CyFi has its origins in the [Tick Tick Bloom](https://www.drivendata.org/competitions/143/tick-tick-bloom/) machine learning competition, hosted by DrivenData and created on behalf of [NASA](https://www.nasa.gov/). The goal in that challenge was to detect and classify the severity of cyanobacteria blooms in small, inland water bodies using publicly available satellite, climate, and elevation data. Labels were based on "in situ" samples that were collected manually by [many organizations](https://www.drivendata.org/competitions/143/tick-tick-bloom/page/651/#about-the-project-team) across the U.S. The model in CyFi is based on the [winning solutions](https://github.com/drivendataorg/tick-tick-bloom) from that challenge, and has been optimized for generalizability and efficiency.
