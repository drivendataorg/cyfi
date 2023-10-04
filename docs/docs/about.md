# About the project

Inland water bodies provide a variety of critical services for both human and aquatic life, including drinking water, recreational and economic opportunities, and marine habitats. A significant challenge water quality managers face is the formation of harmful algal blooms, which can harm human health, threaten other mammals like pets, and damage aquatic ecosystems.

Cyanobacteria are microscopic algae that can multiply very quickly in warm, nutrient-rich environments, often creating visible blue or green blooms. These blooms can block sunlight from reaching the rest of the aquatic ecosystem beneath the surface, and take away oxygen and nutrients from other organisms. Cyanobacteria can produce toxins that are poisonous to humans, pets, and livestock. The effect of climate change on marine environments likely makes harmful algal blooms form more often.

Manual water sampling, or “in situ” sampling, is generally used to monitor cyanobacteria in inland water bodies. In situ sampling is accurate, but time intensive and difficult to perform continuously. Public health managers also rely on the public to notice and report blooms.

**The goal of CyFi is to help water quality managers better allocate resources for in situ sampling, and make more informed decisions around public health warnings for critical resources like drinking water reservoirs.** Ultimately, more accurate and more timely detection of algal blooms helps keep both the human and marine life that rely on these water bodies safe and healthy.

CyFi was born out of the [Tick Tick BLoom](https://www.drivendata.org/competitions/143/tick-tick-bloom/) machine learning competition, hosted by DrivenData. The goal in that challenge was to detect and classify the severity of cyanobacteria blooms in small, inland water bodies using publicly available satellite, climate, and elevation data. Labels were based on "in situ" samples that were collected manually by [many organizations](https://www.drivendata.org/competitions/143/tick-tick-bloom/page/651/#about-the-project-team) across the U.S. The model in CyFi is based on the [winning solutions](https://github.com/drivendataorg/tick-tick-bloom) from that challenge, and has been optimized for generalizability and efficiency.

## Additional resources

**Tick Tick Bloom machine learning competition**

- [Tick Tick Bloom competition](https://www.drivendata.org/competitions/143/tick-tick-bloom/)
- [Meet the winners blog post](https://drivendata.co/blog/tick-tick-bloom-challenge-winners)
- [Code from winning solutions](https://github.com/drivendataorg/tick-tick-bloom)

**About harmful algal blooms (HABs)**

- [CDC resources on HABs](https://www.cdc.gov/habs/general.html)
- [EPA resources on HABs](https://www.epa.gov/cyanohabs)

**Related tools**

There are other groups working on cyanobacteria estimates from satellite imagery. Here are a few that use Sentinel-3 (300m resolution) imagery:

- [NOAA's Harmful Algal Bloom Monitoring System](https://coastalscience.noaa.gov/science-areas/habs/hab-monitoring-system/)
- [Cyanobacteria Assessment Network (CyAN)](https://oceancolor.gsfc.nasa.gov/about/projects/cyan/)
    - [Dashboard](https://qed.epa.gov/cyanweb/)
    - [Paper](https://www.sciencedirect.com/science/article/pii/S1364815218302482?via%3Dihub)

**EPA guidance on HABs**

- [Recommendations for Cyanobacteria and Cyanotoxin Monitoring in Recreational Waters](https://www.epa.gov/sites/default/files/2019-09/documents/recommend-cyano-rec-water-2019-update.pdf)
- [Recommended Human Health Recreational Ambient Water Quality Criteria or Swimming Advisories for Microcystins and Cylindrospermopsin](https://www.epa.gov/sites/default/files/2019-05/documents/hh-rec-criteria-habs-document-2019.pdf)

**Related research on using satellite imagery to monitor HABs**

- [Quantifying national and regional cyanobacterial occurrence in US lakes using satellite remote sensing](https://www.sciencedirect.com/science/article/pii/S1470160X19309719?ref=pdf_download&fr=RR-2&rr=8109976f78329642)
- [Evaluation of a satellite-based cyanobacteria bloom detection algorithm using field-measured microcystin data](https://www.sciencedirect.com/science/article/pii/S0048969721005301?ref=pdf_download&fr=RR-2&rr=7ee00136c8e396d1#f0015)
- [Satellite monitoring of cyanobacterial harmful algal bloom frequency in recreational waters and drinking water sources](https://www.sciencedirect.com/science/article/pii/S1470160X17302194?ref=pdf_download&fr=RR-2&rr=805b0d4bedb0642f)
- [Satellite remote sensing to assess cyanobacterial bloom frequency across the United States at multiple spatial scales](https://www.sciencedirect.com/science/article/pii/S1470160X21004878)
- [Challenges for mapping cyanotoxin patterns from remote sensing of cyanobacteria](https://pubmed.ncbi.nlm.nih.gov/28073474/)
- [Satellites for long-term monitoring of inland U.S. lakes: The MERIS time series and application for chlorophyll-a](https://www.sciencedirect.com/science/article/pii/S0034425721004053)
- [Mapping algal bloom dynamics in small reservoirs using Sentinel-2 imagery in Google Earth Engine](https://www.sciencedirect.com/science/article/pii/S1470160X2200512X)