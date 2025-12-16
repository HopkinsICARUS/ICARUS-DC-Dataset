# ICARUS Data Center Dataset
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/Software%20License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]()

## About ICARUS

ICARUS — _Infrastructure Centric Adaptation and Resiliency in the U.S._ — is a joint research initiative between the [**Ralph O’Connor Sustainable Energy Institute (ROSEI)**](https://energyinstitute.jhu.edu/) at [**Johns Hopkins University**](https://www.jhu.edu/), [**NSF Global Center EPICS**](https://energyinstitute.jhu.edu/epics/), and the [**Johns Hopkins Applied Physics Laboratory (APL)**](https://www.jhuapl.edu/work/mission-areas/research-and-exploratory-development), supported by a [**Johns Hopkins Discovery Award**](https://research.jhu.edu/major-initiatives/discovery-awards/2024-awardees/).

ICARUS develops advanced data and modeling tools to strengthen the reliability, safety, and affordability of critical U.S. infrastructure. We focus on high-resolution projections of environmental conditions and other external stressors that affect large-scale infrastructure systems such as electric power, water supply, and transportation networks. By combining state-of-the-art computational and engineering expertise, the project produces actionable insights for planners and operators who must prepare infrastructure for a range of future scenarios—from extreme weather events to changing population patterns and demand surges.

Our goal is to make these improved datasets, algorithms, and case studies publicly available so that researchers, policymakers, and practitioners can better **assess vulnerabilities, plan investments, and enhance the resilience of essential services** nationwide.

## How Can ICARUS Help Energy Policymakers?

ICARUS aims to make energy policy analysis accessible to stakeholders who often lack the technical expertise, data, modeling, and computational resources needed to conduct quantitative energy policy analysis. ICARUS bridges this gap by providing expert analysis using our comprehensive PJM dataset and advanced modeling capabilities.
To request an ICARUS analysis, we <a href="https://hopkinsicarus.github.io/ICARUS-PJM-Dataset/">invite stakeholders to submit their energy policy requests</a> concerning PJM.
We are committed to maintaining stakeholder privacy—we will not release requests or identify who submitted them—and to responding promptly. When submitting a request, there is a field to indicate urgency.

---
## About ICARUS-DC

The **ICARUS Data Center Dataset** provides a structured and extensible framework for analyzing how data center infrastructure and operations impact regional power dynamics, electricity pricing, and grid load. It supports both statistical analysis and machine learning workflows focused on understanding energy demand and efficiency across data centers in the U.S., with an emphasis on **PJM** and **New York**.


## Repository Structure

```
root/
├── README.md                              # This file — high-level overview
├── data/                                  # Cleaned and imputed datasets and docs
│   ├── formatted_data.md                   # Notes about the formatted/cleaned stage
│   ├── data_center_dataset/                # Cleaned CSVs (DCS_Full, DCS_PJM, DCS_New_York)
│   └── imputed_data_center_dataset/        # Multiple imputed outputs + process doc
│       └── imputation_process.md
├── raw_data/                              # Unprocessed source data
│   ├── data_centers/                       # Source data center CSVs (DCS_*.csv)
│   ├── LMP/    [Excluded]                  # Locational Marginal Price data
│   ├── capacity/    [Excluded]             # Capacity Market data
│   ├── rates/  [Excluded]                  # Rate and incentive data
│   └── raw_data.md                         # Raw data overview
└── src/                                   # Data generation and processing scripts
    ├── analysis/ [Excluded]                # Analysis scripts and notebooks
    │    ├── data_validation/                # Scripts to look at NREL vs ICARUS capacity
    │    ├── lmp/                          # All of the LMP analysis math that I have been working on recently
    │    ├── missing_data /                  # Scripts to compute statistics about missing data
    │    └── regression/                     # Early basic scripts for predicting one column from another
    ├── framework/                          # Data source and helper classes
    └── data_generation.md                  # Processing pipeline documentation
```

## Dataset Overview

The dataset is designed to bridge **real-world power data** and **data center infrastructure information**, providing a unified view suitable for exploratory analysis, model development, and reproducible research.

It includes:
- **Raw data** — foundational records from diverse sources (see [`raw_data/raw_data.md`](raw_data/raw_data.md)).
- **Cleaned datasets** — standardized and preprocessed CSVs (see [`data/formatted_data.md`](data/formatted_data.md)).
- **Imputed datasets** — multiple imputed outputs and documentation (see [`data/imputed_data_center_dataset/imputation_process.md`](data/imputed_data_center_dataset/imputation_process.md)).
- **Data generation code** — scripts that build the cleaned and imputed datasets from raw data (see [`src/data_generation.md`](src/data_generation.md)).

All formatted data is stored as **.csv** files.

### Data Procurement

ICARUS uses a range of public data sources. This repository is intended for research purposes only. ICARUS, and all its supporting groups make no express or implied warranties regarding its accuracy, completeness, or fitness for a particular purpose. Users assume all responsibility and risk associated with the use, dependence, or reliance of this information.


## Environment & Dependencies

The repository was developed and tested using Python 3.11+ with the following key packages:
```
Package            Version
------------------ -----------
certifi            2025.10.5
charset-normalizer 3.4.3
contourpy          1.3.3
cycler             0.12.1
distro             1.9.0
et_xmlfile         2.0.0
fonttools          4.60.1
formulaic          1.2.1
idna               3.10
interface-meta     1.3.0
joblib             1.5.2
kiwisolver         1.4.9
linearmodels       7.0
matplotlib         3.10.6
mypy_extensions    1.1.0
narwhals           2.8.0
numpy              1.26.4
openpyxl           3.1.5
packaging          25.0
pandas             2.2.2
patsy              1.0.2
pillow             11.3.0
pip                25.2
plotly             6.3.1
pyhdfe             0.2.0
pyparsing          3.2.5
python-dateutil    2.9.0.post0
pytz               2025.2
requests           2.32.5
scikit-learn       1.7.2
scipy              1.16.2
seaborn            0.13.2
setuptools         80.9.0
six                1.17.0
statsmodels        0.14.5
tabula-py          2.10.0
threadpoolctl      3.6.0
typing_extensions  4.15.0
tzdata             2025.2
urllib3            2.5.0
wrapt              2.0.0
```

## General Usage

### Generating Clean Data

Running the Python script (`src/generate_all_data.py`) produces cleaned and processed versions of our NY, PJM, and complete data center datasets, along with imputed datasets created through the multiple-imputation procedure described in `data/imputed_data_center_dataset/imputation_process.md`.

The purpose of this workflow is to outline a methodology for generating a dataset suitable for statistical analysis at a larger sample size, while maintaining approximately the same level of bias as simple row-removal of incomplete observations. However, we do not currently provide any guarantees regarding the level of bias introduced by this imputation process. 

### Example Developer Usage

Below is an example analysis workflow that demonstrates how to load a cleaned dataset and print out each column.

```python
from src.framework.data_source import DataCenters

FD = DataCenters("data/data_center_dataset/DCS_Full.csv")

for col in FD.columns:
    print(FD[col])
```

## Citing ICARUS-DC

If you use the ICARUS-DC dataset in any of your work, we ask you to please cite it using the following citation:

### IEEE Format

Y. Dvorkin, M. Klemun, O. Reed, and S. Khanal, "ICARUS Data Center Dataset (ICARUS-DC)," ICARUS Research Initiative, Johns Hopkins University and Johns Hopkins Applied Physics Laboratory, 2025. [Online]. Available: https://github.com/hopkinsicarus/ICARUS-DC

### BibTeX Format

```bibtex
@dataset{icarus_dc_2025,
  title        = {ICARUS Data Center Dataset (ICARUS-DC)},
  author       = {Dvorkin, Yury and Klemun, Magdalena and Reed, Owen and Khanal, Saroj},
  year         = {2025},
  publisher    = {Johns Hopkins University, Ralph O'Connor Sustainable Energy Institute, NSF Global Center EPICS, and Johns Hopkins Applied Physics Laboratory},
  url          = {https://github.com/hopkinsicarus/ICARUS-DC},
  note         = {Versioned GitHub repository. Data licensed under CC BY 4.0}
}
```

---

**Last Updated:** December 2025
