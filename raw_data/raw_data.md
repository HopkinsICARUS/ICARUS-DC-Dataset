# Raw Data Overview

This directory contains the **unprocessed source data** that forms the foundation of the ICARUS Data Center Dataset.  
All data here is stored as `.csv` files and serves as input to the data cleaning and integration pipeline that produces the standardized datasets in the [`/data`](../data/formatted_data.md) directory.

The raw data is organized into three main folders:
```
raw_data/
├── data_centers/ # Data center infrastructure and attributes
├── LMP/ # Locational Marginal Price (LMP) data
└── rates/ # Power usage rate and subsidy data
```

---

## `data_centers/`

Contains the foundational information describing individual data centers.

### Columns
| Column | Description |
|--------|--------------|
| `Name` | Data center name or facility identifier |
| `Operator` | Owning or operating company |
| `State` | U.S. state abbreviation |
| `City` | City or locality |
| `Power (MW)` | Reported or estimated IT power capacity in megawatts |
| `Whitespace (sqft)` | Available whitespace or floor area in square feet |
| `Type` | Data center classification (e.g., enterprise, co-location, hyperscale) |
| `Year Built` | Original construction year |
| `Year Renovated` | Most recent major renovation year |
| `UPS` | Uninterruptible power supply configuration or vendor type |
| `Cooling System` | Cooling system type (e.g., air-cooled, chilled water) |

### Notes
- Serves as the **core dataset** for facility-level modeling.
- Future versions may include additional metadata such as PUE (Power Usage Effectiveness) and renewable sourcing indicators.
- Sources: <INSERT HERE - data sources or collection methods>
---

## Intended Use

All raw data in this folder is processed by scripts in the [`/src`](../src/data_generation.md) directory to create:
- **Cleaned datasets** (standardized, validated, and merged tables).  
- **Imputed datasets** (missing data estimated via statistical imputation).

These outputs are stored in [`/data`](../data/formatted_data.md) and form the basis for downstream analytical and modeling workflows.

---
