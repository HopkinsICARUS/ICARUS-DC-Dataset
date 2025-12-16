# Data Generation and Cleaning

This document describes the process and scripts used to generate the cleaned and imputed datasets from the raw source data in `raw_data/`. The main focus is on processing data centers information, performing missing value analysis, normalization, and multiple imputation, and saving the cleaned outputs for analysis.

---

## 1. Overview

Running `generate_all_data.py` will: 
- Load raw CSV data from `raw_data/`.
- Clean and preprocess the data, including handling missing values and encoding categorical features.
- Add state-level capacity columns from NREL data for estimation purposes.
- Add Latitude and longitude columns to allow for variable resolution analysis.
- Add operating status column for analysis.
- Optionally Perform NREL-based estimation to fill missing Power (MW) values using proportional allocation.
- Conduct multiple imputation (MICE) to generate a second dataset filling missing values with minimal bias.
- Save cleaned and imputed datasets in `data/data_center_dataset/` and `data/imputed_data_center_dataset/`.

The processed datasets are intended for downstream analysis and modeling, including regression, classification, and correlation studies.

---

## 2. Data Sources

The raw input data is organized in:

```
raw_data/
├── data_centers/
│   ├── DCS_Full.csv                          # Full data center dataset
│   ├── DCS_New_York.csv                      # New York specific dataset
│   ├── DCS_PJM.csv                           # PJM region dataset
│   └── datacenter_demand_capacity_by_county.csv  # NREL state capacity data
├── LMP/ [EXCLUDED FROM PUBLIC RELEASE]
│   └── *.csv # PJM electricity prices: Real Time Prices and Day Ahead Prices 
├── Capacity / [EXCLUDED FROM PUBLIC RELEASE]
│   └── *.csv # Capacity Market Data 
├── rates/ [EXCLUDED FROM PUBLIC RELEASE]
│   └── *.csv # Relevant rates for analyzing real rate / cost subsidy effects
```

- **Data Centers**: Columns include `Name, Operator, State, City, Power (MW), Whitespace (sqft), Type, Year Built, Year Renovated, UPS, Cooling System, Zone`.
- **NREL Capacity Data**: County-level data containing `state, Operating (MW), Operating and In Construction (MW), Total (MW)` - aggregated to state level for capacity targets.
- **LMP**: PJM data containing both real-time and day-ahead prices.
- **Rates**: User-supplied or externally-sourced subsidy/rate information for examining effects on costs or consumption.

---

## 3. Cleaning and Preprocessing

### 3.1 Loading the Data

The `DataCenters` class is responsible for loading the CSV data and performing initial cleaning:

- Reads CSV into a pandas DataFrame.
- Handles numeric columns formatted with commas.
- Automatically encodes categorical variables (Operator, UPS, Cooling System, State, City, Type, Zone).
- Creates a `MRC` column (Most Recent Construction/Renovation year).

Example usage:

```python
from framework.data_source import DataCenters

FD = DataCenters("raw_data/data_centers/DCS_Full.csv")
print(FD.data.head())
```

### 3.2 State Code Mapping

After initial cleaning, a `State_Code` column is added to map full state names to their 2-letter abbreviations:

```python
# Automatically applied in generate_all_data.py
# Maps "California" -> "CA", "New York" -> "NY", etc.
```

This enables proper matching with NREL data which uses state codes.

### 3.3 State Capacity Columns

Three state-level capacity columns are added based on NREL data:

- **Operating Capacity State**: Sum of Power (MW) for operating (non-construction) data centers by state
- **Operating and In Construction Capacity State**: NREL target capacity for operating + in-construction facilities by state
- **Total Capacity State**: NREL total capacity target by state

These columns enable NREL-based estimation of missing Power (MW) values.

### 3.4 NREL Estimation

When `NREL_ESTIMATION = True`, missing Power (MW) values are estimated before multiple imputation:

**Default Method: `state_proportional_allocation`**

For each state:
1. Calculates known capacity (sum of non-missing Power (MW) values)
2. Calculates remaining capacity: `NREL Target - Known Capacity`
3. Distributes remaining capacity equally among facilities with missing Power (MW) values

Configuration in `generate_all_data.py`:

```python
NREL_ESTIMATION = True  # Enable/disable NREL estimation
NREL_ESTIMATION_METHOD = "state_proportional_allocation"  # Method to use
```

Example output:

```
State: California (CA)
  NREL Target Capacity: 5000.00 MW
  Known Capacity (from Power MW): 3200.00 MW
  Remaining Capacity: 1800.00 MW
  Missing Values: 120
  → Estimated at 15.0000 MW each
```

### 3.5 Optional Missing Value Analysis

`DataCenters.randomness_of_missing()` checks whether missingness is random:

- For numeric columns, Kolmogorov-Smirnov test is used.
- For categorical columns, Chi-square tests on contingency tables are applied.
- Prints summary of significance of missingness in each column.

```python
FD.randomness_of_missing("Power (MW)")
FD.randomness_of_missing("Whitespace (sqft)")
```

### 3.6 Dropping Highly Incomplete Rows

Rows with more than k missing values can be dropped using:

```python
FD.drop_rows_missing_k(k=3)
```

### 3.7 Exploratory Modeling

Optional methods for regression and classification:

- `linear_regression(X_cols, Y_col)`: Fits a linear model.
- `random_forest_classifier(X_cols, Y_col)`: Fits a Random Forest classifier.
- `random_forest_regressor(X_cols, Y_col)`: Fits a Random Forest regressor.

These methods provide diagnostics, R² scores, and optionally plot results.


### 3.8 Labeling Data Center Status

In the data generation process, we add a column labeled `Status` to approximate the operating status of each data center. This classification is currently based solely on the `MRC` column (Most Recent Construction). Data centers with MRC < 2025 are labeled as `operating`, those with MRC = 2025 are labeled as `in construction`, and those with missing MRC values or MRC > 2025 are labeled as `planned`.

This method assumes that missing and future MRC values indicate construction has not yet begun, which appears to be a reasonable assumption based on qualitative inspection of data centers with missing MRC columns. We believe our `in_construction` label is also a fair approximation given that all data was collected in 2025 and only year-level data is available, limiting the granularity of our classification. To validate this system, we examined a random sample of datapoints and reviewed satellite imagery via Google Maps. From this inspection, it appears that each site labeled as `planned` has yet to begin construction at its listed location. One such example is Iron Mountain Data Centers CHI-1, which as of December 16, 2025, is shown as an empty lot on Google Maps. It also appears that sites labeled as `in_construction` show either incomplete builds or empty lots that have comparatively more progress compared to their `planned` counterparts, for example the Prime Data Centers Campus in Elk Grove, IL.

Future work will refine this classification by collecting additional data, including a "date of connection to the grid" field that would provide a more precise indicator of when a data center becomes operational. This additional data point will help distinguish between data centers that are physically constructed but not yet operational and those that are fully online.

### 3.9 Approximate Coordinate Assignment

The `add_approx_coordinates()` function enriches each data center record with geospatial information by performing a city–state lookup against a national cities database.

```python
def add_approx_coordinates(df):
    cities_db = pd.read_csv("raw_data/Locations/uscities.csv")
    
    # Merge on city and state
    df_merged = df.merge(
        cities_db[['city', 'state_name', 'lat', 'lng']],
        left_on=['City', 'State'],
        right_on=['city', 'state_name'],
        how='left'
    )
    
    df['latitude'] = df_merged['lat']
    df['longitude'] = df_merged['lng']
    
    return df
```

#### **Purpose**

This function assigns a latitude and longitude to each data center using the publicly available `uscities.csv` dataset.

#### **How It Works**

1. **Load City Database**  
   Reads a standardized U.S. cities dataset containing:  
   `city, state_name, lat, lng`

2. **Merge by City and State**  
   Joins the main data center dataframe with the city database using:
   - `City` → `city`
   - `State` → `state_name`

3. **Extract Coordinates**  
   The merged latitude (`lat`) and longitude (`lng`) columns are copied into:
   - `latitude`
   - `longitude`

#### **Notes & Limitations**
- Coordinates are **city-level approximations**, not exact facility locations.  
- In rare cases where city names mismatch (spelling differences, regional variations), coordinates may remain missing.  

---

## 4. Multiple Imputation (MICE)

`DataCentersMI` extends `DataCenters` to support multiple imputation for missing numeric and encoded categorical features.

### 4.1 Normalization

Before imputation, features can be normalized:

```python
FDMI = DataCentersMI("raw_data/data_centers/DCS_Full.csv")
FDMI.normalize_features(method="zscore")
```

### 4.2 Performing Imputation

```python
imputations = FDMI.multiple_imputation(m=7, max_iter=50)
```

- `m`: Number of imputed datasets to generate.
- Each imputed dataset is a pandas DataFrame of the same structure as the cleaned data.

**Note**: NREL estimation (if enabled) runs before MICE imputation, so the multiple imputation process handles remaining missing values.

### 4.3 Pooled Regression Across Imputations

```python
results = FDMI.pooled_regression(
    imputations,
    X_cols=["Whitespace (sqft)", "MRC", "UPS_Encoded", "Cooling_System_Encoded"],
    Y_col="Power (MW)"
)
```

Returns pooled regression coefficients using Rubin's rules, including within-imputation, between-imputation, and total variance.

### 4.4 Visualization

- Histograms and KDE plots compare observed vs imputed distributions.
- Interactive Plotly histograms can show missingness per row, optionally split by categorical variables.

---

## 5. Output Organization

Processed datasets are saved into:

```bash
data/
├── data_center_dataset/
│   ├── DCS_Full.csv               # Cleaned dataset (Full)
│   ├── DCS_New_York.csv           # Cleaned dataset (New York)
│   └── DCS_PJM.csv                # Cleaned dataset (PJM)
├── imputed_data_center_dataset/
│   ├── DCS_Imputed_Full_1.csv     # First imputed dataset (Full)
│   ├── DCS_Imputed_Full_2.csv     # Second imputed dataset (Full)
│   ├── ...                        # Up to m datasets for each target
│   ├── DCS_Imputed_New_York_1.csv
│   ├── DCS_Imputed_PJM_1.csv
│   └── ...
```

- `data_center_dataset/` contains cleaned but non-imputed data with added columns:
  - `State_Code`: 2-letter state abbreviation
  - `Operating Capacity State`: Operating capacity by state
  - `Operating and In Construction Capacity State`: NREL target capacity
  - `Total Capacity State`: NREL total capacity
  
- `imputed_data_center_dataset/` contains multiple imputed datasets for downstream analysis (normalized z-scores).

### 5.1 Automated Generation Script

The `generate_all_data.py` script automates the entire pipeline:

```python
# Run from command line
python generate_all_data.py
```

This script:
1. Loads raw data for Full, New_York, and PJM datasets
2. Loads NREL state capacity data
3. Performs data cleaning and standardization
4. Adds State_Code column
5. Adds state capacity columns from NREL data
6. Applies NREL estimation for missing Power (MW) values
7. Normalizes features for imputation
8. Performs multiple imputation (m=7 datasets)
9. Saves imputed datasets (normalized)
10. Reverses normalization and saves cleaned datasets

**Configuration Options:**

```python
# In generate_all_data.py
M_IMPUTATIONS = 7                              # Number of imputed datasets
MAX_ITER = 50                                  # Max iterations for MICE
NREL_ESTIMATION = True                         # Enable NREL estimation
NREL_ESTIMATION_METHOD = "state_proportional_allocation"  # Estimation method
```

---

## 6. Column Descriptions

### Cleaned Dataset Columns

| Column | Description |
|--------|-------------|
| `Name` | Data center facility name |
| `Operator` | Operating company |
| `State` | Full state name |
| `State_Code` | 2-letter state abbreviation (e.g., CA, NY) |
| `City` | City location |
| `Power (MW)` | Power capacity in megawatts |
| `Whitespace (sqft)` | Available floor space |
| `Type` | Facility type (Colocation, Hyperscale, etc.) |
| `Year Built` | Construction year |
| `Year Renovated` | Renovation year |
| `UPS` | Uninterruptible Power Supply configuration |
| `Cooling System` | Cooling system type |
| `Zone` | PJM zone (if applicable) |
| `MRC` | Most Recent Construction/Renovation year |
| `Operating Capacity State` | Sum of NREL operating capacity by state |
| `Operating and In Construction Capacity State` | NREL capacity (operating + construction) |
| `Total Capacity State` | NREL total capacity target (included planned) |
| `Status` | Operating status of a data center |
| `Latitude` | Latitude of data center by a city lookup |
| `Longitude` | Longitude of data center by a city lookup |
| `*_Encoded` | Encoded categorical variables |

### Imputed Dataset Columns

Imputed datasets contain normalized (z-score) versions of numeric columns:
- `Power (MW)`, `Whitespace (sqft)`, `MRC`
- `*_Encoded` columns (normalized categorical encodings)

---

## 7. Dependencies

- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`
- Python >= 3.9

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn plotly
```

---

## 8. Intended Use

- Provides cleaned and imputed datasets ready for statistical analysis, modeling, and visualization.
- Enables consistent handling of missing values, including:
  - NREL-based proportional allocation for Power (MW) estimation
  - Multiple imputation (MICE) for robust inference on remaining missing values
- Facilitates examination of correlations, regression relationships, and potential effects of rate/subsidy data.
- State capacity columns enable state-level analyses and comparisons with NREL benchmarks.

---

## 9. Adding Custom NREL Estimation Methods

To add a custom estimation method:

1. Define a new function in `generate_all_data.py`:

```python
def custom_estimation_method(df):
    """
    Custom estimation logic for missing Power (MW) values.
    
    Args:
        df: DataFrame with required columns
    
    Returns:
        DataFrame with estimated values filled in
    """
    # Your estimation logic here
    return df
```

2. Register it in `nrel_estimation_dispatcher`:

```python
methods = {
    "state_proportional_allocation": state_proportional_allocation,
    "custom_method": custom_estimation_method,  # Add here
}
```

3. Update configuration:

```python
NREL_ESTIMATION_METHOD = "custom_method"
```

---

## 10. Troubleshooting

### Missing NREL Data File

If you see:
```
Warning: NREL capacity file not found at raw_data/data_centers/datacenter_demand_capacity_by_county.csv
```

Ensure the NREL capacity file exists at the specified path with columns: `state`, `Operating (MW)`, `Operating and In Construction (MW)`, `Total (MW)`.

### Unmapped States

If you see:
```
Warning: The following states could not be mapped to codes: ['Some State']
```

Add the missing state to the `STATE_MAP` dictionary in `generate_all_data.py`.

### Zero Remaining Capacity

If NREL estimation sets values to 0:
```
Warning: No remaining capacity. Known capacity meets or exceeds NREL target.
```

This indicates your known data centers already account for all NREL-estimated capacity in that state. Consider:
- Verifying NREL data is current
- Checking if state capacity targets are correct
- Reviewing if some facilities should be marked as "In Construction