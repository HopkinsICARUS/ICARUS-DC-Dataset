"""
generate_all_data.py

Generates cleaned and imputed data center datasets from raw CSV sources.
Outputs:
- Cleaned dataset -> data/data_center_dataset/
- Multiple imputed datasets -> data/imputed_data_center_dataset/
"""

import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from framework.data_source import DataCentersMI
from geopy.geocoders import Nominatim

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_NAMES = ["Full","New_York","PJM"]
RAW_DC_FILES = [os.path.join(BASE_DIR, "raw_data", "data_centers", f"DCS_{TARGET_NAME}.csv").replace("\\", "/") for TARGET_NAME in TARGET_NAMES]
CLEAN_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "data_center_dataset").replace("\\", "/")
IMPUTED_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "imputed_data_center_dataset").replace("\\", "/")

# NREL state capacity data file (contains actual state-level capacity targets)
NREL_STATE_CAPACITY_FILE = os.path.join(BASE_DIR, "raw_data","data_centers", "datacenter_demand_capacity_by_county.csv").replace("\\", "/")

# Number of imputations
M_IMPUTATIONS = 7
MAX_ITER = 50

# NREL Estimation Configuration
NREL_ESTIMATION = False
NREL_ESTIMATION_METHOD = "state_proportional_allocation"

# State name to code mapping
STATE_MAP = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District Of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

# --- Ensure output directories exist ---
os.makedirs(CLEAN_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMPUTED_OUTPUT_DIR, exist_ok=True)


# -- cleanup helper functions --
def add_pjm_zone(inp_df, pjm_df, ny_df):
    # 1 Merge PJM first
    merged = inp_df.merge(
        pjm_df[["Name", "Operator", "City", "State", "Zone"]],
        on=["Name", "Operator", "City", "State"],
        how="left"
    )

    # 2 Handle rows still missing Zone (no PJM match)
    missing = merged[merged["Zone"].isna()].copy()
    if not missing.empty:
        # Try to get Zone from NY
        ny_merge = missing.merge(
            ny_df[["Name", "Operator", "City", "State", "Zone"]],
            on=["Name", "Operator", "City", "State"],
            how="left",
            suffixes=("", "_ny")
        )

        # Fill in only where NY provided a Zone
        merged = merged.merge(
            ny_merge[["Name", "Operator", "City", "State", "Zone_ny"]],
            on=["Name", "Operator", "City", "State"],
            how="left"
        )
        merged["Zone"] = merged["Zone"].fillna(merged["Zone_ny"])
        merged = merged.drop(columns=["Zone_ny"])

    # 3 Fill any remaining NaN Zones with "NONE"
    merged["Zone"] = merged["Zone"].fillna("NONE")

    return merged


def add_state_code_column(df):
    """
    Adds a State_Code column to the dataframe by mapping full state names to their abbreviations.
    
    Args:
        df: DataFrame with 'State' column containing full state names
    
    Returns:
        DataFrame with added 'State_Code' column
    """
    df['State_Code'] = df['State'].map(STATE_MAP)
    
    # Check for unmapped states
    unmapped = df[df['State_Code'].isna()]['State'].unique()
    if len(unmapped) > 0:
        print(f"Warning: The following states could not be mapped to codes: {unmapped}")
    
    return df


def add_status_column(df):
    """
    Adds a Status column to the dataframe by looking at MRC.
    
    This is implemnted under the assumption that data centers with an MRC == 2025 are in construction. 
    Data centers with an MRC less than 2025 are marked as operating and data centers that do not have
    an MRC or have an MRC > 2025 are labeled as planned. These assumptions are futher discussed in data_generation.md

    """
    def f(date):
        if date < 2025:
            return 'Operating'
        if date == 2025:
            return 'In Construction'
        else:
            return 'Planned'
    df['Status'] = df['MRC'].map(lambda x:f(x))
    

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



def add_state_capacity_columns(df, nrel_capacity_df=None):
    """
    Adds three state-level capacity aggregation columns:
    - Operating Capacity State: Sum of Power (MW) for operating data centers by state
    - Operating and In Construction Capacity State: Sum for operating + in construction (from NREL data)
    - Total Capacity State: Total sum of all Power (MW) by state (from NREL data)
    
    Args:
        df: Main dataframe with data center information (must have 'State_Code' column)
        nrel_capacity_df: Optional dataframe with NREL state capacity targets
                         Expected columns: 'state', 'Operating (MW)', 'Operating and In Construction (MW)', 'Total (MW)'
    """
    # Create a copy to avoid modifying the original during calculation
    df_copy = df.copy()
    
    # Calculate Operating Capacity State from current data (known values only)
    operating_mask = ~df_copy['Type'].str.contains('Construction', case=False, na=False)
    operating_capacity = df_copy[operating_mask].groupby('State')['Power (MW)'].sum()
    
    # Map operating capacity back to dataframe
    df['Operating Capacity State'] = df['State'].map(operating_capacity).fillna(0)
    
    # If NREL capacity data is provided, use it for target capacities
    if nrel_capacity_df is not None:
        print("Using NREL state capacity targets for estimation...")
        
        # Aggregate NREL data by state code (sum across all counties in each state)
        nrel_by_state = nrel_capacity_df.groupby('state').agg({
            'Operating (MW)': 'sum',
            'Operating and In Construction (MW)': 'sum',
            'Total (MW)': 'sum'
        }).reset_index()
        
        # Create a mapping from state code to NREL capacities
        nrel_mapping = nrel_by_state.set_index('state')
        
        # Map NREL capacities to dataframe using State_Code
        df['Operating and In Construction Capacity State'] = df['State_Code'].map(
            nrel_mapping['Operating and In Construction (MW)']
        ).fillna(0)
        
        df['Total Capacity State'] = df['State_Code'].map(
            nrel_mapping['Total (MW)']
        ).fillna(0)
        
        print(f"Loaded NREL capacity data for {len(nrel_by_state)} states")
        
        # Debug: Show sample of mapping
        print("\nSample NREL capacity mapping:")
        print(nrel_by_state.head(10))
    else:
        # Fallback: Calculate from current data (will result in no remaining capacity)
        print("Warning: No NREL capacity data provided. Using calculated values from current dataset.")
        total_capacity = df_copy.groupby('State')['Power (MW)'].sum()
        df['Total Capacity State'] = df['State'].map(total_capacity).fillna(0)
        df['Operating and In Construction Capacity State'] = df['Total Capacity State']
    
    return df


def state_proportional_allocation(df):
    """
    Default NREL estimation method: Fills missing Power (MW) values using proportional allocation.
    
    For each state:
    - Calculates: (Operating and In Construction Capacity - Known Capacity) / Number of Missing Values
    - Distributes the remaining capacity equally among facilities with missing Power values
    
    Args:
        df: DataFrame with 'Power (MW)', 'State', and 'Operating and In Construction Capacity State' columns
    
    Returns:
        DataFrame with estimated Power (MW) values filled in
    """
    df = df.copy()
    
    # Identify missing Power (MW) values
    missing_mask = df['Power (MW)'].isna()
    
    if not missing_mask.any():
        print("No missing Power (MW) values to estimate.")
        return df
    
    print("\n=== NREL Estimation Details ===")
    
    # Group by state and calculate estimations
    for state in df['State'].unique():
        state_mask = df['State'] == state
        state_missing_mask = state_mask & missing_mask
        
        if not state_missing_mask.any():
            continue
        
        # Get state capacity target from NREL data
        state_target_capacity = df.loc[state_mask, 'Operating and In Construction Capacity State'].iloc[0]
        
        # Calculate known capacity in this state (sum of non-missing Power (MW) values)
        known_capacity = df.loc[state_mask & ~missing_mask, 'Power (MW)'].sum()
        
        # Handle NaN in known_capacity sum
        if pd.isna(known_capacity):
            known_capacity = 0
        
        # Calculate remaining capacity to distribute
        remaining_capacity = state_target_capacity - known_capacity
        
        # Count missing values in this state
        n_missing = state_missing_mask.sum()
        
        # Get state code for display
        state_code = df.loc[state_mask, 'State_Code'].iloc[0]
        
        # Debug info
        print(f"\nState: {state} ({state_code})")
        print(f"  NREL Target Capacity: {state_target_capacity:.2f} MW")
        print(f"  Known Capacity (from Power MW): {known_capacity:.2f} MW")
        print(f"  Remaining Capacity: {remaining_capacity:.2f} MW")
        print(f"  Missing Values: {n_missing}")
        
        if n_missing > 0 and remaining_capacity > 0:
            # Distribute equally among missing values
            estimated_value = remaining_capacity / n_missing
            df.loc[state_missing_mask, 'Power (MW)'] = estimated_value
            print(f"  → Estimated at {estimated_value:.4f} MW each")
        elif n_missing > 0 and remaining_capacity <= 0:
            # No remaining capacity or negative (known capacity exceeds NREL target)
            print(f"  → Warning: No remaining capacity. Known capacity meets or exceeds NREL target.")
            print(f"  → Setting missing values to 0 MW")
            df.loc[state_missing_mask, 'Power (MW)'] = 0
    
    print("\n===============================\n")
    
    return df


def nrel_estimation_dispatcher(df, method_name):
    """
    Dispatcher function that calls the appropriate NREL estimation method.
    
    Args:
        df: DataFrame to process
        method_name: Name of the estimation method to use
    
    Returns:
        DataFrame with estimated Power (MW) values
    """
    methods = {
        "state_proportional_allocation": state_proportional_allocation,
        # Add more methods here as needed
        # "other_method": other_method_function,
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown NREL estimation method: {method_name}. Available methods: {list(methods.keys())}")
    
    print(f"Applying NREL estimation method: {method_name}")
    return methods[method_name](df)


def main():
    # Load NREL state capacity data if NREL estimation is enabled
    nrel_capacity_df = None
    if NREL_ESTIMATION:
        try:
            if os.path.exists(NREL_STATE_CAPACITY_FILE):
                print(f"Loading NREL state capacity data from {NREL_STATE_CAPACITY_FILE}...")
                nrel_capacity_df = pd.read_csv(NREL_STATE_CAPACITY_FILE)
                print(f"Loaded NREL data for {len(nrel_capacity_df)} counties")
            else:
                print(f"Warning: NREL capacity file not found at {NREL_STATE_CAPACITY_FILE}")
                print("NREL estimation will not be effective without target capacities.")
        except Exception as e:
            print(f"Error loading NREL capacity data: {e}")
    
    FDMI = {}
    for RAW_DC_FILE, TARGET_NAME in zip(RAW_DC_FILES,TARGET_NAMES):
        print(f"Loading raw data for {TARGET_NAME}...")
        FDMI[TARGET_NAME] = DataCentersMI(RAW_DC_FILE,delay_clean=True)
    
    for RAW_DC_FILE, TARGET_NAME in zip(RAW_DC_FILES,TARGET_NAMES):  
        print(f"Processing data for {TARGET_NAME}...")
        fdmi = FDMI[TARGET_NAME]
        #general cleanup
        if TARGET_NAME == "Full":#Add PJM and NY Zones
            fdmi.data = add_pjm_zone(fdmi.data,FDMI["PJM"].data,FDMI["New_York"])
            
        replacements = {
            "Washington D.C.": "District Of Columbia",
            "Washington Dc": "District Of Columbia",
            "Dc": "District Of Columbia",
            "D.C.": "District Of Columbia",
            "New Hamphire": "New Hampshire",  # typo fix
        }
        fdmi.data["State"] = fdmi.data["State"].replace(replacements)

        fdmi.clean_data()
        
        # Add state code column for NREL mapping
        print("Adding state code column...")
        fdmi.data = add_state_code_column(fdmi.data)
        
        # Add state capacity columns before NREL estimation
        print("Adding state capacity columns...")
        fdmi.data = add_state_capacity_columns(fdmi.data, nrel_capacity_df)
        
        # Add status column
        print("Adding status columns...")
        fdmi.date = add_status_column(fdmi.data)
        
        # Add lat and long columns
        print("Adding Latitude and Longitute columns...")
        fdmi.date = add_approx_coordinates(fdmi.data)
        
        # Apply NREL estimation if enabled
        if NREL_ESTIMATION:
            print("Applying NREL estimation for missing Power (MW) values...")
            fdmi.data = nrel_estimation_dispatcher(fdmi.data, NREL_ESTIMATION_METHOD)
        
        print("Normalizing features for imputation...")
        fdmi.normalize_features(method="zscore")

        print(f"Performing multiple imputation ({M_IMPUTATIONS} datasets)...")
        imputations = fdmi.multiple_imputation(m=M_IMPUTATIONS, max_iter=MAX_ITER)

        # Save imputed datasets
        print("Saving imputed datasets...")
        for i, imp in enumerate(imputations, start=1):
            out_file = os.path.join(IMPUTED_OUTPUT_DIR, f"DCS_Imputed_{TARGET_NAME}_{i}.csv")
            imp.to_csv(out_file, index=False)
            print(f"Saved {out_file}")

        # Reverse normalization to restore original scale
        fdmi.normalize_features(reverse=True)

        # Save cleaned, non-imputed dataset
        clean_file = os.path.join(CLEAN_OUTPUT_DIR, f"DCS_{TARGET_NAME}.csv")
        fdmi.data.to_csv(clean_file, index=False)
        print(f"Saved cleaned dataset: {clean_file}")

        print(f"Data generation for {TARGET_NAME} completed successfully.")

    print("\nAll data generation completed successfully.")

if __name__ == "__main__":
    main()