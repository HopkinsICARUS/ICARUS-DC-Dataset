import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, ks_2samp
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
 

class DataCenters(object):
    def __init__(self, file_source,delay_clean=False):
        self.file_name = file_source.split("/")[-1].split(".")[0]
        self.data = pd.read_csv(file_source, thousands=',')
        if not delay_clean:
            self.clean_data()
        
    def __str__(self):
        return self.data.head().to_string()
    
    def __getitem__(self,key):
        return self.data[key]
    
    @property
    def columns(self):
        return self.data.columns 
        
    
    def correlations(self,params):
        N = len(params)
        corr = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                corr[i,j] = self.data[params[i]].corr(self.data[params[j]])
        return corr
    
    
    def clean_data(self):
        #most recent year of construction
        self.data['MRC'] = self.data["Year Rennovated"].where(self.data["Year Rennovated"].notna() & (self.data["Year Rennovated"] != ""),self.data["Year Built"])
        
        #correct data from NREL
        
        
        #encoded categorical data
        self.data['Operator_Encoded'] = self.data['Operator'].astype('category').cat.codes
        self.data['UPS_Encoded'] = self.data['UPS'].astype('category').cat.codes
        self.data['Cooling_System_Encoded'] = self.data['Cooling System'].astype('category').cat.codes
        if 'Zone' in self.data.columns:
            self.data['Zone_Encoded'] = self.data['Zone'].astype('category').cat.codes
        self.data['State_Encoded'] = self.data['State'].astype('category').cat.codes
        self.data['City_Encoded'] = self.data['City'].astype('category').cat.codes
        self.data['Type_Encoded'] = self.data['Type'].astype('category').cat.codes
        
        
    def randomness_of_missing(self,column):
        # Split into missing vs non-missing groups
        MD = self.data[self.data[column].isna()]
        NMD = self.data[self.data[column].notna()]

        # Columns to exclude from analysis
        exclude_cols = {'Operator_Encoded',"UPS_Encoded", 'State_Encoded', 'City_Encoded', 'Type_Encoded', 
                        'Cooling_System_Encoded', 'Zone_Encoded', 'Name', column}

        results = {}

        for col in self.data.columns:
            if col in exclude_cols:
                continue  # skip excluded cols and the target col itself

            try:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    # Numerical: KS test (distribution comparison)
                    if len(MD[col].dropna()) > 0 and len(NMD[col].dropna()) > 0:
                        stat, p_val = ks_2samp(MD[col].dropna(), NMD[col].dropna())
                        test_type = "KS test (numeric)"
                    else:
                        p_val, test_type = np.nan, "Insufficient data"
                else:
                    # Categorical: Chi-square test
                    contingency = pd.crosstab(
                        self.data[col].fillna("MISSING"),
                        self.data[column].isna()
                    )
                    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                        stat, p_val, _, _ = chi2_contingency(contingency)
                        test_type = "Chi-square (categorical)"
                    else:
                        p_val, test_type = np.nan, "Insufficient categories"

                results[col] = {
                    "test": test_type,
                    "p_value": p_val,
                    "significant": (p_val < 0.05) if not pd.isna(p_val) else False
                }

            except Exception as e:
                results[col] = {"error": str(e)}

        # Print analysis summary
        print("\n=== Randomness of Missingness Analysis ===")
        for col, res in results.items():
            if "error" in res:
                print(f"{col}: ERROR - {res['error']}")
            else:
                print(f"{col}: {res['test']} | p={res['p_value']:.4f} | Significant={res['significant']}")
                
    def linear_regression(self, X_cols, Y_col):
        # Drop rows with missing data in X or Y
        df_clean = self.data[X_cols + [Y_col]].dropna()

        X = df_clean[X_cols].values
        y = df_clean[Y_col].values

        model = LinearRegression()
        model.fit(X, y)

        # Collect results
        results = {
            "intercept": model.intercept_,
            "coefficients": dict(zip(X_cols, model.coef_)),
            "r2": model.score(X, y),
            "n_samples": len(df_clean)
        }

        return model, results
        
    def plot_regressions(self, X_cols, Y_col, n_cols=2):
        """
        Plot scatter + regression line for each X in X_cols vs Y_col in a grid layout.
        
        Args:
            X_cols (list or str): Predictor column(s).
            Y_col (str): Target column.
            n_cols (int): Number of columns in the grid (default=2).
        """
        if isinstance(X_cols, str):
            X_cols = [X_cols]

        n = len(X_cols)
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = np.array(axes).reshape(-1)  # flatten in case it's 2D

        for ax, X_col in zip(axes, X_cols):
            df_clean = self.data[[X_col, Y_col]].dropna()
            X = df_clean[[X_col]].values
            y = df_clean[Y_col].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            ax.scatter(X, y, alpha=0.5, label="Data")
            ax.plot(X, y_pred, color="red", label="Fit")
            ax.set_xlabel(X_col)
            ax.set_ylabel(Y_col)
            ax.set_title(f"{X_col} vs {Y_col}\nR² = {model.score(X, y):.3f}")
            ax.legend()

        # Hide any unused subplots
        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()
                
    def random_forest_classifier(self, X_cols, Y_col, test_size=0.2, n_estimators=100, max_depth=None,show_plots=False):
        df_clean = self.data[X_cols + [Y_col]].dropna()
        X = df_clean[X_cols]
        y = df_clean[Y_col].astype('category')

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=13)

        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
            remainder='passthrough'
        )

        clf = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=13))
        ])

        clf.fit(X_train, y_train)

        # Predictions
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # Plot confusion matrices side by side
        if (show_plots):
            fig, axes = plt.subplots(1, 2, figsize=(12,6))
            fig.suptitle(f"Random Forest Classifier Results - Predicting {Y_col}")

            cm_train = confusion_matrix(y_train, y_train_pred, labels=y.cat.categories)
            cm_test = confusion_matrix(y_test, y_test_pred, labels=y.cat.categories)

            disp_train = ConfusionMatrixDisplay(cm_train, display_labels=y.cat.categories)
            disp_train.plot(cmap=plt.cm.Blues, ax=axes[0], colorbar=False)
            axes[0].set_title("Train Set")

            disp_test = ConfusionMatrixDisplay(cm_test, display_labels=y.cat.categories)
            disp_test.plot(cmap=plt.cm.Blues, ax=axes[1], colorbar=False)
            axes[1].set_title("Test Set")

            plt.tight_layout()
            plt.show()

        return clf


    def random_forest_regressor(self, X_cols, Y_col, test_size=0.2, n_estimators=100, max_depth=None,show_plots=False):
        df_clean = self.data[X_cols + [Y_col]].dropna()
        X = df_clean[X_cols]
        y = df_clean[Y_col].astype(float)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=13)

        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
            remainder='passthrough'
        )

        reg = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=13))
        ])

        reg.fit(X_train, y_train)

        # Predictions
        y_train_pred = reg.predict(X_train)
        y_test_pred = reg.predict(X_test)

        if (show_plots):
            # Plot side by side
            fig, axes = plt.subplots(1, 2, figsize=(12,6))
            fig.suptitle(f"Random Forest Regressor Results - Predicting {Y_col}")
            # Train plot
            
            axes[0].scatter(y_train, y_train_pred, alpha=0.5)
            axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', lw=2)
            axes[0].set_xlabel("Actual")
            axes[0].set_ylabel("Predicted")
            axes[0].set_title(f"Train Set (R²={r2_score(y_train, y_train_pred):.3f})")

            # Test plot
            axes[1].scatter(y_test, y_test_pred, alpha=0.5)
            axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
            axes[1].set_xlabel("Actual")
            axes[1].set_ylabel("Predicted")
            axes[1].set_title(f"Test Set (R²={r2_score(y_test, y_test_pred):.3f})")

            plt.tight_layout()
            plt.show()

        return reg
    
    def drop_rows_missing_k(self, k):
        """
        Drop rows that are missing more than k elements.
        
        Args:
            k (int): Maximum number of missing values allowed per row.
                     Rows with more than k missing values will be dropped.
        
        Returns:
            pd.DataFrame: The cleaned dataframe with rows removed.
        """
        if k < 0:
            raise ValueError("k must be non-negative")
            
        original_shape = self.data.shape
        # Count missing values per row and keep only rows with <= k missing values
        self.data = self.data[self.data.isnull().sum(axis=1) <= k]
        new_shape = self.data.shape
        
        print(f"Removed {original_shape[0] - new_shape[0]} rows with more than {k} missing values")
        print(f"Data shape changed from {original_shape} to {new_shape}")
        
        return self.data

    def missingness_histogram(self, control_factor=None, top_n=None,advanced=True):
        """
        Plot histogram of missing value counts per row.
        Uses Plotly for interactivity if available.

        Args:
            control_factor (str, optional): Column name to split histograms by.
            top_n (int, optional): Limit to top N unique values of control_factor (by count).
        """
        cols = ['Name', 'Operator', 'State', 'City', 'Power (MW)', 'Whitespace (sqft)',
                'Type', 'UPS', 'Cooling System', 'MRC']

        missing_counts_per_row = self.data[cols].isnull().sum(axis=1)
        df = pd.DataFrame({"MissingCount": missing_counts_per_row})

        # --- interactive version using Plotly ---
        if (advanced):

            if control_factor is not None and control_factor in self.data.columns:
                df[control_factor] = self.data[control_factor].astype(str)

                # drop NAs for clarity
                df = df.dropna(subset=["MissingCount", control_factor])

                if top_n is not None:
                    # Limit to top N groups by frequency
                    top_groups = df[control_factor].value_counts().nlargest(top_n).index
                    df = df[df[control_factor].isin(top_groups)]

                fig = px.histogram(
                    df,
                    x="MissingCount",
                    color=control_factor,
                    nbins=int(df["MissingCount"].max() + 1),
                    barmode="stack",
                    opacity=0.7,
                    title=f"Distribution of Missing Values per Row by {control_factor} - {self.file_name}",
                    labels={"MissingCount": "Number of Missing Values", "count": "Row Count"},
                )

                fig.update_layout(
                    legend_title=control_factor,
                    xaxis=dict(dtick=1),
                    template="plotly_white",
                    bargap=0.05,
                )
                fig.show()
            else:
                fig = px.histogram(
                    df,
                    x="MissingCount",
                    nbins=int(df["MissingCount"].max() + 1),
                    title=f"Distribution of Missing Values per Row - {self.file_name}",
                    labels={"MissingCount": "Number of Missing Values", "count": "Row Count"},
                )
                fig.update_layout(
                    xaxis=dict(dtick=1),
                    template="plotly_white",
                    bargap=0.05,
                )
                fig.show()

        # --- fallback to Matplotlib ---
        else:
            if control_factor is not None and control_factor in self.data.columns:
                unique_vals = self.data[control_factor].dropna().unique()
                n = len(unique_vals)

                plt.figure(figsize=(10, 6))

                # Prepare list of arrays — one for each group
                data_to_plot = [
                    missing_counts_per_row[self.data[control_factor] == val]
                    for val in unique_vals
                ]

                # Compute global bin range across all groups
                max_missing = int(missing_counts_per_row.max())
                bins = range(0, max_missing + 2)

                # Pass all data arrays at once for stacking
                plt.hist(
                    data_to_plot,
                    bins=bins,
                    stacked=True,          
                    label=[str(v) for v in unique_vals],
                    edgecolor="black",
                    alpha=0.8
                )

                plt.legend(loc="upper right", fontsize="small")
                plt.title(f"Distribution of Missing Values per Row by {control_factor} - {self.file_name}")

            else:
                plt.figure(figsize=(8, 5))
                plt.hist(missing_counts_per_row, bins=range(0, missing_counts_per_row.max() + 2), edgecolor="black")
                plt.title(f"Distribution of Missing Values per Row - {self.file_name}")

            plt.xlabel("Number of Missing Values")
            plt.ylabel("Count of Rows")
            plt.grid(axis="y", alpha=0.75)
            plt.show()

        # Return summary counts
        return {c: int((missing_counts_per_row == c).sum()) for c in range(0, int(missing_counts_per_row.max() + 1))}
    
    def missing_per_column_bar(self, advanced=True):
        """
        Generate a bar chart showing how many rows are missing each of the key columns.
        """
        cols = ['Name', 'Operator', 'State', 'City', 'Power (MW)',
                'Whitespace (sqft)', 'Type', 'UPS', 'Cooling System', 'MRC']

        missing_counts = self.data[cols].isna().sum().sort_values(ascending=False)
        df_missing = missing_counts.reset_index()
        df_missing.columns = ["Column", "MissingRows"]

        if advanced:
            fig = px.bar(
                df_missing,
                x="Column",
                y="MissingRows",
                title=f"Missing Rows per Column - {self.file_name}",
                labels={"MissingRows": "Number of Missing Rows", "Column": "Data Column"},
            )
            fig.update_layout(xaxis_tickangle=-45, template="plotly_white")
            fig.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.bar(df_missing["Column"], df_missing["MissingRows"])
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Column")
            plt.ylabel("Number of Missing Rows")
            plt.title(f"Missing Rows per Column - {self.file_name}")
            plt.tight_layout()
            plt.show()

        return df_missing
    
    
    def missing_per_column_rate_bar(self, advanced=True):
        """
        Generate a bar chart showing how many rows are missing each of the key columns.
        """
        cols = ['Name', 'Operator', 'State', 'City', 'Power (MW)',
                'Whitespace (sqft)', 'Type', 'UPS', 'Cooling System', 'MRC']

        missing_counts = self.data[cols].isna().sum().sort_values(ascending=False)
        df_missing = missing_counts.reset_index()
        df_missing.columns = ["Column", "MissingRows"]
        
        num_rows = len(self.data)
        
        df_missing["MissingRows"] = df_missing["MissingRows"]/num_rows
        
        if advanced:
            fig = px.bar(
                df_missing,
                x="Column",
                y="MissingRows",
                title=f"Missing Rows per Column - {self.file_name}",
                labels={"MissingRows": "Number of Missing Rows", "Column": "Data Column"},
            )
            fig.update_layout(xaxis_tickangle=-45, template="plotly_white")
            fig.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.bar(df_missing["Column"], df_missing["MissingRows"])
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Column")
            plt.ylabel("Number of Missing Rows")
            plt.title(f"Missing percentage per Column - {self.file_name}")
            plt.tight_layout()
            plt.show()

        return df_missing


    def regional_missingness_bar(self, element, region_col="State", advanced=True):
        """
        Generate a bar chart showing, by region, how many data centers are missing
        a given element (limited to the key columns).
        """
        cols = ['Name', 'Operator', 'State', 'City', 'Power (MW)',
                'Whitespace (sqft)', 'Type', 'UPS', 'Cooling System', 'MRC']

        if element not in cols:
            raise ValueError(f"Column '{element}' is not in the key column list.")
        if region_col not in self.data.columns:
            raise ValueError(f"Region column '{region_col}' not found in dataset.")
        
        

        df = self.data[cols].copy()
        df["IsMissing"] = df[element].isna()
        regional_missing = df.groupby(region_col)["IsMissing"].sum().sort_values(ascending=False).reset_index()
        regional_missing.columns = [region_col, "MissingCount"]

        if advanced:
            fig = px.bar(
                regional_missing,
                x=region_col,
                y="MissingCount",
                title=f"Missing '{element}' by {region_col} - {self.file_name}",
                labels={"MissingCount": "Number Missing", region_col: region_col},
            )
            fig.update_layout(xaxis_tickangle=-45, template="plotly_white")
            fig.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.bar(regional_missing[region_col], regional_missing["MissingCount"])
            plt.xticks(rotation=45, ha="right")
            plt.xlabel(region_col)
            plt.ylabel("Number Missing")
            plt.title(f"Missing '{element}' by {region_col} - {self.file_name}")
            plt.tight_layout()
            plt.show()

        return regional_missing
    
    def regional_missingness_rate_bar(self, element, region_col="State", advanced=True):
        """
        Generate a bar chart showing, by region, the FRACTION of rows missing
        a given element (scaled by total entries per region).
        """
        cols = ['Name', 'Operator', 'State', 'City', 'Power (MW)',
                'Whitespace (sqft)', 'Type', 'UPS', 'Cooling System', 'MRC']

        if element not in cols:
            raise ValueError(f"Column '{element}' is not in the key column list.")
        if region_col not in self.data.columns:
            raise ValueError(f"Region column '{region_col}' not found in dataset.")

        df = self.data[cols].copy()

        # Compute missing flag
        df["IsMissing"] = df[element].isna()

        # Group by region
        regional_stats = df.groupby(region_col).agg(
            MissingCount=("IsMissing", "sum"),
            TotalCount=("IsMissing", "size")
        ).reset_index()

        # Compute fraction missing
        regional_stats["MissingRate"] = regional_stats["MissingCount"] / regional_stats["TotalCount"]
        regional_stats = regional_stats.sort_values("MissingRate", ascending=False)

        if advanced:
            fig = px.bar(
                regional_stats,
                x=region_col,
                y="MissingRate",
                title=f"Fraction Missing '{element}' by {region_col} - {self.file_name}",
                labels={"MissingRate": "Fraction Missing", region_col: region_col},
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                template="plotly_white",
                yaxis_tickformat=".0%",
            )
            fig.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.bar(regional_stats[region_col], regional_stats["MissingRate"])
            plt.xticks(rotation=45, ha="right")
            plt.xlabel(region_col)
            plt.ylabel("Fraction Missing")
            plt.title(f"Fraction Missing '{element}' by {region_col} - {self.file_name}")
            plt.tight_layout()
            plt.show()

        return regional_stats



class DataCentersMI(DataCenters):
    def __init__(self, file_source,delay_clean=False):
        super().__init__(file_source,delay_clean)
        self.scaler = None
        self.imputations = None
        self.impute_cols = [
            "Power (MW)", "Whitespace (sqft)", "MRC",
            "UPS_Encoded", "Cooling_System_Encoded", "Type_Encoded",
            "Operator_Encoded", "State_Encoded", "City_Encoded"
        ]
        
    def normalize_features(self, columns=None, method="zscore", reverse=False):
        """
        Normalize numeric features in the dataset or reverse normalization.
        Only normalizes columns that will be imputed (self.impute_cols) to avoid mismatch.
        """
        df = self.data.copy()

        # If not specified, normalize the columns used for imputation
        if columns is None:
            columns = getattr(self, "impute_cols", None)
            if columns is None:
                raise ValueError("No columns specified and no impute_cols set.")

        if not reverse:
            # Fit scaler only on these columns
            if method == "zscore":
                self.scaler = StandardScaler()
            elif method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError("method must be 'zscore' or 'minmax'")

            df[columns] = self.scaler.fit_transform(df[columns])
            self.scaler_columns = columns  # store columns scaler was fit on
        else:
            # Reverse-transform only the same columns the scaler was fit on
            df[self.scaler_columns] = self.scaler.inverse_transform(df[self.scaler_columns])

            # Reverse-transform imputations
            if self.imputations is not None:
                for imp_index in range(len(self.imputations)):
                    # Only transform columns present in imputed dataframe
                    cols_in_imp = [c for c in self.scaler_columns if c in self.imputations[imp_index].columns]
                    self.imputations[imp_index][cols_in_imp] = self.scaler.inverse_transform(
                        self.imputations[imp_index][cols_in_imp]
                    )

        self.data = df
        return df



    def multiple_imputation(self, m=5, max_iter=20, random_state=13):
        """
        Perform Multiple Imputation using IterativeImputer (MICE).
        
        Args:
            m (int): number of imputed datasets to generate.
            max_iter (int): max iterations for each imputation run.
            random_state (int): random seed.
        
        Returns:
            imputations (list of pd.DataFrame): list of completed datasets.
        """
        # Select columns to impute (exclude identifiers)
        self.impute_cols = [
            "Power (MW)", "Whitespace (sqft)", "MRC",
            "UPS_Encoded", "Cooling_System_Encoded", "Type_Encoded",
            "Operator_Encoded", "State_Encoded", "City_Encoded"
        ]
        df = self.data[self.impute_cols].copy()

        imputations = []
        for i in range(m):
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=max_iter,
                random_state=random_state + i,
                sample_posterior=True
            )
            imputed_array = imputer.fit_transform(df)
            imputed_df = pd.DataFrame(imputed_array, columns=self.impute_cols)
            imputations.append(imputed_df)
            
        self.imputations = imputations
        
        return imputations

    def pooled_regression(self, imputations, X_cols, Y_col):
        """
        Fit regression model across multiple imputations and pool with Rubin's rules.
        """
        estimates, variances = [], []
        
        for imp in imputations:
            X = imp[X_cols].values
            y = imp[Y_col].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            estimates.append(model.coef_)
            # Variance of residuals as within-imputation variance proxy
            resid_var = np.var(y - y_pred)
            variances.append(resid_var)

        estimates = np.array(estimates)
        variances = np.array(variances)

        # Rubin’s rules
        q_bar = estimates.mean(axis=0)              # pooled coefficients
        u_bar = variances.mean()                    # within-imputation variance
        b = estimates.var(axis=0, ddof=1).mean()    # between-imputation variance
        t = u_bar + (1 + 1/len(imputations)) * b    # total variance

        results = {
            "pooled_coefficients": q_bar,
            "within_var": u_bar,
            "between_var": b,
            "total_var": t
        }

        return results

    
    
if __name__ == "__main__":
    np.set_printoptions(precision=4)
    
    FD = DataCenters("data/DCS_Full.csv")
    print(FD.randomness_of_missing("Power (MW)"))
    print(FD.randomness_of_missing("Whitespace (sqft)"))
    print(FD.randomness_of_missing("MRC"))
    print(FD.randomness_of_missing("UPS_Encoded"))
    print(FD.randomness_of_missing("Cooling_System_Encoded"))
    
    # model, results = FD.linear_regression(["Whitespace (sqft)","MRC",'UPS_Encoded','Cooling_System_Encoded'],"Power (MW)")
    # print(model)
    # print(results)
    
    # model, results = FD.linear_regression(["MRC"],"Power (MW)")
    # print(model)
    # print(results)
    
