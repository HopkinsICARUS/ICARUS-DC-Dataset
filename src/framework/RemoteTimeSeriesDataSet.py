import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Literal
import pandas as pd
from datetime import datetime
import warnings
import pickle


class RemoteTimeSeriesDataSet:
    """Interface for working with large partitioned time series data stored on remote drives.
    
    Data files follow naming convention: <type>_<freq>_<data>_<YYYY>_<MM>_batch_<####>.csv
    Example: da_hrl_lmps_2018_01_batch_0001.csv
    
    Provides DataFrame-like interface without loading entire dataset into memory.
    Now supports aggregation during query with automatic caching of aggregated results.
    """
    
    def __init__(self, base_path: str, data_type: str, freq: str, data_name: str,
                 datetime_col: str = 'datetime_beginning_utc',
                 datetime_format: str = "%m/%d/%Y  %I:%M:%S %p"):
        """Initialize the dataset interface.
        
        Args:
            base_path: Path to the folder containing the CSV files
            data_type: Type identifier (e.g., 'da' for day-ahead)
            freq: Frequency identifier (e.g., 'hrl' for hourly)
            data_name: Data name identifier (e.g., 'lmps')
            datetime_col: Name of the datetime column
            datetime_format: Format string for parsing datetime
        """
        self.base_path = Path(base_path)
        self.cache_location = "src/analysis/lmp/cache"
        self.data_type = data_type
        self.freq = freq
        self.data_name = data_name
        self.datetime_col = datetime_col
        self.datetime_format = datetime_format
        
        if not self.base_path.exists():
            raise ValueError(f"Path does not exist: {base_path}")
        
        # Expected filename pattern
        self.pattern = f"{data_type}_{freq}_{data_name}_{{year}}_{{month}}_batch_{{batch}}.csv"
        self.regex = re.compile(
            f"{re.escape(data_type)}_{re.escape(freq)}_{re.escape(data_name)}_"
            r"(\d{4})_(\d{2})_batch_(\d{4})\.csv"
        )
        
        # Cache for file inventory
        self._file_cache: Optional[List[Dict[str, Any]]] = None
        self._header_cache: Optional[List[str]] = None
        
        # Data cache directory: store caches under the analysis lmp/cache folder
        # so cached pickles live locally with analysis outputs (faster local access).
        repo_root = Path(__file__).resolve().parents[2]
        self.cache_dir = repo_root / 'src' / 'analysis' / 'lmp' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _scan_files(self) -> List[Dict[str, Any]]:
        """Scan directory for matching files and extract metadata."""
        if self._file_cache is not None:
            return self._file_cache
            
        files = []
        for filepath in self.base_path.glob("*.csv"):
            match = self.regex.match(filepath.name)
            if match:
                year, month, batch = match.groups()
                files.append({
                    'path': filepath,
                    'year': int(year),
                    'month': int(month),
                    'batch': int(batch),
                    'filename': filepath.name
                })
        
        # Sort by year, month, batch
        files.sort(key=lambda x: (x['year'], x['month'], x['batch']))
        self._file_cache = files
        return files
    
    def _get_header(self) -> List[str]:
        """Get column names from first file."""
        if self._header_cache is not None:
            return self._header_cache
            
        files = self._scan_files()
        if not files:
            raise ValueError("No matching files found")
        
        # Read header from first file
        first_file = files[0]['path']
        df_sample = pd.read_csv(first_file, nrows=0)
        self._header_cache = df_sample.columns.tolist()
        return self._header_cache
    
    @property
    def columns(self) -> List[str]:
        """Return column names."""
        return self._get_header()
    
    def _get_cache_file(self, frequency: str, start_year: int, end_year: int, 
                        value_col: Optional[str] = None, 
                        group_by: Optional[List[str]] = None,
                        agg_func: Optional[str] = None) -> Path:
        """Generate cache filename for a specific frequency and date range."""
        # Build cache key including aggregation parameters
        cache_key = f'{self.data_type}_{self.freq}_{self.data_name}_{frequency}_{start_year}_{end_year}'
        
        if value_col:
            cache_key += f'_{value_col}'
        if group_by:
            cache_key += f'_{"_".join(sorted(group_by))}'
        if agg_func:
            agg_str = agg_func if isinstance(agg_func, str) else "_".join(sorted(agg_func))
            cache_key += f'_{agg_str}'
            
        return self.cache_dir / f'{cache_key}.pkl'
    
    def _load_from_cache(self, frequency: str, start_year: int, end_year: int, 
                         columns: Optional[List[str]] = None,
                         value_col: Optional[str] = None,
                         group_by: Optional[List[str]] = None,
                         agg_func: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load data from cache if it exists."""
        cache_file = self._get_cache_file(frequency, start_year, end_year, value_col, group_by, agg_func)
        if cache_file.exists():
            try:
                print(f"Loading from cache: {cache_file.name}")
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                
                if columns:
                    # Only return requested columns if they exist
                    cols_to_return = [col for col in columns if col in df.columns]
                    df = df[cols_to_return]
                return df
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}")
                return None
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, frequency: str, start_year: int, end_year: int,
                       value_col: Optional[str] = None,
                       group_by: Optional[List[str]] = None,
                       agg_func: Optional[str] = None) -> None:
        """Save data to cache."""
        cache_file = self._get_cache_file(frequency, start_year, end_year, value_col, group_by, agg_func)
        try:
            print(f"Saving to cache: {cache_file.name}")
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")
    
    def query(self,
              year: Optional[Union[int, List[int]]] = None,
              month: Optional[Union[int, List[int]]] = None,
              batch: Optional[Union[int, List[int]]] = None,
              columns: Optional[List[str]] = None,
              frequency: Optional[Literal['hourly', 'daily', 'weekly', 'monthly', 'yearly']] = None,
              value_col: Optional[str] = None,
              group_by: Optional[List[str]] = None,
              agg_func: Union[str, List[str]] = 'mean',
              use_cache: bool = True,
              chunksize: Optional[int] = None,
              **pandas_kwargs) -> pd.DataFrame:
        """Query data with optional aggregation and automatic caching.
        
        Args:
            year: Year(s) to filter (int or list of ints)
            month: Month(s) to filter (int or list of ints)
            batch: Batch(es) to filter (int or list of ints)
            columns: Specific columns to load (for non-aggregated queries)
            frequency: Aggregation frequency ('hourly', 'daily', 'weekly', 'monthly', 'yearly')
                      If None, returns raw data without aggregation
            value_col: Column to aggregate (required if frequency is specified)
            group_by: Additional columns to group by during aggregation
            agg_func: Aggregation function(s) - 'mean', 'sum', 'min', 'max', 'std', or list
            use_cache: Whether to use cache (True) or force disk read (False)
            chunksize: For raw queries, if provided, return iterator yielding DataFrames
            **pandas_kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            DataFrame with data (aggregated if frequency specified), loaded from cache if available
        """
        # Determine year range for caching
        start_year = year if isinstance(year, int) else (min(year) if year else None)
        end_year = year if isinstance(year, int) else (max(year) if year else None)
        
        # If aggregation is requested
        if frequency is not None:
            if value_col is None:
                raise ValueError("value_col must be specified when frequency is provided")
            
            if start_year is None or end_year is None:
                raise ValueError("year must be specified when using aggregation")
            
            # Try loading from cache first
            if use_cache:
                cached_df = self._load_from_cache(
                    frequency, start_year, end_year, 
                    columns=None, value_col=value_col, 
                    group_by=group_by, agg_func=agg_func
                )
                if cached_df is not None:
                    return cached_df
            
            # Perform aggregation
            print(f"Aggregating {value_col} by {frequency}...")
            df_aggregated = self._aggregate_query(
                year=year, month=month, batch=batch,
                value_col=value_col, frequency=frequency,
                group_by=group_by, agg_func=agg_func,
                chunksize=chunksize or 100000
            )
            
            # Cache the aggregated result
            if use_cache and not df_aggregated.empty:
                self._save_to_cache(
                    df_aggregated, frequency, start_year, end_year,
                    value_col=value_col, group_by=group_by, agg_func=agg_func
                )
            
            return df_aggregated
        
        # Raw data query (no aggregation)
        else:
            # For raw queries, still try to use cache if years are specified
            if use_cache and start_year is not None and end_year is not None:
                cached_df = self._load_from_cache(
                    'raw', start_year, end_year, columns=columns
                )
                if cached_df is not None:
                    return cached_df
            
            # Load raw data
            df = self._query_raw(
                year=year, month=month, batch=batch,
                columns=columns, chunksize=chunksize,
                **pandas_kwargs
            )
            
            # Cache raw data if requested and years are specified
            if use_cache and start_year is not None and end_year is not None and not isinstance(df, pd.io.parsers.TextFileReader):
                if not df.empty:
                    self._save_to_cache(df, 'raw', start_year, end_year)
            
            return df
    
    def _query_raw(self, 
                   year: Optional[Union[int, List[int]]] = None,
                   month: Optional[Union[int, List[int]]] = None,
                   batch: Optional[Union[int, List[int]]] = None,
                   columns: Optional[List[str]] = None,
                   chunksize: Optional[int] = None,
                   **pandas_kwargs) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """Query raw data without aggregation."""
        files = self._scan_files()
        
        # Filter files
        if year is not None:
            years = [year] if isinstance(year, int) else year
            files = [f for f in files if f['year'] in years]
        
        if month is not None:
            months = [month] if isinstance(month, int) else month
            files = [f for f in files if f['month'] in months]
            
        if batch is not None:
            batches = [batch] if isinstance(batch, int) else batch
            files = [f for f in files if f['batch'] in batches]
        
        if not files:
            warnings.warn("No files match the query criteria")
            return pd.DataFrame(columns=columns or self.columns)
        
        # Load and concatenate
        usecols = columns if columns else None
        
        if chunksize is not None:
            # Return iterator
            def file_iterator():
                for file_info in files:
                    path = file_info['path']
                    print(f"Reading file: {path.name}")
                    try:
                        for chunk in pd.read_csv(
                            path,
                            usecols=usecols,
                            chunksize=chunksize,
                            encoding='utf-8',
                            on_bad_lines='skip',
                            **pandas_kwargs
                        ):
                            yield chunk
                    except UnicodeDecodeError:
                        warnings.warn(f"⚠️ Encoding issue in {path}, retrying with 'latin1'...")
                        try:
                            for chunk in pd.read_csv(
                                path,
                                usecols=usecols,
                                chunksize=chunksize,
                                encoding='latin1',
                                on_bad_lines='skip',
                                **pandas_kwargs
                            ):
                                yield chunk
                        except Exception as e:
                            print(f"❌ Skipping file: {path.name}")
                            print(f"   Error: {repr(e)}")
                            continue
                    except Exception as e:
                        print(f"❌ Error reading {path.name}: {repr(e)}")
                        continue

            return file_iterator()
        else:
            # Load all matching files with error handling
            dfs = []
            for file_info in files:
                path = file_info['path']
                try:
                    df = pd.read_csv(
                        path,
                        usecols=usecols,
                        encoding='utf-8',
                        on_bad_lines='skip',
                        **pandas_kwargs
                    )
                    dfs.append(df)
                except UnicodeDecodeError:
                    # Try latin1 encoding as fallback
                    try:
                        df = pd.read_csv(
                            path,
                            usecols=usecols,
                            encoding='latin1',
                            on_bad_lines='skip',
                            **pandas_kwargs
                        )
                        dfs.append(df)
                    except Exception as e:
                        # Skip file that can't be read
                        warnings.warn(f"⚠️ Skipping {path.name}: {repr(e)}")
                        continue
                except Exception as e:
                    # Skip file with other errors
                    warnings.warn(f"⚠️ Skipping {path.name}: {repr(e)}")
                    continue
            
            if not dfs:
                return pd.DataFrame(columns=columns or self.columns)
            
            return pd.concat(dfs, ignore_index=True)
    
    def _aggregate_query(self,
                         value_col: str,
                         frequency: Literal['hourly', 'daily', 'weekly', 'monthly', 'yearly'],
                         year: Optional[Union[int, List[int]]] = None,
                         month: Optional[Union[int, List[int]]] = None,
                         batch: Optional[Union[int, List[int]]] = None,
                         group_by: Optional[List[str]] = None,
                         agg_func: Union[str, List[str]] = 'mean',
                         chunksize: int = 100000) -> pd.DataFrame:
        """Internal method to aggregate data during query."""
        
        # Determine columns to load
        load_cols = [self.datetime_col, value_col]
        if group_by:
            load_cols.extend([col for col in group_by if col not in load_cols])
        
        # Frequency mapping for pandas resampling
        freq_map = {
            'hourly': 'H',
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'yearly': 'Y'
        }
        
        if frequency not in freq_map:
            raise ValueError(f"Invalid frequency: {frequency}. Must be one of {list(freq_map.keys())}")
        
        pandas_freq = freq_map[frequency]
        
        # Process data in chunks and accumulate aggregations
        aggregated_chunks = []
        
        for chunk in self._query_raw(year=year, month=month, batch=batch, 
                                     columns=load_cols, chunksize=chunksize):
            try:
                # Parse datetime
                chunk[self.datetime_col] = chunk[self.datetime_col].astype(str)
                chunk[self.datetime_col] = chunk[self.datetime_col].str.replace(
                    r'[\x00-\x1f\x7f-\x9f]', ' ', regex=True
                )
                
                try:
                    chunk['datetime_parsed'] = pd.to_datetime(
                        chunk[self.datetime_col],
                        format=self.datetime_format,
                        errors='coerce'
                    )
                except Exception:
                    chunk['datetime_parsed'] = pd.to_datetime(
                        chunk[self.datetime_col],
                        errors='coerce'
                    )
                
                # Drop rows with invalid dates
                chunk = chunk.dropna(subset=['datetime_parsed'])
                
                if chunk.empty:
                    continue
                
                # Set datetime as index for resampling
                chunk = chunk.set_index('datetime_parsed')
                
                # Group and resample
                if group_by:
                    # Group by additional columns, then resample
                    grouped = chunk.groupby(group_by)
                    resampled = grouped.resample(pandas_freq)[value_col].agg(agg_func)
                else:
                    # Just resample
                    resampled = chunk.resample(pandas_freq)[value_col].agg(agg_func)
                
                aggregated_chunks.append(resampled.reset_index())
                
            except Exception as e:
                print(f"⚠️ Error processing chunk: {repr(e)}")
                continue
        
        if not aggregated_chunks:
            raise ValueError("No valid data found for aggregation")
        
        # Combine all chunks
        print("Combining aggregated chunks...")
        df_combined = pd.concat(aggregated_chunks, ignore_index=True)
        
        # Final aggregation to handle overlapping time periods from different chunks
        agg_col = value_col if isinstance(agg_func, str) else f"{value_col}"
        
        if group_by:
            final_agg = df_combined.groupby(group_by + ['datetime_parsed']).agg({
                agg_col: 'mean'
            }).reset_index()
        else:
            final_agg = df_combined.groupby('datetime_parsed').agg({
                agg_col: 'mean'
            }).reset_index()
        
        # Rename datetime column based on frequency
        final_agg = final_agg.rename(columns={'datetime_parsed': f'{frequency}_timestamp'})

        # Add year/week/month columns for downstream compatibility
        ts_col = f'{frequency}_timestamp'
        if ts_col in final_agg.columns:
            final_agg[ts_col] = pd.to_datetime(final_agg[ts_col])
            final_agg['year'] = final_agg[ts_col].dt.year
            if frequency == 'weekly':
                # ISO week number
                try:
                    final_agg['week'] = final_agg[ts_col].dt.isocalendar().week
                except Exception:
                    final_agg['week'] = final_agg[ts_col].dt.week
            else:
                final_agg['month'] = final_agg[ts_col].dt.month
        
        print(f"Aggregation complete: {len(final_agg):,} {frequency} records")
        
        return final_agg
    
    @property
    def files(self) -> pd.DataFrame:
        """Return DataFrame with file inventory."""
        file_list = self._scan_files()
        return pd.DataFrame(file_list)
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Return first n rows from earliest file."""
        files = self._scan_files()
        if not files:
            raise ValueError("No files found")
        return pd.read_csv(files[0]['path'], nrows=n)
    
    def tail(self, n: int = 5) -> pd.DataFrame:
        """Return last n rows from latest file."""
        files = self._scan_files()
        if not files:
            raise ValueError("No files found")
        df = pd.read_csv(files[-1]['path'])
        return df.tail(n)
    
    def sample(self, n: int = 5, random_state: Optional[int] = None) -> pd.DataFrame:
        """Return random sample of n rows from a random file."""
        files = self._scan_files()
        if not files:
            raise ValueError("No files found")
        
        import random
        if random_state is not None:
            random.seed(random_state)
        
        random_file = random.choice(files)
        df = pd.read_csv(random_file['path'])
        return df.sample(n=min(n, len(df)), random_state=random_state)
    
    def get_date_range(self) -> tuple:
        """Return (earliest_year, earliest_month, latest_year, latest_month)."""
        files = self._scan_files()
        if not files:
            return None
        
        first = files[0]
        last = files[-1]
        return (first['year'], first['month'], last['year'], last['month'])
    
    def __repr__(self) -> str:
        files = self._scan_files()
        date_range = self.get_date_range()
        
        if not files:
            return f"RemoteTimeSeriesDataSet('{self.data_type}_{self.freq}_{self.data_name}', no files found)"
        
        return (f"RemoteTimeSeriesDataSet(\n"
                f"  type='{self.data_type}_{self.freq}_{self.data_name}',\n"
                f"  files={len(files)},\n"
                f"  date_range={date_range[0]}/{date_range[1]:02d} to {date_range[2]}/{date_range[3]:02d},\n"
                f"  columns={len(self.columns)}\n"
                f")")
    
    def __len__(self) -> int:
        """Return total number of files."""
        return len(self._scan_files())


if __name__ == '__main__':
    # Simple local test: instantiate dataset and aggregate 2018 weekly
    base_path = Path('C:/Users/OwenJ/OneDrive/Desktop/ROSEI/ICARUS-DC-Dataset/raw_data/capacity')
    ds = RemoteTimeSeriesDataSet(str(base_path), data_type='day', freq='hrl', data_name='capacity')
    print(f"Found {len(ds)} files matching pattern. Example columns: {ds.columns[:8]}")
    df = ds.query(
        year=range(2012,2024),
        frequency='weekly',
        value_col='total_committed',
        agg_func='mean',
        use_cache=False
    )
    
    print(df.head())
    
    
    
    exit()
    # Simple remote test: instantiate dataset and aggregate 2018 weekly
    try:
        print("Running RemoteTimeSeriesDataSet quick test: weekly 2018 aggregation")
        base_path = Path('D:/CSVs_dalmps')
        

        print(f"Using base_path: {base_path}")

        # Create dataset object
        ds = RemoteTimeSeriesDataSet(str(base_path), data_type='da', freq='hrl', data_name='lmps')

        # Show inventory
        print(f"Found {len(ds)} files matching pattern. Example columns: {ds.columns[:8]}")

        # Query with aggregation - now aggregation happens in query()
        df = ds.query(
            year=[2018],
            frequency='weekly',
            value_col='total_lmp_da',
            group_by=['zone'],
            agg_func='mean',
            use_cache=True
        )

        if df is None or df.empty:
            print("No data returned for 2018 — check raw CSVs in the base_path or adjust naming conventions.")
        else:
            print(f"Loaded aggregated dataframe with {len(df):,} rows. Columns: {list(df.columns)}")
            print(f"Sample data:\n{df.head(3)}")

        print(f"Cache directory used: {ds.cache_dir}")
    except Exception as e:
        print(f"RemoteTimeSeriesDataSet quick test failed: {e}")