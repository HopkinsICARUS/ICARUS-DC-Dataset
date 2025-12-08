import http.client
import urllib.parse
import json
import pandas as pd
from datetime import datetime, timedelta

class PJMDataFetcher:
    """
    Fetch PJM Actual/Schedule Summary data over a specified time range
    and save it as CSV.
    """

    BASE_URL = "api.pjm.com"
    FEED_PATH = "/api/v1/act_sch_interchange"

    def __init__(self, subscription_key: str):
        self.headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    def fetch_data(self, start: str, end: str, frequency: str = "hourly") -> pd.DataFrame:
        """
        Fetch PJM data for a given time range and sampling frequency.
        Args:
            start (str): Start datetime in YYYY-MM-DD format
            end (str): End datetime in YYYY-MM-DD format
            frequency (str): "hourly", "daily", "weekly", "monthly"
        Returns:
            pd.DataFrame: DataFrame with requested data
        """
        # Map human frequency to PJM rowCount and filtering
        freq_map = {
            "hourly": 1,
            "daily": 24,
            "weekly": 24 * 7,
            "monthly": 24 * 30
        }
        if frequency not in freq_map:
            raise ValueError("frequency must be one of: hourly, daily, weekly, monthly")

        # Build the request body
        body_json = {
            "rowCount": 10000,  # maximum per request, can page later if needed
            "startRow": 1,
            "filters": [
                {"datetime_beginning_utc": start},
                {"datetime_ending_utc": end}
            ],
            "sort": "datetime_beginning_utc",
            "order": 0
        }
        body = json.dumps(body_json)

        try:
            conn = http.client.HTTPSConnection(self.BASE_URL)
            conn.request("GET", f"{self.FEED_PATH}?{urllib.parse.urlencode({})}", body, self.headers)
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {response.reason}")
            data = response.read()
            conn.close()

            # Convert to DataFrame
            data_json = json.loads(data)
            if "items" in data_json:
                data_list = data_json["items"]
            else:
                raise RuntimeError("Unexpected JSON structure from PJM API")

            # Keep only selected columns
            cols = ["datetime_beginning_utc", "tie_line", "actual_flow", "sched_flow", "inadv_flow"]
            df = pd.DataFrame(data_list)[cols]
            
            print(df.head)

            # Resample according to frequency
            if not df.empty:
                df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'])
                df.set_index('datetime_beginning_utc', inplace=True)

                if frequency != "hourly":
                    rule_map = {"daily": "D", "weekly": "W", "monthly": "M"}
                    df = df.resample(rule_map[frequency]).mean()

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to fetch PJM data: {e}")

    def save_csv(self, df: pd.DataFrame, filename: str):
        df.to_csv(filename)
        print(f"Saved data to {filename}")


# Example usage
if __name__ == "__main__":
    fetcher = PJMDataFetcher(subscription_key="ae2bb21b5122412f9a20914426a49d5c")
    df = fetcher.fetch_data("2018-12-01", "2025-12-31", frequency="hourly")
    fetcher.save_csv(df, "raw_data/DM2_data/pjm_oct2025_daily.csv")
