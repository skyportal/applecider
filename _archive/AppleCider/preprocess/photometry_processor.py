import os

import pandas as pd
from sklearn.preprocessing import StandardScaler


class PhotometryProcessor:

    """☆ procces object's photometry, metadata"""

    @staticmethod
    def clean_photometry(df, df_type):
        """cleans photometry dataframe"""
        df = PhotometryProcessor.clean_dataframe(df)
        df["type"] = df_type[df_type["obj_id"] == df["obj_id"].iloc[0]]["type"].values[0]
        df.dropna(subset=["mag", "magerr"], inplace=True)
        return df.reset_index(drop=True)

    @staticmethod
    def clean_dataframe(df):
        """renames columns, converts jd to MJD"""
        df = df.rename(
            columns={
                "magpsf": "mag",
                "sigmapsf": "magerr",
                "fid": "filter",
                "scorr": "snr",
                "diffmaglim": "limiting_mag",
            }
        )
        df["filter"] = df["filter"].replace({1: "ztfg", 2: "ztfr", 3: "ztfi"})
        df["mjd"] = df["jd"] - 2400000.5
        df = df[["obj_id", "jd", "mjd", "mag", "magerr", "snr", "limiting_mag", "filter"]]
        return df

    @staticmethod
    def process_csv(object_id, df_bts, base_path):
        """creates file path for photometry.csv, cleans photometry"""
        file_path = os.path.join(base_path, object_id, "photometry.csv")
        return (
            PhotometryProcessor.clean_photometry(pd.read_csv(file_path), df_bts)
            if os.path.exists(file_path)
            else pd.DataFrame()
        )

    @staticmethod
    def get_first_valid_index(df, min_points=1):
        """counts occurences of each filter, finds index that meets minimum number of points in each filter"""
        filter_counts = {"ztfr": 0, "ztfg": 0, "ztfi": 0}
        for i in range(len(df)):
            current_filter = df["filter"].iloc[i]
            if current_filter in filter_counts:
                filter_counts[current_filter] += 1
                if filter_counts[current_filter] >= min_points:
                    return i
        return -1

    @staticmethod
    # formerly "add_metadata_to_photometry"
    # add alerts to photometry
    def add_alert_to_photometry(photo_df, alert_df):
        """cleans "metadata", merges photometry_df with alert_df"""

        metadata_df_copy = PhotometryProcessor.clean_dataframe(alert_df.copy())

        photo_df["source"] = "photometry.csv"
        metadata_df_copy["source"] = "alerts.npy"

        df = pd.merge(
            photo_df,
            metadata_df_copy,
            on=["obj_id", "jd", "mjd", "mag", "magerr", "snr", "limiting_mag", "filter", "source"],
            how="outer",
            suffixes=("", "_metadata"),
        )
        df = df[["obj_id", "jd", "mjd", "mag", "magerr", "snr", "limiting_mag", "filter", "type", "source"]]
        df["obj_id"] = df["obj_id"].ffill().bfill()
        df["type"] = df["type"].ffill().bfill()
        df = df.drop_duplicates(subset=["mjd", "filter"], keep="first")
        df = df.sort_values(by=["mjd"])
        df.reset_index(drop=True, inplace=True)
        return df

    def find_valid_alert_index(df):
        for index, row in df.iterrows():
            if row["flux_ztfg"] == 0 and row["flux_ztfr"] == 0 and row["flux_ztfi"] == 0:
                return index - 1

        return len(df)

    def normalize_light_curve(df):
        flux_data = df.loc[: len(df), ["flux_ztfg", "flux_ztfr", "flux_ztfi"]]
        scaler = StandardScaler()
        normalized_flux = scaler.fit_transform(flux_data)
        df.loc[: len(df), ["flux_ztfg", "flux_ztfr", "flux_ztfi"]] = normalized_flux

        return df
