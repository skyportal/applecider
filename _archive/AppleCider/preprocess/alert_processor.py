import gzip
import io
import os
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning


class AlertProcessor:
    """☆ procces object's alert package ☆ (see arXiv:1902.02227 for more info)"""

    @staticmethod
    def get_alerts(base_path, obj_id):
        return np.load(os.path.join(base_path, obj_id, "alerts.npy"), allow_pickle=True)

    @staticmethod
    def process_image(data, normalize=True):
        """returns processed image as a 63x63 np array"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyWarning)
            warnings.simplefilter("ignore")
            with gzip.open(io.BytesIO(data), "rb") as f:
                image = np.nan_to_num(fits.open(io.BytesIO(f.read()), ignore_missing_end=True)[0].data)
        if normalize:
            norm = np.linalg.norm(image)
            if norm != 0:
                image /= norm
        return np.pad(image, [(0, 63 - s) for s in image.shape], mode="constant", constant_values=1e-9)[
            :63, :63
        ]

    @staticmethod
    def process_alert(alert):
        """process metadata, images from alerts"""
        metadata = alert["candidate"]
        metadata_df = pd.DataFrame([metadata])
        metadata_df["obj_id"] = alert["objectId"]

        cutout_dict = {
            cutout: AlertProcessor.process_image(alert[f"cutout{cutout.capitalize()}"]["stampData"])
            for cutout in ("science", "template", "difference")
        }
        assembled_image = np.zeros((63, 63, 3))
        assembled_image[:, :, 0] = cutout_dict["science"]
        assembled_image[:, :, 1] = cutout_dict["template"]
        assembled_image[:, :, 2] = cutout_dict["difference"]

        return metadata_df, assembled_image

    @staticmethod
    def get_process_alerts(obj_id, base_path):
        alerts = AlertProcessor.get_alerts(base_path, obj_id)
        metadata_list = []
        images = []

        for alert in alerts:
            metadata_df, image = AlertProcessor.process_alert(alert)
            metadata_list.append(metadata_df)
            images.append(image)

        return pd.concat(metadata_list, ignore_index=True), images

    @staticmethod
    def select_alerts(data, max_alerts=6):
        """sample from maximum of XYZ alerts"""

        def sample_alerts(alerts):
            num_alerts = len(alerts)
            if num_alerts <= max_alerts:
                return alerts
            selected_alerts = [alerts[0], alerts[-1]]
            if num_alerts > 2:
                step = (num_alerts - 2) / (max_alerts - 2)
                selected_alerts += [alerts[int(step * i + 1)] for i in range(max_alerts - 2)]
            return selected_alerts

        data_by_obj_id = {}
        for sample in data:
            obj_id = sample["obj_id"]
            if obj_id not in data_by_obj_id:
                data_by_obj_id[obj_id] = []
            data_by_obj_id[obj_id].append(sample)

        selected_data = []
        for obj_id, alerts in data_by_obj_id.items():
            alerts_sorted = sorted(alerts, key=lambda x: x["alerte"])
            selected_data.extend(sample_alerts(alerts_sorted))

        return selected_data
