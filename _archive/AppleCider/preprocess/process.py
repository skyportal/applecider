import sys

sys.path.insert(0, "/applecider")
import pandas as pd
from AppleCider.preprocess.transient_dataset import TransientDataset

data_dir = "/applecider/dataset/"
cider_BTS = pd.read_csv("/applecider/files/cider_BTS.csv")

cider_bts_data_path = "/cider_bts/"


dataset = TransientDataset(
    cider_bts_data_path,
    base_path=data_dir,
    max_mjd=10,
    normalize_light_curve=False,
    include_spectra=True,
    include_flux_err=True,
)

dataset.preprocess_data(cider_BTS, data_dir, max_mjd=5)
dataset.preprocess_and_save()
