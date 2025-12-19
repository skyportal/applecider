from pathlib import Path
from applecider.preprocessing_utils.preprocess_multimodal import Config, build_all_preprocessed, compute_feature_stats_safe, make_splits_from_manifest

def preprocess_data(raw_path, spec_path, output_path):
    """Process raw data and save preprocessed data to output path.

    Parameters
    ----------
    raw_filepath : `str`
        Path to the raw data directory.
    spec_filepath : `str`
        Path to the obj_spectra_info.csv file.
    output_filepath : `str`
        Path to save the preprocessed data.
    """
    cfg = Config(
        data_dir=Path(raw_path), # path to folder containing raw files
        spec_csv=Path(spec_path), # path to obj_spectra_info.csv
        output_root=Path(output_path), # output path
        delta_t_hours=12.0,                # window for light curve per-filter merging 
        alert_tol_days=0.5,                # window for searching for cutouts + meta corresponding to a point in LC
        require_all_3_cuts=True,
        allow_fallback_nearest_any=True,   # set False to disable safety 
        max_nearest_any_dt_days=None,      # use ~5.0 to cap nearest-any reach
        spectrum_wave_min=4500.0,
        spectrum_wave_max=7980.0,          # wavelenght range for spectra apodization
        spectrum_step=1.0,                 # interpol. step
        random_seed=1337,
    )
    build_all_preprocessed(cfg)
    make_splits_from_manifest(
        cfg.output_root/'built_all.csv',
        out_root=cfg.output_root,
        min_per_class=7, # minimum number of obj_ids per class to get sucessful stratified splits
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        seed=cfg.random_seed, strict_stratify=True
    )
    compute_feature_stats_safe(cfg.output_root/'manifest_train.csv', 'event', cfg.output_root) # stats for the continuous light curve features
    compute_feature_stats_safe(cfg.output_root/'manifest_train.csv', 'meta',  cfg.output_root) # stats for metadata columns

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess raw data for AppleCider.")
    parser.add_argument("--raw_path", type=str, required=True, help="Path to the raw data directory.")
    parser.add_argument("--spec_path", type=str, required=True, help="Path to the obj_spectra_info.csv file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the preprocessed data.")

    args = parser.parse_args()

    preprocess_data(args.raw_path, args.spec_path, args.output_path)
