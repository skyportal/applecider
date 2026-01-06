import os
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing

from AppleCider.preprocess.alert_processor import AlertProcessor
from AppleCider.preprocess.photometry_processor import PhotometryProcessor
from AppleCider.preprocess.data_preprocessor import SpectraProcessor
from AppleCider.preprocess.data_preprocessor import DataPreprocessor


class TransientDataset():
    
    def __init__(self, preprocessed_path, df_bts=None, base_path=None, max_mjd=None, normalize_light_curve=False, include_spectra=False, include_flux_err=False):
        
        self.preprocessed_path = preprocessed_path
        self.df_bts = df_bts
        self.base_path = base_path
        self.data = []
        self.data_preprocess = []
        self.max_mjd = max_mjd
        self.normalize_light_curve = normalize_light_curve
        self.include_spectra = include_spectra
        self.include_flux_err = include_flux_err

    def preprocess_data(self, df_bts, base_path, max_mjd):
        ''' preprocess photometry, metadata, images  by creating dictionary for each object alert sample'''
        
        self.df_bts, self.data_preprocess, self.base_path, self.max_mjd = df_bts, [], base_path, max_mjd
                 
        for idx, row in tqdm(df_bts.iterrows(), total=df_bts.shape[0], desc="Loading data", leave=True):
            try:
                obj_id, target = row['obj_id'], row['type']
                if any(obj_id in file for file in os.listdir(self.preprocessed_path)):
                    continue
                
                # Get photometry, metadata, images
                photo_df, alert_df, images = PhotometryProcessor.process_csv(obj_id, df_bts, base_path), *AlertProcessor.get_process_alerts(obj_id, base_path)
                photo_df, alert_df = photo_df.sort_values(by='jd'), alert_df.sort_values(by='jd')
                photo_df = PhotometryProcessor.add_alert_to_photometry(photo_df, alert_df)
                
                # Convert magnitude to flux, flux error
                photo_df = DataPreprocessor.convert_photometry(photo_df)

                # Cut metadata to max_mjd
                max_ = min(photo_df['mjd'].max(), max_mjd)
                photo_df = photo_df[photo_df['mjd'] <= max_]
                alert_df = alert_df[alert_df['jd'] <= photo_df['jd'].max()]
                
                if len(alert_df) == 0:
                    print(f"Metadata unavailable at max_mjd = {max_mjd}. No alert saved for {obj_id}.")
                    continue

                metadata_df = DataPreprocessor.preprocess_metadata(alert_df)            
                metadata_df_norm = metadata_df.drop(columns=['jd'])
                
                start_index = PhotometryProcessor.get_first_valid_index(photo_df)
                
                if start_index == -1:
                    print(f"{obj_id} start_index == -1")
                    continue
                
                alert_indices = list(range(start_index, len(metadata_df)))
                for i in alert_indices:
                    photo_ready = DataPreprocessor.cut_photometry(photo_df, metadata_df, i, max_mjd)
                    
                    # Skip saving alert if photometry only has 1 point 
                    if len(photo_ready) <= 1:
                        print(f"{obj_id} Failed photometry requirement after time cuts. Skip {obj_id} at index {i}!")
                        continue
                    if photo_ready is None:
                        print(f"{obj_id} FAILED. BREAK!")
                        break
                    
                    # Get Matching index for metadata, image    
                    get_index = metadata_df_norm.iloc[i].name
                    
                    if self.include_spectra:
                        # Get wavelength, flux from spectra.csv
                        spectra = SpectraProcessor.read_spectra_csv(obj_id, base_path)
                        spectra = SpectraProcessor.preprocess_spectra(spectra)
                        
                        self.data_preprocess.append({
                                'obj_id': obj_id,
                                'alerte': i,
                                'photometry': photo_ready,
                                'metadata': metadata_df_norm.iloc[i],
                                'images': images[get_index],
                                'spectra': spectra,
                                'target': target})
                    
                    else:
                        self.data_preprocess.append({
                                'obj_id': obj_id,
                                'alerte': i,
                                'photometry': photo_ready,
                                'metadata': metadata_df_norm.iloc[i],
                                'images': images[get_index],
                                'target': target})

            except Exception as e:
                print(f"Error processing {obj_id} at index {idx}: {e}")
                 
    def process_and_save_sample(args):
        ''' save dictionary w/processed photometry, metadata, images to .npy at desired path '''
        
        res_dict = {}
        
        sample, save_dir, include_spectra, include_flux_err, normalize_light_curve = args
        obj_id = sample['obj_id']
        alerte = sample['alerte']  # keep in french <3
        type_obj = sample['target']
        
        photometry = sample['photometry']
        if len(photometry) == 0:
            return

        save_path = os.path.join(save_dir, f"{obj_id}_alert_{alerte}.npy")
        if os.path.exists(save_path):
            return

        res_df = pd.DataFrame()
        photometry = sample['photometry'].pivot_table(index=['mjd'], columns='filter', values=['flux', 'flux_error'])
        photometry = photometry.reset_index()
        photometry.columns = [col[0] if col[0] == 'mjd' else '_'.join(col).strip() for col in photometry.columns.values]
        photometry['obj_id'] = obj_id

        res_df = pd.concat([res_df, photometry])
        res_df = res_df.reset_index(drop=True, inplace=True)

        if include_flux_err:
            columns = ['flux_ztfg', 'flux_error_ztfg', 'flux_ztfr', 'flux_error_ztfr','flux_ztfi', 'flux_error_ztfi']
            for col in columns:
                if col not in photometry.columns:
                    photometry[col] = 0.

            photometry = photometry[['obj_id', 'mjd', 'flux_ztfg', 'flux_error_ztfg', 'flux_ztfr', 'flux_error_ztfr', 'flux_ztfi',  'flux_error_ztfi']]
            photometry = photometry.fillna(0)

            # TODO:
            #if normalize_light_curve:
            #    photometry = PhotometryProcessor.normalize_light_curve(photometry)
            
            # Get date, flux ztfr, flux ztfg, flux_ztfi
            useful_columns = ['mjd', 'flux_ztfg', 'flux_error_ztfg', 'flux_ztfr', 'flux_error_ztfr', 'flux_ztfi', 'flux_error_ztfi']
            
            photometry = photometry[useful_columns].values
        
        else:
            columns = ['flux_ztfg', 'flux_ztfr','flux_ztfi']
            for col in columns:
                if col not in photometry.columns:
                    photometry[col] = 0.
                    
            photometry = photometry[['obj_id', 'mjd', 'flux_ztfg', 'flux_ztfr', 'flux_ztfi']]
            photometry = photometry.fillna(0)
            
            if normalize_light_curve:
                photometry = PhotometryProcessor.normalize_light_curve(photometry)
                
            # Get date, flux ztfr, flux ztfg, flux_ztfi
            useful_columns = ['mjd', 'flux_ztfg', 'flux_ztfr', 'flux_ztfi']
            photometry = photometry[useful_columns].values
        
        
        if include_spectra:
            res_dict.update({
                'obj_id': obj_id,
                'photometry': photometry,
                'metadata': sample['metadata'],
                'images': sample['images'],
                'spectra':sample['spectra'],
                'target': sample['target'],
                'alerte': alerte})
            np.save(save_path, res_dict)
            
        else:
            res_dict.update({
                'obj_id': obj_id,
                'photometry': photometry,
                'metadata': sample['metadata'],
                'images': sample['images'],
                'target': sample['target'],
                'alerte': alerte})
            np.save(save_path, res_dict)
        
    def preprocess_and_save(self):
        os.makedirs(self.preprocessed_path, exist_ok=True)
        
        args = [(sample, self.preprocessed_path, self.include_spectra, self.include_flux_err,  self.normalize_light_curve) for sample in self.data_preprocess]
    
        num_workers = multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(num_workers) as pool:
            list(tqdm(pool.imap(TransientDataset.process_and_save_sample, args), total=len(self.data_preprocess), desc="Saving", leave=True))

    