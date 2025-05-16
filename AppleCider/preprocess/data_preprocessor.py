import os
import warnings
import numpy as np
import pandas as pd
import multiprocessing
import gzip
import io
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from scipy.interpolate import interp1d
from scipy import stats

from tqdm import tqdm
import pickle
import random
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight



class AlertProcessor:
    """ ☆ procces object's alert package ☆ (see arXiv:1902.02227 for more info) """
    
    @staticmethod
    def get_alerts(base_path, obj_id):
        return np.load(os.path.join(base_path, obj_id, 'alerts.npy'), allow_pickle=True)

    @staticmethod
    def process_image(data, normalize=True):
        ''' returns processed image as a 63x63 np array '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyWarning)
            warnings.simplefilter('ignore')
            with gzip.open(io.BytesIO(data), "rb") as f:
                image = np.nan_to_num(fits.open(io.BytesIO(f.read()), ignore_missing_end=True)[0].data)
        if normalize:
            norm = np.linalg.norm(image)
            if norm != 0:
                image /= norm
        return np.pad(image, [(0, 63 - s) for s in image.shape], mode="constant", constant_values=1e-9)[:63, :63]

    @staticmethod
    def process_alert(alert):
        ''' process metadata, images from alerts '''
        metadata = alert['candidate']
        metadata_df = pd.DataFrame([metadata])
        metadata_df['obj_id'] = alert['objectId']

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
        
        ''' sample from maximum of XYZ alerts '''
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
            obj_id = sample['obj_id']
            if obj_id not in data_by_obj_id:
                data_by_obj_id[obj_id] = []
            data_by_obj_id[obj_id].append(sample)

        selected_data = []
        for obj_id, alerts in data_by_obj_id.items():
            alerts_sorted = sorted(alerts, key=lambda x: x['alerte'])
            selected_data.extend(sample_alerts(alerts_sorted))

        return selected_data



class PhotometryProcessor:
    
    """ ☆ procces object's photometry, metadata """
    
    @staticmethod
    def clean_photometry(df, df_type):
        ''' cleans photometry dataframe '''
        df = PhotometryProcessor.clean_dataframe(df)
        df['type'] = df_type[df_type['obj_id'] == df['obj_id'].iloc[0]]['type'].values[0]
        df.dropna(subset=['mag', 'magerr'], inplace=True)
        return df.reset_index(drop=True)
    
    @staticmethod
    def clean_dataframe(df):
        ''' renames columns, converts jd to MJD  '''
        df = df.rename(columns={
            'magpsf': 'mag',
            'sigmapsf': 'magerr',
            'fid': 'filter',
            'scorr': 'snr',
            'diffmaglim': 'limiting_mag' })
        df['filter'] = df['filter'].replace({1: 'ztfg', 2: 'ztfr', 3: 'ztfi'})
        df['mjd'] = df['jd'] - 2400000.5
        df = df[['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter']]
        return df

    @staticmethod
    def process_csv(object_id, df_bts, base_path):   
        ''' creates file path for photometry.csv, cleans photometry'''
        file_path = os.path.join(base_path, object_id, 'photometry.csv')
        return PhotometryProcessor.clean_photometry(pd.read_csv(file_path), df_bts) if os.path.exists(file_path) else pd.DataFrame()

    @staticmethod
    def get_first_valid_index(df, min_points=1):
        '''counts occurences of each filter, finds index that meets minimum number of points in each filter'''
        filter_counts = {'ztfr': 0, 'ztfg': 0, 'ztfi':0}
        for i in range(len(df)):
            current_filter = df['filter'].iloc[i]
            if current_filter in filter_counts:
                filter_counts[current_filter] += 1
                if filter_counts[current_filter] >= min_points:
                    return i
        return -1

    @staticmethod
    def add_metadata_to_photometry(photo_df, metadata_df):
        ''' cleans "metadata", merges photometry_df with metadata_df'''
        
        metadata_df_copy = PhotometryProcessor.clean_dataframe(metadata_df.copy())
        
        # ... but first, add new column so we always know the source of each row in df
        photo_df['source'] = 'photometry.csv'
        metadata_df_copy['source'] = 'alerts.npy'
        
        df = pd.merge(photo_df, metadata_df_copy, on=['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter', 'source'], how='outer', suffixes=('', '_metadata')) 
        df = df[['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter', 'type', 'source']]
        df['obj_id'] = df['obj_id'].ffill().bfill()
        df['type'] = df['type'].ffill().bfill()
        df = df.drop_duplicates(subset=['mjd', 'filter'], keep='first')
        df = df.sort_values(by=['mjd'])
        df.reset_index(drop=True, inplace=True)
        return df
    
    def find_valid_alert_index(df):
        for index, row in df.iterrows():
            if row['flux_ztfg'] == 0 and row['flux_ztfr'] == 0 and row['flux_ztfi'] == 0:
                return index - 1
        
        return len(df)
    
    def normalize_light_curve(df):

        flux_data = df.loc[:len(df), ['flux_ztfg', 'flux_ztfr', 'flux_ztfi']]
        scaler = StandardScaler() # standardizes by removing mean, scaling to unit variance
        normalized_flux = scaler.fit_transform(flux_data)
        df.loc[:len(df), ['flux_ztfg', 'flux_ztfr','flux_ztfi']] = normalized_flux
  
        return df
    

class SpectraProcessor:
    
    """ ☆ procces object's spectra (not in alerts.npy) ☆ """
    
    @staticmethod
    def get_spectra_df(object_id, base_path):
        ''' for when we want all of the columns in spectra.csv '''
        file_path = os.path.join(base_path, object_id, 'spectra.csv')
        spectra_df = pd.read_csv(file_path)
        return spectra_df
    
    @staticmethod
    def read_spectra_csv(object_id, base_path):
        """ get wavelength, flux from spectra csv """
        file_path = os.path.join(base_path, object_id, 'spectra.csv')
        spectra_df = pd.read_csv(file_path)
        spectra_df = spectra_df[['wavelength', 'flux']]
      
        return spectra_df
    
    @staticmethod
    def preprocess_spectra(spectra):
        """ limit wavelength to 4500 - 7980, interpolate and normalize """
        
        spectra = spectra.to_numpy()
        spectra = spectra.astype(float)
        
        new_wavelength = np.linspace(4500, 7980, 7980 - 4500 + 1)

        # remove nans from flux
        spectra = spectra[~np.isnan(spectra).any(axis=1)]

        f = interp1d(spectra[:, 0], spectra[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
        flux = f(new_wavelength)

        mean = np.mean(flux)
        mad = stats.median_abs_deviation(flux)
        flux = (flux - mean) / mad

        flux = flux.reshape((1, -1))
        flux = flux.astype(np.float32)

        return flux

    
class DataSorter:
    """ ☆ filter out objects w/o SEDM spectra, split train test sets and save alert names ☆ """
    
    def open_pkl(file_path):
        with open(file_path, 'rb') as file:
            files = pickle.load(file)
        return files

    def dump_pkl(file_path, file_2_dump):
        with open(file_path, 'wb') as file:
            pickle.dump(file_2_dump, file)  
    
    def get_obj_wSEDM_spectra(obj_id_list, data_dir):
        ''' ☆ list of object ids that have SEDM spectra ☆ '''
        
        obj_sedm_list = []
        
        for object_id in tqdm(obj_id_list, desc='Checking for SEDM spectra', leave=True):
            spectra_path = os.path.join(data_dir, object_id, 'spectra.csv')
            
            if os.path.isfile(spectra_path):
                spectra_df = pd.read_csv(os.path.join(data_dir, object_id, 'spectra.csv'))
                # check if spectra from Fritz (which has all these columns)
                if {'instrument_name', 'telescope_name', 'data_length'}.issubset(spectra_df.columns):
                    instrument = spectra_df['instrument_name'][0]
                    if instrument == 'SEDM':
                        obj_sedm_list.append(object_id)
    
        return obj_sedm_list

    
    def create_df_of_object_alerts_in_dataset(test_df, train_df, test_data_dir, train_data_dir):
        '''  ☆ creates df for testing and training sets that has the object IDs, alerts for each object ID, classification ☆
        
        Parameters
        ----------
        test_df : Dataframe
            object IDs in testing set
        train_df : dataframe
            object IDs in training set
        test_data_dir_path : 
            where the test set object alerts have been saved to 
        train_data_dir_path : 
            where the training set object alerts have been saved to 
        label col:
            name of the column with the numerical classifications

        Returns
        ----------
        test_data : Dataframe
            bject IDs, names of their alerts, real classification, and numerical classification label
        train_data : Dataframe
            object IDs, names of their alerts, real classification, and numerical classification label
        '''

        # test part
        test_data_files = [f for f in os.listdir(test_data_dir) if f.endswith('.npy')]
        test_data_names = [f.split('_')[0] for f in test_data_files]

        test_data = pd.DataFrame(test_data_names, columns=['name'])
        test_data['file'] = test_data_files
        test_data = test_data.merge(test_df[['obj_id','type']],
                                    left_on='name', right_on='obj_id', how='left')

        test_data = test_data.drop(columns=['obj_id'])
        test_data = test_data.sort_values(by='file')
        test_data = test_data.reset_index(drop=True)
        
        # train part
        train_data_files = [f for f in os.listdir(train_data_dir) if f.endswith('.npy')]
        train_data_names = [f.split('_')[0] for f in train_data_files]

        train_data = pd.DataFrame(train_data_names, columns=['name'])
        train_data['file'] = train_data_files
        train_data = train_data.merge(train_df[['obj_id','type']],
                      left_on='name', right_on='obj_id', how='left')

        train_data = train_data.drop(columns=['obj_id'])
        train_data = train_data.sort_values(by='file')
        train_data = train_data.reset_index(drop=True)
        
        return test_data, train_data


    def split_and_compute_class_weights(df, step, max_samples, file_suffix=None, class_label_str=False, group_labels=False, group_labels_SN=False, save_files=False, save_path = None,split_ratio=0.8, random_seed=42, nb=None, verbose=False):
    
        """ create train files, val files + class weight dictionary and save them
            note: personally, i format names like this train_files_{mode: multi, ztf}_{max_mjd: 10, 30, etc}_{max_samples}              
        """
        
        train_file = f'train_files_{file_suffix}.pkl'
        val_file = f'val_files_{file_suffix}.pkl'
        class_weights_file = f'class_weights_{file_suffix}.pkl'
        
        if group_labels:
            id2target = {0: 'SN I', 1:'SN II', 2:'Cataclysmic', 3:'AGN', 4:'Tidal Disruption Event'}
            target2id = {v: k for k, v in id2target.items()}
            
            group_SN_types = {'SN Ia': 'SN I', 'SN Ic': 'SN I', 'SN Ib': 'SN I', 'SN II': 'SN II', 'SN IIP': 'SN II', 'SN IIn': 'SN II', 'SN IIb': 'SN II'}
            df = df.replace({step: group_SN_types})
            df = df.replace({step: target2id})
            class_weights_file = f'{class_weights_file[:-4]}-group_labels.pkl'
            
        elif group_labels_SN:
            id2target = {0: 'SN', 1:'Cataclysmic', 2:'AGN', 3:'Tidal Disruption Event'}
            target2id = {v: k for k, v in id2target.items()}
            
            group_SN = {'SN Ia': 'SN', 'SN Ic': 'SN', 'SN Ib': 'SN', 'SN II': 'SN', 'SN IIP': 'SN', 'SN IIn': 'SN', 'SN IIb': 'SN'}
            df = df.replace({step:group_SN})
            df = df.replace({step: target2id})
            class_weights_file = f'{class_weights_file[:-4]}-group_labels_SN.pkl'
        else:
            id2target = {0: 'SN Ia', 1:'SN Ic',  2:'SN Ib' , 3: 'SN II', 4:'SN IIP', 5:'SN IIn', 6:'SN IIb', 7:'Cataclysmic', 8:'AGN', 9:'Tidal Disruption Event'}
            target2id = {v: k for k, v in id2target.items()}
            df = df.replace({step: target2id})
        
        
        train_df_list, val_df_list = [], []
        unique_labels = df[step].unique()
        
        ## downsample classes:
        for cls in unique_labels:
            df_cls = df[df[step] == cls]
            df_not_cls = df[df[step] != cls]
            #print(cls)
            if len(df_cls) > max_samples:
                print(f'Down sampled class {cls} from {len(df_cls)} to {max_samples}')
                df_cls_down = df_cls.sample(n=max_samples, random_state=random_seed)
                df = pd.concat([df_not_cls, df_cls_down], ignore_index=True)
        
    
        train_file_path = os.path.join(save_path, train_file)
        val_file_path = os.path.join(save_path, val_file)
        class_weights_path = os.path.join(save_path, class_weights_file)
            
        if os.path.isfile(train_file_path):
            ## check if class weight exists: 
            print(f'File: {train_file} exists. Check if class {class_weights_file} exists.')
            train_files = joblib.load(train_file_path)
            val_files = joblib.load(val_file_path)
            
            if os.path.isfile(class_weights_path):
                try:
                    class_weight_dict = joblib.load(class_weights_path)
                    print(class_weight_dict)
                    if class_label_str and [type(k) for k in class_weight_dict.keys()][0] == np.int64:
                        class_weight_dict = dict((v, class_weight_dict.get(k, k)) for (k, v) in id2target.items())
                        print("Converting class weight dict from int to str.")
                    print("Your train, val, class weight files already exist!")
                        
                except Exception as e:
                    print(f'Your val files or class weights file probably DNE! {e}')
            else: 
                
                print(f'No class weights. Calc {class_weights_file}')
                train_df = df[df['file'].isin(train_files)] ## df from existing train files
                
                class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=train_df[step])
                class_weight_dict = dict(zip(unique_labels, class_weights))
                if class_label_str: ## save class weight dictionary with class str instead of int
                    class_weight_dict = dict((v, class_weight_dict.get(k, k)) for (k, v) in id2target.items())
                if save_files:
                    joblib.dump(class_weight_dict, class_weights_path)
                    print(f'Saved {class_weights_path}.')
                
    
        else:
            print(f'Files do not exist, create files!')
            
            ## create files
            for label in unique_labels:
                df_filtered = df[df[step] == label]
                unique_obj_ids = df_filtered['name'].unique()
                random.seed(random_seed)
                random.shuffle(unique_obj_ids)
                split_idx = int(len(unique_obj_ids) * split_ratio)
                train_obj_ids = unique_obj_ids[:split_idx]
                val_obj_ids = unique_obj_ids[split_idx:]
                train_df_list.append(df_filtered[df_filtered['name'].isin(train_obj_ids)])
                val_df_list.append(df_filtered[df_filtered['name'].isin(val_obj_ids)])
                
            train_df = pd.concat(train_df_list).reset_index(drop=True)
            val_df = pd.concat(val_df_list).reset_index(drop=True)
            
            train_obj_ids = train_df['name'].unique()
            val_obj_ids = val_df['name'].unique()
            
            assert len(set(train_obj_ids).intersection(set(val_obj_ids))) == 0
            train_files = train_df['file'].tolist() ; val_files = val_df['file'].tolist()
                
            class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=train_df[step])
            class_weight_dict = dict(zip(unique_labels, class_weights))
            if class_label_str: ## save class weight dictionary with class str instead of int
                class_weight_dict = dict((v, class_weight_dict.get(k, k)) for (k, v) in id2target.items())
                        
            if save_files:
                print(f'Files saved.')
                joblib.dump(train_files, train_file_path)
                joblib.dump(val_files, val_file_path)
                joblib.dump(class_weight_dict, class_weights_path)
                
    
        return train_files, val_files, class_weight_dict
    
    
class DataPreprocessor:
    
    """ ☆ additional pre-processing of photometry, metadata ☆ """
    
    @staticmethod
    def Mag2Flux(df):
        ''' converts magnitude to flux'''
        df_copy = df.dropna().copy()
        df_copy['flux'] = 10 ** (-0.4 * (df_copy['mag'] - 23.9))
        df_copy['flux_error'] = (df_copy['magerr'] / (2.5 / np.log(10))) * df_copy['flux']
        df_copy = df_copy[['obj_id', 'mjd', 'flux', 'flux_error', 'filter', 'type', 'jd']]
        return df_copy
    
    @staticmethod    
    def Normalize_mjd(df):
        ''' normalize modified julian date'''
        df_copy = df.copy()
        df_copy['mjd'] = df_copy.groupby('obj_id')['mjd'].transform(lambda x: x - np.min(x))
        df_copy.reset_index(drop=True, inplace=True)
        return df_copy
    
    @staticmethod
    def convert_photometry(photo_df):
        ''' converts magnitude to flux, normalizes modifed Julian date of photometry df '''
        df_gp_ready = DataPreprocessor.Mag2Flux(photo_df)
        df_gp_ready = DataPreprocessor.Normalize_mjd(df_gp_ready).drop_duplicates().reset_index(drop=True)
        return df_gp_ready

    @staticmethod
    def cut_photometry(photo_df, metadata_df, index, max_mjd=10):    
        ''' ensure mjd max not exceeded'''
        jd_current = metadata_df['jd'].iloc[index]
        photometry_filtered = photo_df[photo_df['jd'] <= jd_current]
        return None if photometry_filtered['mjd'].max() > max_mjd else photometry_filtered

    @staticmethod
    def preprocess_metadata(metadata_df):
        ''' removes metadata duplicates and irrelevant columns '''
        metadata_df = metadata_df.drop_duplicates(subset=['jd'], keep='first')
        # old version
        #columns_metadata = [ "sgscore1", "sgscore2", "distpsnr1", "distpsnr2", "ra", "dec", "nmtchps", "sharpnr", "scorr", "sky", 'jd' ]
        
        # new metadata values / CALC SOME NEW COLUMNS
        metadata_df_copy = metadata_df.copy()
        metadata_df_copy['nnondet'] = metadata_df_copy['ncovhist'] - metadata_df_copy['ndethist']
        


        # added: diffmaglim, ndethist, sigmapsf, chinr, magpsf
        columns_metadata = [ "sgscore1", "sgscore2", "distpsnr1", "distpsnr2", "ra", "dec", "nmtchps", "sharpnr", "scorr", "sky",  "diffmaglim",  "ndethist",  "ncovhist", "sigmapsf", "chinr", "magpsf", "nnondet", "classtar", 'jd' ]
        
        
        return metadata_df_copy[columns_metadata].fillna(-999.0)
