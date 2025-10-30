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
        metadata_df_copy = metadata_df.copy()
        metadata_df_copy['nnondet'] = metadata_df_copy['ncovhist'] - metadata_df_copy['ndethist']

        columns_metadata = [ "sgscore1", "sgscore2", "distpsnr1", "distpsnr2", "ra", "dec", "nmtchps", "sharpnr", "scorr", "sky",  "diffmaglim",  "ndethist",  "ncovhist", "sigmapsf", "chinr", "magpsf", "nnondet", "classtar", 'jd' ]
        
        
        return metadata_df_copy[columns_metadata].fillna(-999.0)



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

