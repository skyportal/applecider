import os
import numpy as np
import pandas as pd
import joblib
import pickle
import random

import torch
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import Dataset


class DataGenerator(Dataset):

    def __init__(self, config, split='train'):
        super(DataGenerator, self).__init__()

        self.split = split                                                #  'train', 'val', 'test'
        self.mode = config['mode']                                        #  'all', 'ztf', 'photo', 'meta', 'image', 'spectra'
        self.preprocessed_path = config['preprocessed_path']              #  preprocessed train, validation alerts 
        self.preprocessed_test_path = config['preprocessed_test_path']    #  preprocessed test alerts
        self.step = config['step']                                        #  name of df column with object type.... in our case 'type'... i should rename this
        self.random_seed = config['random_seed']  # 42, 66, 0, 12, 123
        
        self.group_labels = config['group_labels']        #  labels become -> SN I, SN II, CV, AGN, TDE
        self.group_labels_SN = config['group_labels_SN']  #  labels become -> SN, CV, AGN, TDE
        
        self.max_samples = config['max_samples']       #  max sample per class (not currently in use). downsampling when splitting train and validation alert files
        self.seq_len = config['seq_len']               #  maximum photometry length
        self.train_files = config['train_files_path']  #  .pkl w/list of train alert files
        self.val_files = config['val_files_path']      #  .pkl w/list of validation alert files
        self.weights = config['class_weights_path']    #  .pkl w/dictionary of class weights
        self.df = config['df_path']                    #  df with object ids, alert file, object type (has train and validation objects)
        self.df_test = config['test_df_path']          #  df with object ids, alert file, object type (only has test objects)
    

        require_meta = ['meta', 'ztf','all']    # need to standardize metadata if mode incldues metadata
        if self.mode in require_meta: 
            self.scaler = joblib.load(config['scaler_path'])
            if self.scaler is None:
                raise ValueError('No scaler path. Add path.')
         
        if self.split =='train':
            self.df = pd.read_csv(self.df)
        elif self.split == 'val':
            self.df = pd.read_csv(self.df)
        elif self.split == 'test':
            self.df = pd.read_csv(self.df_test)
            self.preprocessed_path = self.preprocessed_test_path 
        else:
            raise ValueError('Split must be either train, val, or test.')
            
        self._limit_samples()
        self._split()

        ## create convenient mapping for label from str to int and from int to str
        if self.group_labels:
            ## group -> SN I, SN II, CV, AGN, TDE
            self.id2target = {0: 'SN I', 1: 'SN II', 2: 'Cataclysmic', 3: 'AGN', 4: 'Tidal Disruption Event'}
            self.target2id = {'SN Ia': 0 , 'SN Ic': 0,  'SN Ib': 0, 'SN II': 1, 'SN IIP': 1, 'SN IIn': 1, 'SN IIb': 1, 'Cataclysmic': 2, 'AGN': 3, 'Tidal Disruption Event': 4}
        elif self.group_labels_SN:
            ## group SN -> SN, CV, AGN, TDE
            self.id2target = {0: 'SN', 1:'Cataclysmic', 2:'AGN', 3:'Tidal Disruption Event'}
            self.target2id = {'SN Ia':0, 'SN Ic':0, 'SN Ib':0 , 'SN II':0, 'SN IIP':0, 'SN IIn':0, 'SN IIb':0, 'Cataclysmic':1, 'AGN':2, 'Tidal Disruption Event':3}
            
        else:
            self.id2target = {0: 'SN Ia', 1:'SN Ic',  2:'SN Ib' , 3: 'SN II', 4:'SN IIP', 5:'SN IIn', 6:'SN IIb', 7:'Cataclysmic', 8:'AGN', 9:'Tidal Disruption Event'}
            self.target2id = {v: k for k, v in self.id2target.items()}

        self.num_classes = len(self.target2id)

    def _limit_samples(self):
        """ downsample samples for each class if max_samples is set """
        if self.max_samples:
            for cls in self.df[self.step].unique():
                df_cls = self.df[self.df[self.step] == cls]
                df_not_cls = self.df[self.df[self.step] != cls]

                if len(df_cls) > self.max_samples:
                    print(f'Down sampled class {cls} from {len(df_cls)} to {self.max_samples}')
                    df_cls_down = df_cls.sample(n=self.max_samples, random_state=self.random_seed)
                    self.df = pd.concat([df_not_cls, df_cls_down], ignore_index=True)
    
    def _split(self):
        """ sort train, val based on alert names in already created pkl from preprocessing steps """
        
        if self.split == 'train':
            if os.path.isfile(self.train_files):
                with open(self.train_files, 'rb') as file:
                    train_files_saved = pickle.load(file)
                    self.df = self.df[self.df['file'].isin(train_files_saved)] 
            else:
                raise ValueError('Missing train_files at train_files_path.')   
        elif self.split == 'val':
            if os.path.isfile(self.val_files):
                with open(self.val_files, 'rb') as file:
                    val_files_saved = pickle.load(file)
                    self.df = self.df[self.df['file'].isin(val_files_saved)]
            else:
                raise ValueError('Missing val_files at val_files_path.')
        elif self.split == 'test':
            self.df = pd.read_csv(self.df_test)
        else:
            raise ValueError('Split must be train, val, or test.')
                                                                      
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """ load processed object alerts to get photometry, metadata, images, spectra """
        
        el = self.df.iloc[index]                 #  get df row for object 
        target = self.target2id[el[self.step]]   #  get object type, make str label int 

        file_path = os.path.join(self.preprocessed_path, el['file'])  #  get object alert file name from df
        sample = np.load(file_path, allow_pickle=True).item()         #  load object alert file: .npy 

        ## Photometry
        ## mjd, ztf-g, ztf-r, ztf-i
        require_photometry = ['photo','ztf', 'all']
        if self.mode in require_photometry:
            photometry = sample['photometry']
            photometry = torch.tensor(photometry, dtype=torch.float32)
        else: ## make blank photometry tensor
            photometry = torch.ones(self.seq_len, 7)
            
        ## Metadata
        require_meta = ['meta', 'ztf','all']
        if self.mode in require_meta:
            metadata = sample['metadata'].to_numpy()
            metadata = self.scaler.transform(metadata.reshape(1, -1))[0]
            metadata = metadata.astype(np.float32)
            metadata = torch.tensor(metadata)
        else: ## make blank metadata tensor
            metadata = torch.ones(10)
            
        ## Images
        require_image = ['image', 'ztf', 'all']
        if self.mode in require_image :
            images = sample['images']
            images = np.transpose(images, (2, 0, 1))
            images = images.astype(np.float32)
            images = torch.tensor(images)
        else: ## make blank image tensor
            images = torch.ones(3, 63, 63)
        
        require_spectra = ['spectra', 'all']
        if self.mode in require_spectra:
            spectra = sample['spectra']
            spectra = torch.tensor(spectra)
        else: ## make blank spectra tensor
            spectra = torch.ones(1, 3481)

        return photometry, images, metadata, spectra, target
