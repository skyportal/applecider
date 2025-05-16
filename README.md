<p align="center">
<img align="center" src="https://github.com/skyportal/applecider/blob/main/logo/AppleCiDEr_github.png" alt="Draft AppleCiDEr Logo" height="350px">
</p>


`AppleCiDEr` (`App`lying multimoda`l` l`e`arning to `C`lassify trans`i`ent `D`etections `E`a`r`ly): is a multimodal transient classifer that uses photometry, metadata, images and spectra. <i>Name inspired by University of Minnesota's famous [apple program](https://mnhardy.umn.edu/apples).</i> <br>



<br><br>
***
IP structure:
```
AppleCider
└── core
    ├── dataset.py             # DataGenerator
    ├── model.py               # implement multimodal models. contains: AppleCider (for all modalities), ZwickyCoder (for photo, image, metadata)
└── models                          # contains individual files for each model     
    ├── Informer.py                 # photometry model -> 
    ├── BTSModel.py                 # image model      -> CNN 
    ├── MetaModel.py                # metadata model   -> perceptron
    └── GalSpecNet.py               # spectra model    -> CNN

└── preprocess
    ├── data_preprocessor.py     # includes: AlertProcessor, PhotometryProcessor, DataPreprocessor
    ├── transient_dataset.py     # preprocess all alerts, saves "new" object alerts

```
