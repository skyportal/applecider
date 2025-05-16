# 🍏 AppleCider : Applying multimodal learning to Classify transient Detections Early🍏 





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
