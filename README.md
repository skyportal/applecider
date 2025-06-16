<p align="center">
<img align="center" src="https://github.com/skyportal/applecider/blob/main/logo/AppleCiDEr_github.png" alt="Draft AppleCiDEr Logo" height="350px">
</p>



<p align="center"><i><b><ins>App</ins></b>lying multimoda<ins><b>l</b></ins> l<ins><b>e</ins></b>arning to <ins><b>C</ins></b>lassify trans<b><ins>i</ins></b>ent <b><ins>D</ins></b>etections <b><ins>E</ins></b>a<b><ins>r</ins></b>ly</i></p>
<p align="center"><i>(repo under construction circa 5/16)</i></p>

`AppleCiDEr` is a multimodal transient classifer that uses photometry, metadata, images and spectra. <i>Name inspired by University of Minnesota's famous [apple program](https://mnhardy.umn.edu/apples).</i> <br>



<br><br>
***
IP structure:

```
AppleCider
└── core
    ├── dataset.py             # DataGenerator
    ├── model.py               # implement multimodal models. contains: AppleCider (for all modalities), ZwickyCoder (for photo, image, metadata)
    └── trainer.py             
└── models                           # collection of models used in AppleCiDEr and baseline models    
    ├── Informer.py                  # photometry model
    ├── BTSModel.py                  # image model
    ├── MetaModel.py                 # metadata model
    ├── GalSpecNet.py                # spectra model
    ├── SpectraEfficientNetV2L.py    # spectra model
    ├── SpectraViTBase.py            # spectra model
    └── SpectraConvNeXtBase_train.py # spectra model
└── preprocess
    ├── process.py                   # preprocess script
    ├── alert_processor.py           # for ZTF alerts
    ├── photometry_processor.py      # for aux ZTF alerts
    ├── data_preprocessor.py         # combined ZTF, aux
    ├── transient_dataset.py         # preprocess dataset, save as "new" object alerts


```
