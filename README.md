<p align="center">
<img align="center" src="https://github.com/skyportal/applecider/blob/main/logo/AppleCiDEr%20-%20use%20over%20black.png" alt="Draft AppleCiDEr Logo" height="350px">
</p>



<p align="center"><i><b><ins>App</ins></b>lying multimoda<ins><b>l</b></ins> l<ins><b>e</ins></b>arning to <ins><b>C</ins></b>lassify trans<b><ins>i</ins></b>ent <b><ins>D</ins></b>etections <b><ins>E</ins></b>a<b><ins>r</ins></b>ly</i></p>
<p align="center"><i>(repo under construction circa 5/16)</i></p>


`AppleCiDEr` ([arXiv](https://arxiv.org/abs/2507.16088) ) is a multimodal transient classifer that uses photometry, metadata, images and spectra. <i>Name inspired by University of Minnesota's famous [apple program](https://mnhardy.umn.edu/apples).</i> <br>


<br><br>
### `AppleCiDEr` in ZTF production
<img align="center" src="https://github.com/skyportal/applecider/blob/main/img/ZTF_Production%20-%20use%20over%20black.png" alt="ZTF production diagram" height="550px">


<br><br>
***
IP structure:

```
AppleCider Architecture 
└── core
    ├── dataset.py             # DataGenerator
    ├── model.py               # implement multimodal models. contains: AppleCider (for all modalities), ZwickyCoder (for photo, image, metadata)
    └── trainer.py             
└── models                               # collection of models used in AppleCiDEr and baseline models    
    ├── Time2Vec.py, BaselineCLS.py      # photometry model   
    ├── AstroMiNN.py                     # image, metadata model
    └── other models           # old models for comparison
        ├── Informer.py        # photometry model
        ├── BTSModel.py        # image model
        ├── MetaModel.py       # metadata model
        └── GalSpecNet.py      # spectra model

└── preprocess
    ├── process.py                   # preprocess script
    ├── alert_processor.py           # for ZTF alerts
    ├── photometry_processor.py      # for aux ZTF alerts
    ├── data_preprocessor.py         # combined ZTF, aux
    ├── transient_dataset.py         # preprocess dataset, save as "new" object alerts
└── notebooks
└── files
    ├── ZTF_IDs.txt    # all ZTF IDs used in AppleCiDEr's dataset
    └── cider_BTS.csv  # objects (+ classification) used to train AppleCiDEr that are in the public Bright Transient Survey

└── logo

```

### citation

```
@article{junell2025AppleCiDEr,
      title={Applying multimodal learning to Classify transient Detections Early (AppleCiDEr) I: Data set, methods, and infrastructure}, 
      author={Alexandra Junell and Argyro Sasli and Felipe Fontinele Nunes and Maojie Xu and Benny Border and Nabeel Rehemtulla and Mariia Rizhko and Yu-Jing Qin and Theophile Jegou Du Laz and Antoine Le Calloch and Sushant Sharma Chaudhary and Shaowei Wu and Jesper Sollerman and Niharika Sravan and Steven L. Groom and David Hale and Mansi M. Kasliwal and Josiah Purdum and Avery Wold and Matthew J. Graham and Michael W. Coughlin},
      year={2025},
      eprint={2507.16088},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2507.16088}, 
}

```

