<p align="center">
<img align="center" src="https://github.com/skyportal/applecider/blob/hyrax-main/docs/static/AppleCiDEr_use_over_black.png" alt="AppleCiDEr Logo" height="350px">
</p>



<p align="center"><i><b><ins>App</ins></b>lying multimoda<ins><b>l</b></ins> l<ins><b>e</ins></b>arning to <ins><b>C</ins></b>lassify trans<b><ins>i</ins></b>ent <b><ins>D</ins></b>etections <b><ins>E</ins></b>a<b><ins>r</ins></b>ly</i></p>


`AppleCiDEr` ([arXiv](https://arxiv.org/abs/2507.16088)) is a multimodal transient classifier that uses photometry, metadata, images, and spectra. <i>Name inspired by University of Minnesota's famous [apple program](https://mnhardy.umn.edu/apples).</i> <br>


<br><br>
### `AppleCiDEr` in ZTF production
<img align="center" src="https://github.com/skyportal/applecider/blob/hyrax-main/docs/static/ZTF_Production_use_over_black.png" alt="ZTF production diagram" height="550px">


<br><br>
***
## Repository layout

```
applecider/
├── src/applecider/
│   ├── datasets/              # dataset classes and sampling helpers
│   ├── models/                # AppleCiDEr and baseline model implementations
│   └── preprocessing_utils/   # multimodal preprocessing pipeline utilities
├── scripts/
│   └── fusion_preprocessing.py
├── tests/
│   └── applecider/
├── docs/
│   ├── conf.py
│   ├── index.rst
│   ├── notebooks/
│   ├── pre_executed/
│   └── static/
├── _archive/                  # legacy code/notebooks kept for reference
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Developer quickstart

Install in editable mode with development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Set up pre-commit hooks:

```bash
pre-commit install
```

Run all checks locally (recommended before push):

```bash
pre-commit run --all-files
```

Run tests directly:

```bash
python -m pytest --cov=./src --cov-report=html
```

Build docs directly:

```bash
sphinx-build -T -E -b html -d ./docs/_build/doctrees ./docs ./_readthedocs
```

## Citation

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
