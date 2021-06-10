CellMincer
===========

CellMincer is a software package for learning self-supervised denoising models for voltage-imaging movies.

Installation
============

```
git clone https://github.com/broadinstitute/CellMincer.git
pip install -e CellMincer/
```

Modules
=======

CellMincer provides the following scripts:

`preprocess.py` adaptively dejitters, detrends, and estimates the PG-noise in a raw VI-movie.
`python preprocess.py <path_to_config_yaml>`

`features.py` computes a set of global features spanning the xy-dimensions, including cross-correlations between adjacent pixels.
`python features.py <path_to_config_yaml>`

`run.py` trains a denoising model using noise2self-like loss, then denoises movies with the trained model.
`python run.py [--train] [--denoise] <path_to_config_yaml>`

Template `.yaml` configurations, as well as notebooks for dumping Python dictionaries to .yaml, are provided in `configs/`.
