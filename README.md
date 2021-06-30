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

CellMincer provides the following tools:

`preprocess` adaptively dejitters, detrends, and estimates the PG-noise in a raw VI-movie.
`cellmincer preprocess -i <path_to_config_yaml>`

`feature` computes a set of global features spanning the xy-dimensions, including cross-correlations between adjacent pixels.
`cellmincer feature -i <path_to_config_yaml>`

`train` trains a denoising model using noise2self-like loss.
`cellmincer train -i <path_to_config_yaml>`

`denoise` uses a trained model to denoise VI-movies.
`cellmincer denoise -i <path_to_config_yaml>`

`insight` produces performance statistics.
`cellmincer insight -i <path_to_config_yaml>`

Template `.yaml` configurations are provided in `configs/`.
