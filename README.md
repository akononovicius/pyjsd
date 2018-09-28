# PyJSD: Python implementation of the Jensen-Shannon divergence

This Python module implements estimation of the JSD scores for the observed
data assuming some distribution. This module was developed when performing
empirical analysis for the forthcomming paper by Mark Levene and Aleksejus
Kononovicius (link will be added at a later time).

Feel free to reuse the code or modify it. We would like to encourage you to
reference our paper, if it would be appropriate, but we surely do not require
you to do it.

If you want to reference the repository itself, you can do that in the
following manner:
* A. Kononovicius and M. Levene. *PyJSD: Python implementation of the
Jensen-Shannon divergence*. http://github.com/akononovicius/pyjsd.

# How to use the module
 
Here we have implemented a `JSD` function, which does three things:
1. It estimates distribution parameter values given the assumed (theoretical)
distribution and the data using Maximum likelihood estimation.
1. It estimates Jensen-Shannon Divergence (JSD) between the empirical and the
assumed distribution. Lower scores are better.
1. It estimates confidence intervals for the JSD using moving block bootstrap
method.

The `JSD` function takes four parameters: `data`, `empiricalDist`, `theorDist`
and `bootstrap`.

* `data` should be an array containing empirically observed values.
* `empiricalDist` should be a dictionary with three keys: `start` (minimal value
reflected in the obtained empirical distribution), `stop` (maximum value reflected
in the obtained empirical distribution) and `bins` (number of bins to used when
estimating the empirical distribution). For example:
```python
empiricalDist={
    "start": 0.0,
    "stop": 505.0,
    "bins": 1000,
}
```
* `theorDist` should be a dictionary with three keys: `cdf` (a function which
returns CDF values at given points), `likelihood` (a function which returns
likelihood of the given empirically observed values) and `params` (initial
parameter values from which MLE algorithm will start). For example:
```python
norm={
    "cdf": lambda params,x: scipy.stats.norm.cdf(x,loc=params[0],scale=params[1]),
    "likelihood": lambda params,data: -numpy.sum(scipy.stats.norm.logpdf(data,loc=params[0],scale=params[1])),
    "params": [1,1],
}
```
Note that some distributions are defined `jsd.distributions` submodule.
* `bootstrap` should be a dictionary with three keys: `iterations` (number of
sample to obtain), `blockSize` (what block size to use) and `percentiles`
(which percentiles to report). Example:
```python
bootstrap={
    "iterations": 1000,
    "blockSize": 1,
    "percentiles": [2.5,97.5],
}
```
