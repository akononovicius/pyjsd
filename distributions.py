# -*- coding: utf-8 -*-

import numpy as __np
import scipy.stats as __stats

# distributions for use in JSD function as the theoretical (assumed) distributions
norm={
    "cdf": lambda params,x: __stats.norm.cdf(x,loc=params[0],scale=params[1]),
    "likelihood": lambda params,data: -__np.sum(__stats.norm.logpdf(data,loc=params[0],scale=params[1])),
    "params": [1,1],
}
logn={
    "cdf": lambda params,x: __stats.lognorm.cdf(x,params[1],loc=0,scale=__np.exp(params[0])),
    "likelihood": lambda params,data: -__np.sum(__stats.lognorm.logpdf(data,params[1],loc=0,scale=__np.exp(params[0]))),
    "params": [1,1],
}
gamma={
    "cdf": lambda params,x: __stats.gamma.cdf(x,params[0],scale=params[1]),
    "likelihood": lambda params,data: -__np.sum(__stats.gamma.logpdf(data,params[0],scale=params[1])),
    "params": [3,3],
}
weibull={
    "cdf": lambda params,x: __stats.weibull_min.cdf(x,params[0],scale=params[1]),
    "likelihood": lambda params,data: -__np.sum(__stats.weibull_min.logpdf(data,params[0],scale=params[1])),
    "params": [3,3],
}
qgauss={
    "cdf": lambda params,x: __stats.t.cdf(x,params[0]-1,scale=params[1]/__np.sqrt(params[0]-1)),
    "likelihood": lambda params,data: -__np.sum(__stats.t.logpdf(data,params[0]-1,scale=params[1]/__np.sqrt(params[0]-1))),
    "params": [4,1],
}
beta={
    "cdf": lambda params,x: __stats.beta.cdf(x,params[0],params[1]),
    "likelihood": lambda params,data: -__np.sum(__stats.beta.logpdf(data,params[0],params[1])),
    "params": [3,3],
}
exp={
    "cdf": lambda params,x: __stats.expon.cdf(x,scale=params[0]),
    "likelihood": lambda params,data: -__np.sum(__stats.expon.logpdf(data,scale=params[0])),
    "params": [1],
}
pareto={
    "cdf": lambda params,x: __stats.pareto.cdf(x,params[0]),
    "likelihood": lambda params,data: -__np.sum(__stats.pareto.logpdf(data,params[0])),
    "params": [1],
}
