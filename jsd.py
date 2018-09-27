# -*- coding: utf-8 -*-

from scipy.optimize import __minimize

# Estimation of the empirical cdf
def __EmpiricalCDF(bins,data):
    empiricalHistogram=__np.histogram(data,bins=bins)[0]
    empiricalCH=__np.cumsum(empiricalHistogram)
    return empiricalCH/empiricalCH[-1]

# Maximum likelihood estimation of the parameters
def __MLE(distLike,distParams,data):
    return __minimize(distLike,distParams,args=(data),
                      method="nelder-mead").x

# estimate JSD of a data given assumed distribution
def __JSD(data,empiricalDist,theorDist,returnParams=True):
    # run MLE
    distParams=__MLE(theorDist["likelihood"],theorDist["params"],data)
    # setup empirical and theoretical (assumed) distribution
    cdfBins=__np.linspace(empiricalDist["start"],
                          empiricalDist["stop"],
                          num=empiricalDist["bins"])
    cdf1=__EmpiricalCDF(cdfBins,data)
    cdf2=theorDist["cdf"](distParams,cdfBins[1:])
    # estimate JSD
    mcdf=0.5*(cdf1+cdf2)
    with __np.errstate(divide="ignore",invalid="ignore"):
        term1=cdf1*__np.log(cdf1/mcdf)
        term2=cdf2*__np.log(cdf2/mcdf)
        # 0*log(x)=0 (even if x is zero)
        term1[cdf1==0]=0
        term2[cdf2==0]=0
    normalization=__np.sum(cdf1)+__np.sum(cdf2)
    jsd=__np.sqrt(__np.sum(term1+term2)/(normalization*__np.log(2)))
    if returnParams:
        return jsd,distParams
    return jsd

# estimate assumed distribution parameters, JSD score, JSD confidence intervals
def JSD(data,empiricalDist,theorDist,bootstrap):
    # estimate JSD of the original data
    jsdEstimate,distParams=__JSD(data,empiricalDist,theorDist)
    # estimate confidence interval of the JSD using bootstrap methods
    jsdConfidence=None
    if bootstrap["iterations"]>0:
        if bootstrap["blockSize"]<=1:
            # ordinary bootstrap
            tmpJSD=[]
            for rep in range(bootstrap["iterations"]):
                resample=__np.random.choice(data,size=len(data))
                tmpJSD+=[__JSD(resample,empiricalDist,theorDist,returnParams=False),]
            jsdConfidence=__np.percentile(tmpJSD,bootstrap["percentiles"])
        else:
            # moving block bootstrap
            origLen=len(data)
            data=__np.append(data,data[:bootstrap["blockSize"]-1])
            getBlocks=origLen//bootstrap["blockSize"]+1
            tmpJSD=[]
            for rep in range(bootstrap["iterations"]):
                selectedBlocks=__np.random.choice(range(origLen),size=getBlocks)
                resample=[data[sb:sb+bootstrap["blockSize"]] for sb in selectedBlocks]
                resample=resample[:origLen]
                tmpJSD+=[__JSD(resample,empiricalDist,theorDist,returnParams=False),]
            jsdConfidence=__np.percentile(tmpJSD,bootstrap["percentiles"])
            pass
    return {
        "parameterEstimates": distParams,
        "jsdEstimate": jsdEstimate,
        "jsdConfidence": jsdConfidence,
    }
