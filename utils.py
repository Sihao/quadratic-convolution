"""
Utilities for stimulus processing

"""
from numpy.lib.stride_tricks import as_strided
from warnings import filterwarnings
import warnings

from numpy import isfinite, array, ndarray


def normStim(stim,pixelNorm=True):
    """
    Normalize stimulus
        stim: numpy array with sample number as last dimension
        pixelNorm: If True, normalize each location by individual mean and stdev.
           If a tuple, use first element as mean and second as stdev.
           Otherwise, normalize using global statistics.
        Returns normalized stimulus, mean, and stdev.
    """
    # Ignore divide by zero warnings
    filterwarnings('ignore','invalid value encountered in divide')

    # Normalize using given values
    if isinstance(pixelNorm,tuple):
        # Should contain two values: mean, stdev
        assert len(pixelNorm) == 2
        stimAve,stimStDev = pixelNorm

        # Convert number to numpy array
        if isinstance(stimAve,int) or isinstance(stimAve,float):
            stimAve = array(stimAve)
        # Otherwise, make sure input is in a compatible shape
        elif isinstance(stimAve,ndarray):
            assert stimAve.shape == stim.shape[:-1]+(1,) or stimAve.size == 1

    # Normalize using pixel statistics
    elif pixelNorm:
        stimAve = stim.mean(axis=-1)
        stimAve.shape += (1,)
        stimStDev = stim.std(axis=-1)
        stimStDev.shape += (1,)

    # Normalize using full stimulus statistics
    else:
        stimAve = stim.mean()
        stimStDev = stim.std()

    # Normalize stim
    stim -= stimAve
    stim /= stimStDev

    # Check for bad pixels (usually pixel with no variation)
    stim[~isfinite(stim)]=0.

    # Remove filter added above so it does not affect later code
    warnings.filters = warnings.filters[1:]

    return stim,stimAve,stimStDev

# Check if x is an integer
def IntCheck(x):
    if isinstance(x,int):
        return x
    else:
        assert x == int(x)
        return int(x)

# Create collection of patches
def gridStim(stim,fsize,nlags=1):

    FSIZE = stim.shape[:-1]+(nlags,)
    gsize = tuple([F-f+1 for F,f in zip(FSIZE,fsize)])

    ssh = stim.shape
    sst = stim.strides

    Ssh = (ssh[-1]-nlags+1,)+fsize+gsize
    Sst = sst[-1:]+2*sst

    return as_strided(stim,shape=Ssh,strides=Sst)