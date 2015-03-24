from math import log, exp
import math
import itertools
import numpy
import re

def discrete_distribution(probabilities):
    """
    Implement realization of discrete distributions with given probabilities
    :return: realization
    """
    values = range(len(probabilities))
    cumsums = numpy.cumsum(probabilities)
    # digitize - Return the indices of the bins to which each value in input array belongs
    # random_sample(1) - Return random floats in the half-open interval [0.0, 1.0)
    return values[numpy.digitize(numpy.random.random_sample(1), cumsums)]


def eexp(x):
    """
    Extended exponential function
    """
    if math.isnan(x):
        return 0
    else:
        return exp(x)

def eln(x):
    """
    Extended logarithm
    :param x: x
    :return: ln(x), if x > 0
    """
    if x == 0:
        return float('Nan')
    else:
        if x > 0:
            return log(x)
        else:
            print 'Logarithm from negative number!'
            exit(1)

def logSum(loga, logb):
    """
    Compute log(x + y), base on log(x) and log(y)
    :param loga: log(x)
    :param logb: log(y)
    """
    if math.isnan(loga) or math.isnan(logb):
        if math.isnan(loga):
            return logb
        else:
            return loga
    if logb > loga:
        loga, logb = logb, loga
    return loga + log(1 + exp(logb - loga))

def log_product(loga, logb):
    """
    Compute log(x * y), base on log(x) and log(y)
    :param loga: log(x)
    :param logb: log(y)
    """
    if math.isnan(loga) or math.isnan(logb):
        return float('NaN')
    #elif loga == 0 or logb == 0:
    #    return 0
    else:
        return loga + logb

def iter_slog(l):
    #implement logarifm of number of elements
    it = iter(l)
    acc = next(it)
    for snd in it:
        acc = logSum(acc, snd)
    return acc

def iter_plog(l):
    #implement logarifm of number of elements
    it = iter(l)
    acc = next(it)
    for snd in it:
        acc = log_product(acc, snd)
    return acc