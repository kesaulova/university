from math import log, exp
import math
import numpy

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
        return float('-Inf')
    else:
        if x > 0:
            return log(x)
        else:
            print 'Logarithm from negative number!'
            exit(1)


def log_sum(loga, logb):
    """
    Compute log(x + y), base on log(x) and log(y)
    :param loga: log(x)
    :param logb: log(y)
    """
    if numpy.isinf(loga) or numpy.isinf(logb):
        if numpy.isinf(loga):
            return logb
        else:
            return loga
    if logb > loga:
        loga, logb = logb, loga
    return loga + numpy.log1p(exp(logb - loga))


def log_product(loga, logb):
    """
    Compute log(x * y), base on log(x) and log(y)
    :param loga: log(x)
    :param logb: log(y)
    """
    #if numpy.isinf(loga) or numpy.isinf(logb):
    #    return float('-inf')
    #else:
    #    return loga + logb
    return loga + logb

def iter_slog(l_tmp):
    """
    Compute log(\sum\lim{i}(x_i)), when we know log(x_i)
    :param l_tmp: list with logarifms of elements
    :return: logarifm of sum of elements
    """
    l = numpy.array(l_tmp)
    s = float("-Inf")
    for item in numpy.nditer(l):
        s = log_sum(s, item)
    return s


def iter_plog(l_tmp):
    """
    Compute log(\product\lim{i}(x_i)), when we know log(x_i)
    :param l:  list with logarifms of elements
    :return: logarifm of sum of elements
    """
    l = numpy.array(l_tmp)
    s = 0
    for item in numpy.nditer(l):
        s = log_product(s, item)
    return s


def by_iter_slog(it):
    """
    Compute log(\sum\lim{i}(x_i)), when we know log(x_i)
    :param it: iter by some structure
    :return: logarifm of sum of elements
    """
    acc = next(it)
    for snd in it:
        acc = log_sum(acc, snd)
    return acc


def get_eln(item):
    """
    Take a log from every element in array
    :param item: array (one or more dimensional)
    :return: ndarray with same shape, with log-elements
    """
    nd_item = numpy.array(item).copy()
    for x in numpy.nditer(nd_item, op_flags=['readwrite']):
        x[...] = eln(x)
    return nd_item