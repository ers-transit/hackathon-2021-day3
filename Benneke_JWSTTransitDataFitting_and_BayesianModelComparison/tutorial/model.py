"""
Functions that describe model
"""
from typing import List, Union
import numpy as np
import batman

from scipy import stats


# ==============================================================================
# Physical model
# ==============================================================================
def impact_to_inc(impact: float, a: float) -> float:
    """
    Get inclination from impact parameter

    :param impact: Impact parameter
    :type impact: float
    :param a: Semi-major axis in units of Rstar
    :type a: float
    :return: Inclination in degrees
    :rtype: float
    """

    return np.degrees(np.arccos(impact / a))


def forward_model(
    p: np.ndarray,
    batparams: Union[List[batman.TransitParams], batman.TransitParams],
    batmodel: Union[List[batman.TransitModel], batman.TransitModel],
) -> np.ndarray:
    """
    Forward model to generate light curve with three parameters per object
    (assuming others are fixed).

    Batman parameter and model objects are required to handle the physics and
    provide times and extra parameters.

    :param p: Input parameter array (t0, rp, b)
    :type p: np.ndarray
    :param batparams: Batman parameter object to use in model
    :type batparams: Union[List[batman.TransitParams], batman.TransitParams]
    :param batmodel: Batman model to generate light curve
    :type batmodel: Union[List[batman.TransitModel], batman.TransitModel]
    :return: Light curve at model times
    :rtype: np.ndarray
    """
    if isinstance(batparams, batman.TransitParams):
        batparams = [batparams]
    if isinstance(batmodel, batman.TransitModel):
        batmodel = [batmodel]

    p = np.array(p)

    assert len(p) / 3 == len(
        batparams
    ), "Forward model requires 3 parameters per object"

    # Update parameter object
    for i, bp in enumerate(batparams):
        bp.t0 = p[0 + 3 * i]
        bp.rp = p[1 + 3 * i]
        bp.inc = impact_to_inc(p[2 + 3 * i], bp.a)

    # Generate light curve with model
    lc = 1.0
    for i, bm in enumerate(batmodel):
        lc += bm.light_curve(batparams[i]) - 1

    return lc


# ==============================================================================
# Prior information
# ==============================================================================
# Prior information is defined here

# ------------------------------------------------------------------------------
# MCMC priors
# ------------------------------------------------------------------------------
def uniform(pval: float, pmin: float, pmax: float) -> float:
    """
    Logarithm of a uniform prior

    :param pval: parameter value
    :type pval: float
    :param pmin: Upper bound
    :type pmin: float
    :param pmax: Lower bound
    :type pmax: float
    :return: Logartithm of uniform distribution
    :rtype: float
    """
    assert pmax > pmin, "Upper bound should be larger than lower bound"

    if pmin < pval < pmax:
        return -np.log(pmax - pmin)
    else:
        return -np.inf


def log_uniform(pval: float, pmin: float, pmax: float) -> float:
    """
    Logarithm of a log-uniform prior (i.e. uniform in log-space of the
    parameter value)

    :param pval: parameter value
    :type pval: float
    :param pmin: Upper bound
    :type pmin: float
    :param pmax: Lower bound
    :type pmax: float
    :return: Logartithm of log-uniform distribution
    :rtype: float
    """
    assert pmax > pmin, "Upper bound should be larger than lower bound"

    if pmin < pval < pmax:
        return -np.log(np.log(pmax / pmin)) - np.log(pval)
    else:
        return -np.inf


def log_prior(p: np.ndarray) -> float:
    """
    Logarithm of the prior probability for a given set of parameter values.

    The type of priors and boundaries are hardcoded for the case studied in
    the homework.

    :param p: Parameter values
    :type p: np.ndarray
    :return: Logarithm of the prior probability
    :rtype: float
    """

    assert len(p) % 3 == 0, "Should have exactly 3 parameters per object"

    lp = 0.0
    for i in range(int(len(p) / 3)):
        t0 = p[0 + 3 * i]
        rp = p[1 + 3 * i]
        b = p[2 + 3 * i]

        lp += uniform(t0, 1.212, 1.362)
        lp += log_uniform(rp, 0.01, 0.1)
        lp += uniform(b, 0.0, 1.0)

    return lp


# ------------------------------------------------------------------------------
# Nested sampling transforms
# ------------------------------------------------------------------------------
def uniform_transform(u: float, minval: float, maxval: float) -> float:
    """
    Transform uniform variable u ~ U(0, 1) to uniform between two bounds.

    :param u: Value from U(0, 1)
    :type u: float
    :param minval: Lower bound of parameter distribution
    :type minval: float
    :param maxval: Upper bound of parameter distribution
    :type maxval: float
    :return: variable transformed to new uniform parameter
    :rtype: float
    """
    assert maxval > minval, "Upper bound should be larger than lower bound"

    scale = maxval - minval

    # The scipy ppf below is equivalent to this (used only as sanity check)
    # return scale * u + minval

    return stats.uniform.ppf(u, loc=minval, scale=scale)


def log_uniform_transform(u: float, minval: float, maxval: float) -> float:
    """
    Transform uniform variable u ~ U(0, 1) to log-uniform between two bounds.

    :param u: Value from U(0, 1)
    :type u: float
    :param minval: Lower bound of parameter distribution
    :type minval: float
    :param maxval: Upper bound of parameter distribution
    :type maxval: float
    :return: variable transformed to log-uniform parameter
    :rtype: float
    """

    assert maxval > minval, "Upper bound should be larger than lower bound"

    return stats.loguniform.ppf(u, minval, maxval)


def prior_transform(u: np.ndarray, force_smaller: bool = False) -> np.ndarray:
    """
    Transform unit-uniform variables to parameters of interest

    :param u: Unit-uniform variable with one dimension per parameter
    :type u: np.ndarray
    :param force_smaller: Force second body to be smaller than the first (default is False)
    :type force_smaller: bool
    :return: Parameter array
    :rtype: np.ndarray
    """
    p = np.array(u)

    assert len(p) % 3 == 0, "Should have exactly 3 parameters per object"

    for i in range(int(len(p) / 3)):
        p[0 + i * 3] = uniform_transform(u[0 + i * 3], 1.212, 1.362)  # t0
        if force_smaller:
            p[1 + i * 3] = log_uniform_transform(
                u[1 + i * 3], 0.01, 0.1 if i == 0 else p[1]
            )  # rp
        else:
            p[1 + i * 3] = log_uniform_transform(u[1 + i * 3], 0.01, 0.1)  # rp
        p[2 + i * 3] = uniform_transform(u[2 + i * 3], 0.0, 1.0)  # b (impact)

    return p


# ==============================================================================
# Log likelihood and posterior functions
# ==============================================================================
def log_like(
    p: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    batparams: Union[List[batman.TransitParams], batman.TransitParams],
    batmodel: Union[List[batman.TransitModel], batman.TransitModel],
    normal: bool = True,
) -> float:
    """
    Log likelihood of data for a given model

    :param p: Parameter array
    :type p: np.ndarray
    :param y: y values of data
    :type y: np.ndarray
    :param yerr: error on y values
    :type yerr: np.ndarray
    :param batparams: batman parameter object(s), one per orbiting body
    :type batparams: Union[List[batman.TransitParams], batman.TransitParams]
    :param batmodel: batman parameter object(s), one per orbiting body
    :type batmodel: Union[List[batman.TransitModel], batman.TransitModel]
    :param normal: Normalize likelihood if True (default is True)
    :type normal: bool
    :return: Log-likelihood value
    :rtype: float
    """

    # Compute model (times given by batman model object
    mod = forward_model(p, batparams, batmodel)

    # Normalization
    sigma2 = yerr ** 2
    if normal:
        norm_const = -0.5 * np.sum(np.log(2 * np.pi * sigma2))
    else:
        norm_const = 0.0

    # Chi2 (how well model matches data)
    chisq = np.sum((y - mod) ** 2 / sigma2)

    return -0.5 * chisq + norm_const


def log_prob(
    p: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    batparams: Union[List[batman.TransitParams], batman.TransitParams],
    batmodel: Union[List[batman.TransitModel], batman.TransitModel],
) -> float:
    """
    Log posterior probability

    :param p: Parameter array
    :type p: np.ndarray
    :param y: y values of data
    :type y: np.ndarray
    :param yerr: error on y values
    :type yerr: np.ndarray
    :param batparams: batman parameter object(s), one per orbiting body
    :type batparams: Union[List[batman.TransitParams], batman.TransitParams]
    :param batmodel: batman parameter object(s), one per orbiting body
    :type batmodel: Union[List[batman.TransitModel], batman.TransitModel]
    :return: Log of posterior probability
    :rtype: float
    """
    lp = log_prior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(p, y, yerr, batparams, batmodel)
