"""
Methods to sample model parameters
"""
from typing import Callable, Tuple, List, Optional, Dict
import emcee
import numpy as np
import dynesty
from dynesty import NestedSampler


# ======================================================================================
# MCMC functions
# ======================================================================================
def run_mcmc(
    nwalk: int,
    nburn: int,
    nsteps: int,
    pguess: np.ndarray,
    logp: Callable,
    logp_args: Tuple,
    moves: emcee.moves.Move = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param nwalk: Number of walkers
    :type nwalk: int
    :param nburn: Number of burn-in steps
    :type nburn: int
    :param nsteps: Number of step after burn in
    :type nsteps: int
    :param pguess: Guess for each parameter
    :type pguess: np.ndarray
    :param logp: Log-probability function
    :type logp: Callable
    :param logp_args: Arguments of lopg other than parameter array
    :type logp_args: Tuple
    :param moves: emcee "moves" to use
    :type moves: emcee.moves.Move
    :return: The flat chains and full chains in separate arrays
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Sampler object
    ndim = len(pguess)
    sampler = emcee.EnsembleSampler(
        nwalk,
        ndim,
        logp,
        args=logp_args,
        moves=moves,
    )

    # Run MCMC
    p0 = pguess + 1e-4 * np.random.randn(nwalk, ndim)
    sampler.run_mcmc(p0, nburn + nsteps, progress=True)

    # Simple quantitative convergence checks
    print(
        f"Nsteps / Autocorrelation length: {nsteps/sampler.get_autocorr_time(quiet=True)}"
    )
    print(f"Mean acceptance rate: {np.mean(sampler.acceptance_fraction)}")

    # Get chains
    flatchain = sampler.get_chain(flat=True, discard=nburn)
    chain = sampler.get_chain(discard=nburn)

    return flatchain, chain


def run_ns(
    logl: Callable,
    ptform: Callable,
    ndim: int,
    nlive: int,
    logl_args: Tuple,
    ptform_kwargs: Optional[Dict] = None,
    dlogz: float = 0.01,
) -> dynesty.results.Results:
    """

    :param logl: Log likelihood function
    :type logl: Callable
    :param ptform: Prior transform function
    :type ptform: Callable
    :param ndim: Number of parameters
    :type ndim: int
    :param nlive: Number of live points
    :type nlive: int
    :param logl_args: Arguments to pass to log likelihood (after parameters)
    :type logl_args: Tuple
    :param ptform_args: Arguments to pass to prior transform (after unit "cube")
    :type ptform_kwargs: Tuple
    :param dlogz: Stopping criterion
    :type dlogz: float
    :return: Dynesty sampling result object
    :rtype: dynesty.results.Results
    """

    ns_sampler = NestedSampler(
        logl,
        ptform,
        ndim,
        nlive=nlive,
        logl_args=logl_args,
        ptform_kwargs=ptform_kwargs,
    )
    ns_sampler.run_nested(dlogz=dlogz)

    return ns_sampler.results