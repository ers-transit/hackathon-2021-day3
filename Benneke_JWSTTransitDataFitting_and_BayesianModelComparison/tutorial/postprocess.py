"""
Post-processing functions
"""
from typing import List, Optional
import numpy as np

import os

import matplotlib.pyplot as plt
import corner
import dynesty
import dynesty.utils as dyfunc
import dynesty.plotting as dyplot

import plots

FILE_EXT = "pdf"


def mcmc_processing(
    flatchain: np.ndarray,
    chain: np.ndarray,
    labels: List[str],
    savedir: Optional[str] = None,
    showplots: bool = True,
):

    quantiles = [0.159, 0.5, 0.841]

    qvals = np.quantile(flatchain, quantiles, axis=0)

    pmed, _, _ = quant_to_uncert(qvals, labels=labels)

    # Corner plot
    corner.corner(flatchain, labels=labels, quantiles=quantiles, titles=True)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f"corner.{FILE_EXT}"))
    if showplots:
        plt.show()

    # Chain plot
    if savedir is None:
        savepath = None
    else:
        savepath = os.path.join(
            savedir, f"chains.{FILE_EXT}" if savedir is not None else savedir
        )
    plots.mcmc_chains(
        chain,
        labels,
        show=showplots,
        save=savepath,
    )

    return pmed


def ns_processing(
    results: dynesty.results.Results,
    labels: List[str],
    savedir: Optional[str] = None,
    showplots: bool = True,
    run_res: Optional[dynesty.results.Results] = None,
):
    quantiles = [0.159, 0.5, 0.841]

    samples = results.samples  # samples
    weights = np.exp(results.logwt - results.logz[-1])  # normalized weights

    qvals = np.array(
        [dyfunc.quantile(samps, quantiles, weights=weights) for samps in samples.T]
    ).T

    pmed, _, _ = quant_to_uncert(qvals, labels=labels)

    # Corner plot
    dyplot.cornerplot(results, labels=labels)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f"corner.{FILE_EXT}"))
    if showplots:
        plt.show()

    # Chain plot
    dyplot.traceplot(results, labels=labels)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f"trace.{FILE_EXT}"))
    if showplots:
        plt.show()

    # Run plot, with custom function if provided
    if run_res is not None:
        dyplot.runplot(run_res)
        if savedir is not None:
            plt.savefig(os.path.join(savedir, f"run.{FILE_EXT}"))
        if showplots:
            plt.show()

    return pmed


def quant_to_uncert(quantiles: np.ndarray, labels=None) -> np.ndarray:
    qlo, qmed, qhi = quantiles

    plo = qmed - qlo
    pmed = qmed.copy()
    phi = qhi - qmed

    if labels is None:
        labels = [""] * len(pmed)

    for i in range(len(pmed)):
        print(f"{labels[i]}: {pmed[i]} + {phi[i]} - {plo[i]}")

    return pmed, phi, plo


def check_residuals(y, yerr, ymod, dofs=None, show=True, savedir=None):

    res = y - ymod

    # Reduced chi-squared
    if dofs is None:
        dofs = len(y)
    chisq = np.sum(res ** 2 / yerr ** 2)
    red_chisq = chisq / dofs
    print(f"Reduced chi2: {red_chisq}")

    # RMS of residuals
    print(f"RMS of residuals: {np.std(res)}")
    print(f"Mean Error bar: {np.mean(yerr)}")
    print(f"Ratio: {np.std(res)/np.mean(yerr)}")

    # Plot histogram of residuals
    plt.hist(res, bins=20)
    plt.title("Residuals distribution")
    if savedir is not None:
        plt.savefig(os.path.join(savedir, f"residuals_hist.{FILE_EXT}"))
    if show:
        plt.show()
