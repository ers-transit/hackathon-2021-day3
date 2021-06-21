"""
Some custom plotting functions for the homework problem
"""
import warnings
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt


def transit_plot(
    t: np.ndarray,
    flux: np.ndarray,
    eflux: Optional[np.ndarray] = None,
    model_flux: Optional[np.ndarray] = None,
    show_res: Optional[bool] = None,
    show: bool = True,
    save: Optional[str] = None,
) -> None:
    """
    Plot transit light curve with or without model and residuals

    :param t: Time values
    :type t: np.ndarray
    :param flux: Flux values
    :type flux: np.ndarray
    :param eflux: Error on flux data
    :type eflux: Optional[np.ndarray]
    :param model_flux: Model flux values
    :type model_flux: Optional[np.ndarray]
    :param show_res: Show residuals panel if true (only works if model_flux provided)
                     By default, shows only if there is a model
    :type show_res: Optional[bool]
    """
    if show_res is None and model_flux is None:
        show_res = False
    elif show_res is None and model_flux is not None:
        show_res = True

    if show_res and model_flux is None:
        warnings.warn(
            "show_res works only if model_flux is provided, will only plot data",
            RuntimeWarning,
        )
        show_res = False

    if show_res:
        fig, axes = plt.subplots(nrows=2)
        axlc = axes[0]
        axres = axes[1]
    else:
        fig, axlc = plt.subplots(nrows=1)
        axres = None
        axes = [axlc]

    axlc.errorbar(t, flux, yerr=eflux, fmt="k.", capsize=2, zorder=-1)
    if model_flux is not None:
        axlc.plot(t, model_flux)
        axlc.set_ylabel("Flux")

    if show_res:
        res = flux - model_flux
        axres.errorbar(t, res, yerr=eflux, fmt="k.", capsize=2)
        axres.set_ylabel("Residuals")

    axes[-1].set_xlabel("Time [d]")
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()


def mcmc_chains(
    samples: np.ndarray,
    labels: List[str],
    show: bool = True,
    save: Optional[str] = None,
) -> None:
    """
    Plot evolution of MCMC chains

    :param samples: full chains with shape (nwalkers, nsteps, ndim)
    :type samples: np.ndarray
    :param labels: Parameter labels
    :type labels: List[str]
    :param show: Show plot if true (default is true)
    :type show: bool
    :param save: Path where to save plot (default is None)
    :type save: Optional[str]
    """
    ndim = samples.shape[-1]
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
