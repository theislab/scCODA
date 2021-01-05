import numpy as np


def sample_size_estimate(
        mcc_desired: float,
        increase: float,
        lf_increase: float
) -> int:
    """
    calculates the estimated number of required samples for
    fixed MCC, absolute increase and log2-fold change using the fitted linear model from
    Büttner, Ostner et al., 2020

    Linear model parameters:

    - `(Intercept)`: -1.3675613850217

    - `total_samples`: 0.0193158965178381

    - `log_fold_increase`: 0.704729538709909

    - `log_increase`: 0.315857162659738

    - `log_fold_increase`: -0.0927955725385892

    Parameters
    ----------
    mcc_desired
        desired MCC value
    increase
        mean absolute increase of cells between the groups
    lf_increase
        mean log2-fold increase of cells from one group to the other

    Returns
    -------
    sample size estimate

    n_samples -- int
        estimated number of required samples

    """
    
    # scale and transform input features
    mscale_min = 0.3440976088844191
    scaled_mcc = (mcc_desired+1)/2
    scaled_mcc = np.log((scaled_mcc+mscale_min)/(1-scaled_mcc+mscale_min))
    log_inc = np.log(increase)
    
    # inverse regress sample size
    increase_effects = 0.704729538709909 * lf_increase + 0.315857162659738 * log_inc - 0.0927955725385892 * lf_increase * log_inc
    n_samples = (scaled_mcc + 1.3675613850217 - increase_effects) / 0.0193158965178381
    n_samples[n_samples < 0] = 0
    n_samples[0.0927955725385892 * lf_increase * log_inc > scaled_mcc + 1.3675613850217] = 0
    return np.round(n_samples)
