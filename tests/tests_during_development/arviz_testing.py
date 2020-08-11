import arviz as az
import pandas as pd
import numpy as np
import importlib
from scdcdm.util import result_classes as res
from scdcdm.util import comp_ana as mod
from scdcdm.util import data_generation as gen
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:10,.3f}'.format
pd.set_option('display.max_columns', None)


#%%
# Artificial data

n = 3

cases = 1
K = 5
n_samples = [n, n]
n_total = np.full(shape=[2*n], fill_value=1000)

data = gen.generate_case_control(cases, K, n_total[0], n_samples,
                                 w_true=np.array([[1, 0, 0, 0, 0]]),
                                 b_true=np.log(np.repeat(0.2, K)).tolist())

print(data.uns["w_true"])
print(data.uns["b_true"])

print(data.X)
print(data.obs)


#%%
importlib.reload(mod)
importlib.reload(res)

ana = mod.CompositionalAnalysis(data, "x_0", baseline_index=2)

#%%
ca_result = ana.sample_hmc(num_results=int(1000), n_burnin=int(500))

ca_result.summary(hdi_prob=0.95)

#%%
az.plot_trace(ca_result, var_names="beta", coords={"cov": [0], "cell_type": ["0", "1", "2", "3", "4"]})
plt.show()

#%%
_, betas_df = ca_result.summary_prepare()
print(betas_df.index)
print(ca_result.posterior)

#%%

hmc_res = params_mcmc.raw_params
y_hat = params_mcmc.y_hat
baseline = False
cell_types = params_mcmc.cell_types
covariate_names = params_mcmc.covariate_names

#%%

arviz_test = az.from_dict(
    posterior={var_name: [var] for var_name, var in hmc_res.items() if
               "prediction" not in var_name},
    posterior_predictive={"prediction": [hmc_res["prediction"]]},
    observed_data={"y": data.X},
    dims={"alpha": ["cell_type"],
          "mu_b": ["1"],
          "sigma_b": ["1"],
          "b_offset": ["covariate", "cell_type"],
          "ind_raw": ["covariate", "cell_type"],
          "ind": ["covariate", "cell_type"],
          "b_raw": ["covariate", "cell_type"],
          "beta": ["cov", "cell_type"],
          "concentration": ["sample", "cell_type"],
          "prediction": ["sample", "cell_type"]
          },
    coords={"cell_type": cell_types,
            "covariate": covariate_names,
            "sample": range(data.X.shape[0])
            },

)

#%%

print(arviz_test.summary())

#%%
az.plot_trace(arviz_test, var_names=["concentration"])
plt.show()


#%%
class CAResult(az.InferenceData):

    def __init__(self, y_hat, baseline, **kwargs):

        super(self.__class__, self).__init__(**kwargs)

        self.baseline = baseline
        self.y_hat = y_hat


#%%


class ResultConverter(az.data.io_dict.DictConverter):

    def to_result_data(self, y_hat, baseline):
        return CaResult(
            y_hat, baseline=baseline,
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                #"log_likelihood": self.log_likelihood_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "prior": self.prior_to_xarray(),
                "sample_stats_prior": self.sample_stats_prior_to_xarray(),
                "prior_predictive": self.prior_predictive_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
                #"constant_data": self.constant_data_to_xarray(),
            }
        )


#%%

def result_from_dict(params, y, cell_types, covariate_names, y_hat, baseline):
    posterior = {var_name: [var] for var_name, var in params.items() if
                 "prediction" not in var_name}

    posterior_predictive = {"prediction": [params["prediction"]]}
    observed_data = {"y": y}
    dims = {"alpha": ["cell_type"],
            "mu_b": ["1"],
            "sigma_b": ["1"],
            "b_offset": ["covariate", "cell_type"],
            "ind_raw": ["covariate", "cell_type"],
            "ind": ["covariate", "cell_type"],
            "b_raw": ["covariate", "cell_type"],
            "beta": ["cov", "cell_type"],
            "concentration": ["sample", "cell_type"],
            "prediction": ["sample", "cell_type"]
            }
    coords = {"cell_type": cell_types,
              "covariate": covariate_names,
              "sample": range(y.shape[0])
              }

    return ResultConverter(posterior=posterior,
                           posterior_predictive=posterior_predictive,
                           observed_data=observed_data,
                           dims=dims,
                           coords=coords).to_result_data(y_hat, baseline)

#%%



r = result_from_dict(hmc_res, data.X, cell_types, covariate_names, y_hat, False)

print(az.summary(r, kind="stats", var_names=["alpha", "beta"]))


#%%

import pymc3 as pm
draws = 500
chains = 1

eight_school_data = {
    'J': 8,
    'y': np.array([28., 8., -3., 7., -1., 1., 18., 12.]),
    'sigma': np.array([15., 10., 16., 11., 9., 11., 10., 18.])
}

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta_tilde = pm.Normal('theta_tilde', mu=0, sd=1, shape=eight_school_data['J'])
    theta = pm.Deterministic('theta', mu + tau * theta_tilde)
    pm.Normal('obs', mu=theta, sd=eight_school_data['sigma'], observed=eight_school_data['y'])

    trace = pm.sample(draws, chains=chains)
    prior = pm.sample_prior_predictive()
    posterior_predictive = pm.sample_posterior_predictive(trace)

    pm_data = az.from_pymc3(
            trace=trace,
            prior=prior,
            posterior_predictive=posterior_predictive,
            coords={'school': np.arange(eight_school_data['J'])},
            dims={'theta': ['school'], 'theta_tilde': ['school']},
        )
#pm_data

#%%
az.plot_posterior(pm_data)
plt.show()

#%%
data = az.load_arviz_data('centered_eight')
az.plot_posterior(data, coords={"school": ["Choate", "Deerfield"]})
plt.show()
