import arviz as az
import pandas as pd
import numpy as np
import importlib
from util import result_classes as res
from util import comp_ana as mod
from util import compositional_analysis_generation_toolbox as gen
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

ana = mod.CompositionalAnalysis(data, "x_0", baseline_index=None)

#%%
params_mcmc = ana.sample_hmc(num_results=int(1000), n_burnin=500)

params_mcmc.summary()

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

print(arviz_test.posterior)

#%%
az.plot_trace(arviz_test, var_names=["concentration"])
plt.show()


#%%
class CaResult(az.InferenceData):

    def __init__(self, **kwargs):

        super(self.__class__, self).__init__(**kwargs)

#%%


class ResultConverter(az.data.io_dict.DictConverter):

    def to_result_data(self):
        return CaResult(
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

def result_from_dict(params, y, cell_types, covariate_names):
    posterior = {var_name: [var] for var_name, var in params.items() if
                 "prediction" not in var_name}

    posterior_predictive = {"prediction": [hmc_res["prediction"]]}
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
                           coords=coords).to_result_data()

#%%



r = result_from_dict(hmc_res, data.X, cell_types, covariate_names)

print(r.posterior)