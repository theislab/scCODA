import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import importlib
import time
import patsy as pt

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental import edward2 as ed

from sccoda.util import result_classes as res
from sccoda.util import data_generation as gen
from sccoda.util import comp_ana as mod
from sccoda.util import cell_composition_data as dat
from sccoda.util import data_visualization as viz
from sccoda.model import dirichlet_models as dm

tfd = tfp.distributions
tfb = tfp.bijectors

pd.options.display.float_format = '{:10,.3f}'.format
pd.set_option('display.max_columns', 10)

#%%

data_path = "C:/Users/Johannes/Documents/PhD/data/rosen_mincount10_maxee2_trim200_results_forpaper/"

otu_file = "rosen_mincount10_maxee2_trim200.otu_table.99.denovo.rdp_assigned.paper_samples.txt"

with open(data_path + otu_file, "rb") as file:
    otus = pd.read_csv(file, sep="\t", index_col=0)

otus = otus[~otus.index.str.endswith('TI')]
otus = otus[~otus.index.str.endswith('F1')]
otus = otus[~otus.index.str.endswith('F1T')]
otus = otus[~otus.index.str.endswith('SI')]

split = otus.index.str.extract(r"^([0-9\-]*)([A-Z]+)$")

split.index = otus.index

otus.loc[:, "location"] = split.loc[:, 1]

otus.index = split.loc[:, 0]

print(pd.unique(otus.index))

print(pd.unique(otus["location"]))


#%%
meta_file = "patient_clinical_metadata.csv"

with open(data_path + meta_file, "rb") as file:
    metadata = pd.read_csv(file, sep=",", index_col=0)

print(metadata)

meta_rel = metadata[(~pd.isna(metadata["mbs_consolidated"])) & (~pd.isna(metadata["bal"]))]

print(meta_rel)

data = pd.merge(otus, meta_rel, right_index=True, left_index=True)

data_bal = data.loc[data["location"] == "B"]

print(np.sum(data_bal, axis=0))

col = metadata.columns[metadata.columns.isin(data_bal.columns)].tolist() + ["location"]

print(col)

# Remove otus with low counts (<100 total) --> Leaves 250 OTUs
# Leaving in low expression OTUs leads to nonconvergence for shorter chains (1000 samples),
# longer chains cant be done on my computer

counts_bal = data_bal.iloc[:, :-33]

counts_bal = counts_bal.loc[:, np.sum(counts_bal, axis=0) >= 100]

data_bal_expr = pd.merge(counts_bal, data_bal.loc[:, col], right_index=True, left_index=True)

print(data_bal_expr)

data_sccoda = dat.from_pandas(data_bal_expr, col)

print(data_sccoda.X.shape)

# Free up some memory

del([counts_bal, data, data_bal, metadata, meta_rel, file, otus, split])

#%%

# Experimental model configuration


class NoBaselineModelExperimental(dm.CompositionalModel):

    """"
    implements statistical model for compositional differential change analysis without specification of a baseline cell type
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor of model class

        Parameters
        ----------
        args -- arguments passed to top-level class
        kwargs -- arguments passed to top-level class
        """
        super(self.__class__, self).__init__(*args, **kwargs)

        self.baseline_index = None
        dtype = tf.float32

        # All parameters that are returned for analysis
        self.param_names = ["alpha", "mu_b", "sigma_b", "b_offset", "ind_raw",
                            "ind", "b_raw", "beta", "concentration", "prediction"]

        def define_model(x, n_total, K):
            """
            Model definition in Edward2

            Parameters
            ----------
            x -- numpy array [NxD]
                covariate matrix
            n_total -- numpy array [N]
                number of cells per sample
            K -- int
                Number of cell types
            """

            N, D = x.shape
            dtype = tf.float32

            # normal prior on bias
            alpha = ed.Normal(loc=tf.zeros([K]), scale=tf.ones([K])*5, name="alpha")

            # Noncentered parametrization for raw slopes (before spike-and-slab)
            mu_b = ed.Normal(loc=tf.zeros(1, dtype=dtype), scale=tf.ones(1, dtype=dtype), name="mu_b")
            sigma_b = ed.HalfCauchy(tf.zeros(1, dtype=dtype), tf.ones(1, dtype=dtype), name="sigma_b")
            b_offset = ed.Normal(loc=tf.zeros([D, K], dtype=dtype), scale=tf.ones([D, K], dtype=dtype), name="b_offset")

            b_raw = mu_b + sigma_b * b_offset

            # Spike-and-slab priors
            sigma_ind_raw = ed.Normal(
                loc=tf.zeros(shape=[D, K], dtype=dtype),
                scale=tf.ones(shape=[D, K], dtype=dtype),
                name='sigma_ind_raw')
            ind_t = sigma_ind_raw*10
            ind = tf.exp(ind_t) / (1 + tf.exp(ind_t))

            # Calculate betas
            beta = ind * b_raw

            # Concentration vector from intercepts, slopes
            concentration_ = tf.exp(alpha + tf.matmul(x, beta))

            # Cell count prediction via DirMult
            predictions = ed.DirichletMultinomial(n_total, concentration=concentration_, name="predictions")
            return predictions

        # Joint posterior distribution
        self.log_joint = ed.make_log_joint_fn(define_model)

        # Function to compute log posterior probability
        self.target_log_prob_fn = lambda alpha_, mu_b_, sigma_b_, b_offset_, sigma_ind_raw_:\
            self.log_joint(x=self.x,
                           n_total=self.n_total,
                           K=self.K,
                           predictions=self.y,
                           alpha=alpha_,
                           mu_b=mu_b_,
                           sigma_b=sigma_b_,
                           b_offset=b_offset_,
                           sigma_ind_raw=sigma_ind_raw_,
                           )

        alpha_size = [self.K]
        beta_size = [self.D, self.K]

        # MCMC starting values
        self.params = [tf.random.normal(alpha_size, 0, 1, name='init_alpha'),
                       tf.zeros(1, name="init_mu_b", dtype=dtype),
                       tf.ones(1, name="init_sigma_b", dtype=dtype),
                       tf.random.normal(beta_size, 0, 1, name='init_b_offset'),
                       tf.zeros(beta_size, name='init_sigma_ind_raw', dtype=dtype),
                       ]

    # Calculate predicted cell counts (for analysis purposes)
    def get_y_hat(self, states_burnin, num_results, n_burnin):
        """
        Calculate predicted cell counts (for analysis purposes) and add intermediate parameters to MCMC results

        Parameters
        ----------
        states_burnin -- List
            MCMC chain without burnin samples
        num_results -- int
            Chain length (with burnin)
        n_burnin -- int
            Number of burnin samples

        Returns
        -------
        y_mean
            predicted cell counts
        """

        chain_size_y = [num_results - n_burnin, self.N, self.K]

        alphas = states_burnin[0]
        alphas_final = alphas.mean(axis=0)

        ind_raw = states_burnin[4] * 50
        mu_b = states_burnin[1]
        sigma_b = states_burnin[2]
        b_offset = states_burnin[3]

        ind_ = np.exp(ind_raw) / (1 + np.exp(ind_raw))

        b_raw_ = mu_b.reshape((num_results - n_burnin, 1, 1)) + np.einsum("...jk, ...j->...jk", b_offset, sigma_b)

        beta_ = np.einsum("..., ...", ind_, b_raw_)

        conc_ = np.exp(np.einsum("jk, ...kl->...jl", self.x, beta_)
                       + alphas.reshape((num_results - n_burnin, 1, self.K)))

        predictions_ = np.zeros(chain_size_y)
        for i in range(num_results-n_burnin):
            pred = tfd.DirichletMultinomial(self.n_total, conc_[i, :, :]).mean().numpy()
            predictions_[i, :, :] = pred

        betas_final = beta_.mean(axis=0)
        states_burnin.append(ind_)
        states_burnin.append(b_raw_)
        states_burnin.append(beta_)
        states_burnin.append(conc_)
        states_burnin.append(predictions_)

        concentration = np.exp(np.matmul(self.x, betas_final) + alphas_final).astype(np.float32)
        y_mean = concentration / np.sum(concentration, axis=1, keepdims=True) * self.n_total.numpy()[:, np.newaxis]
        return y_mean


def model_experimental(data, formula, baseline_index=None):

    cell_types = data.var.index.to_list()

    # Get count data
    data_matrix = data.X.astype("float32")

    # Build covariate matrix from R-like formula
    covariate_matrix = pt.dmatrix(formula, data.obs)
    covariate_names = covariate_matrix.design_info.column_names[1:]
    covariate_matrix = covariate_matrix[:, 1:]

    return NoBaselineModelExperimental(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                       cell_types=cell_types, covariate_names=covariate_names, formula=formula)

#%%

# Set logitNormal parameter to 10

model_mbs = model_experimental(data_sccoda, "mbs_consolidated", baseline_index=None)

result_mbs = model_mbs.sample_nuts(num_results=int(1000), n_burnin=0, num_adapt_steps=500)

result_mbs.summary_extended(hdi_prob=0.95)