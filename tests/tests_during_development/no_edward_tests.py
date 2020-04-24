from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import importlib
import pandas as pd
from tensorflow_probability.python.experimental import edward2 as ed

from scdcdm.util import result_classes as res
from scdcdm.model import dirichlet_models as mod

tfd = tfp.distributions
tfb = tfp.bijectors

pd.set_option('display.max_columns', 500)
#%%
# Testing
from scdcdm.util import data_generation as gen

n = 5

cases = 1
K = 5
n_samples = [n, n]
n_total = np.full(shape=[2*n], fill_value=1000)

data = gen.generate_case_control(cases, K, n_total[0], n_samples,
                                 w_true=np.array([[1, 0, 0, 0, 0]]),
                                 b_true=np.log(np.repeat(0.2, K)).tolist())

x = data.obs.values
y = data.X
print(x)
print(y)

#%%
importlib.reload(mod)
importlib.reload(res)


model = mod.NoBaselineModelNoEdward(x, y)
result = model.sample_hmc(num_results=int(10000), n_burnin=5000)

result.summary()

#%%
model_2 = mod.NoBaselineModel(x, y)
print(model_2.target_log_prob_fn(model_2.params[0], model_2.params[1], model_2.params[2], model_2.params[3], model_2.params[4]))

#%%
params_mcmc_2, y_hat_2 = model_2.sample_hmc(num_results=int(10000), n_burnin=5000)
res_2 = res.CompAnaResult(params=params_mcmc_2, y_hat=y_hat_2, y=y, baseline=False,
                                     cell_types=["type0", "type1", "type2", "type3", "type4"], covariate_names=["x0"])

res_2.summary()


#%%
D = x.shape[1]
K = y.shape[1]
N = y.shape[0]
dtype = tf.float32
beta_size = [D, K]
alpha_size = [1, K]
#tf.random.set_seed(5678)

test_model = tfd.JointDistributionSequential([
    tfd.Independent(
        tfd.Normal(loc=tf.zeros(1, dtype=dtype),
                   scale=tf.ones(1, dtype=dtype),
                   name="mu_b"),
        reinterpreted_batch_ndims=1),

    tfd.Independent(
        tfd.HalfCauchy(tf.zeros(1, dtype=dtype),
                       tf.ones(1, dtype=dtype),
                       name="sigma_b"),
        reinterpreted_batch_ndims=1),

    tfd.Independent(
        tfd.Normal(
            loc=tf.zeros([D, K], dtype=dtype),
            scale=tf.ones([D, K], dtype=dtype),
            name="b_offset"),
        reinterpreted_batch_ndims=2),

    lambda b_offset, sigma_b, mu_b: tfd.Independent(
        tfd.Deterministic(
            loc=mu_b + sigma_b * b_offset,
            name="b_raw"
        ),
        reinterpreted_batch_ndims=2),

    # Spike-and-slab
    tfd.Independent(
            1 / (1 + tf.exp(tfd.Normal(
                loc=tf.zeros(shape=[D, K], dtype=dtype),
                scale=tf.ones(shape=[D, K], dtype=dtype)*50))),
            name="ind"
,
        reinterpreted_batch_ndims=2),

    # Betas
    lambda ind, b_raw: tfd.Independent(
        tfd.Deterministic(
            loc=ind*b_raw,
            name="beta"
        ),
        reinterpreted_batch_ndims=2),

    tfd.Independent(
        tfd.Normal(
            loc=tf.zeros(alpha_size),
            scale=tf.ones(alpha_size) * 5,
            name="alpha"),
        reinterpreted_batch_ndims=2),

    # concentration
    lambda alpha, beta: tfd.Independent(
        tfd.Deterministic(
            loc=tf.exp(alpha + tf.matmul(tf.cast(x, dtype), beta)),
            name="concentration"
        ),
        reinterpreted_batch_ndims=2),

    # Cell count prediction via DirMult
    lambda concentration_: tfd.Independent(
        tfd.DirichletMultinomial(
            total_count=tf.cast(n_total, dtype),
            concentration=concentration_,
            name="predictions"),
        reinterpreted_batch_ndims=1),
])
#%%

init_mu_b = tf.zeros(1, name="init_mu_b", dtype=dtype)
init_sigma_b = tf.ones(1, name="init_sigma_b", dtype=dtype)
init_b_offset = tf.zeros(beta_size, name="init_b_offset", dtype=dtype)
#init_b_offset = tf.random.normal(beta_size, 0, 1, name='init_b_offset', dtype=dtype)
init_ind = tf.ones(beta_size, name='init_ind', dtype=dtype)*0.5
init_ind_raw = tf.zeros(beta_size, name="init_ind_raw")
init_alpha = tf.zeros(alpha_size, name="init_alpha", dtype=dtype)
#init_alpha = tf.random.normal(alpha_size, 0, 1, name='init_alpha', dtype=dtype)
init_b_raw = init_mu_b + init_sigma_b * init_b_offset
init_beta = init_ind * init_b_raw
init_conc = tf.exp(init_alpha + tf.matmul(tf.cast(x, dtype), init_beta))
init_pred = tf.cast(y, dtype)


params_lp = [init_mu_b,
          init_sigma_b,
          init_b_offset,
          init_b_raw,
          init_ind,
          init_beta,
          init_alpha,
          init_conc,
          init_pred
          ]

params = [init_mu_b,
          init_sigma_b,
          init_b_offset,
          #init_b_raw,
          init_ind,
          #init_beta,
          init_alpha,
          #init_conc,
          #init_pred
          ]

#%%
test_sam = test_model.sample()
print(test_sam)
print(test_model.log_prob(params_lp))
print(test_model.resolve_graph())
print(test_model.variables)

#%%
test_model_2 = tfd.JointDistributionSequential([
    tfd.Independent(
        tfd.Normal(loc=tf.zeros(1, dtype=dtype),
                   scale=tf.ones(1, dtype=dtype),
                   name="mu_b"),
        reinterpreted_batch_ndims=1),

    tfd.Independent(
        tfd.HalfCauchy(tf.zeros(1, dtype=dtype),
                       tf.ones(1, dtype=dtype),
                       name="sigma_b"),
        reinterpreted_batch_ndims=1),

    tfd.Independent(
        tfd.Normal(
            loc=tf.zeros([D, K], dtype=dtype),
            scale=tf.ones([D, K], dtype=dtype),
            name="b_offset"),
        reinterpreted_batch_ndims=2),

    # Spike-and-slab
    tfd.Independent(
        tfd.Normal(
            loc=tf.zeros(shape=[D, K], dtype=dtype),
            scale=tf.ones(shape=[D, K], dtype=dtype)*50,
            name='ind_raw'),
        reinterpreted_batch_ndims=2),

    tfd.Independent(
        tfd.Normal(
            loc=tf.zeros(alpha_size),
            scale=tf.ones(alpha_size) * 5,
            name="alpha"),
        reinterpreted_batch_ndims=2),

    # Cell count prediction via DirMult
    lambda alpha, ind_raw, b_offset, sigma_b, mu_b: tfd.Independent(
        tfd.DirichletMultinomial(
            total_count=tf.cast(n_total, dtype),
            concentration=tf.exp(alpha
                                 + tf.matmul(tf.cast(x, dtype),
                                             (1 / (1 + tf.exp(-ind_raw)))
                                             * (mu_b + sigma_b * b_offset)
                                             )),
            name="predictions"),
        reinterpreted_batch_ndims=1),
])

params_2 = [init_mu_b,
          init_sigma_b,
          init_b_offset,
          init_ind_raw,
          init_alpha,
          init_pred
          ]

params_small = [init_mu_b,
          init_sigma_b,
          init_b_offset,
          init_ind_raw,
          init_alpha,
          # init_pred
          ]

#%%
test_sam_2 = test_model_2.sample()
#print(test_sam_2)
print(test_model_2.log_prob(params_2))
print(test_model_2.resolve_graph())
print(test_model_2.log_prob_parts(params_2))

#%%


def target_log_prob_fn_small(mu_b_, sigma_b_, b_offset_, ind_, alpha_):
    return test_model_2.log_prob((mu_b_, sigma_b_, b_offset_, ind_, alpha_, tf.cast(y, dtype)))

num_results = 10000
num_burnin_steps = 5000
step_size = 0.01
num_leapfrog_steps = 10
constraining_bijectors = [
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
        ]

hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn_small,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=hmc_kernel, bijector=constraining_bijectors)
hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel, num_adaptation_steps=int(4000), target_accept_prob=0.9)

@tf.function
def do_sampling_small():
  return tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=params_small,
      kernel=hmc_kernel)

states_small, kernel_results_small = do_sampling_small()

#%%
print(states_small)


#%%


def log_joint_old(y, alpha_, mu_b_, sigma_b_, b_offset_, ind_):
    rv_alpha = tfd.Normal(
            loc=tf.zeros(alpha_size),
            scale=tf.ones(alpha_size) * 5,
            name="alpha")

    rv_mu_b = tfd.Normal(loc=tf.zeros(1, dtype=dtype),
                   scale=tf.ones(1, dtype=dtype),
                   name="mu_b")

    rv_sigma_b = tfd.HalfCauchy(tf.zeros(1, dtype=dtype),
                       tf.ones(1, dtype=dtype),
                       name="sigma_b")

    rv_b_offset = tfd.Normal(
            loc=tf.zeros([D, K], dtype=dtype),
            scale=tf.ones([D, K], dtype=dtype),
            name="b_offset")

    rv_ind = tfd.LogitNormal(
            loc=tf.zeros(shape=[D, K], dtype=dtype),
            scale=tf.ones(shape=[D, K], dtype=dtype)*50,
            name='ind')

    beta_raw_ = mu_b_ + sigma_b_ * b_offset_
    beta_ = ind_ * beta_raw_
    concentration_ = tf.exp(alpha_ + tf.matmul(tf.cast(x, dtype), beta_))
    predictions_ = tfd.DirichletMultinomial(
            total_count=tf.cast(n_total, dtype),
            concentration=concentration_,
            name="predictions")

    return(tf.reduce_sum(rv_alpha.log_prob(alpha_))
           + tf.reduce_sum(rv_mu_b.log_prob(mu_b_))
           + tf.reduce_sum(rv_sigma_b.log_prob(sigma_b_))
           + tf.reduce_sum(rv_b_offset.log_prob(b_offset_))
           + tf.reduce_sum(rv_ind.log_prob(ind_))
           + tf.reduce_sum(predictions_.log_prob(y))
           )

init_ind_raw = tf.zeros(beta_size, name="init_ind_raw")

params_old = [
    init_pred,
    init_alpha,
    init_mu_b,
    init_sigma_b,
    init_b_offset,
    init_ind,
]
#%%

init_old = [
    init_alpha,
    init_mu_b,
    init_sigma_b,
    init_b_offset,
    init_ind,
]

print(log_joint_old(*params_old))
plp_old = lambda *args: log_joint_old(init_pred, *args)

@tf.function
def do_sampling_old():
  return tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=init_old,
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=plp_old,
          step_size=0.01,
          num_leapfrog_steps=10))

states_old, kernel_results_old = do_sampling_old()

#%%

print(states_old)



#%%

def target_log_prob_fn_2(mu_b_, sigma_b_, b_offset_, ind_, alpha_):
    b_raw_ = mu_b_ + sigma_b_ * b_offset_
    beta_ = ind_ * b_raw_
    conc_ = tf.exp(alpha_ + tf.matmul(tf.cast(x, dtype), beta_))
    return test_model.log_prob((mu_b_, sigma_b_, b_offset_, b_raw_, ind_, beta_, alpha_, conc_, tf.cast(y, dtype)))


def target_log_prob_fn(mu_b_, sigma_b_, b_offset_, ind_, alpha_):
    return test_model.log_prob((mu_b_, sigma_b_, b_offset_, ind_, alpha_, tf.cast(y, dtype)))

num_results = 5000
num_burnin_steps = 3000

@tf.function
def do_sampling_seq():
  return tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=params,
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn_2,
          step_size=0.4,
          num_leapfrog_steps=3))

states_seq, kernel_results_seq = do_sampling_seq()

#%%
print(states_seq)


#%%

def edward_model(x, n_total, K):
    """
    Model definition in Edward2
    :param x: numpy array [NxD] - covariate matrix
    :param n_total: numpy array [N] - number of cells per sample
    :param K: Number of cell types
    :return: none
    """
    N, D = x.shape

    # normal prior on bias
    alpha = ed.Normal(loc=tf.zeros([K]), scale=tf.ones([K]) * 5, name="alpha")

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
    ind_t = sigma_ind_raw * 50
    ind = tf.exp(ind_t) / (1 + tf.exp(ind_t))

    # Calculate betas
    beta = ind * b_raw

    # Concentration vector from intercepts, slopes
    concentration_ = tf.exp(alpha + tf.matmul(x, beta))

    # Cell count prediction via DirMult
    predictions = ed.DirichletMultinomial(n_total, concentration=concentration_, name="predictions")
    return predictions


# Joint posterior distribution
log_joint_ed = ed.make_log_joint_fn(edward_model)

# Function to compute log posterior probability
target_log_prob_fn_ed = lambda alpha_, mu_b_, sigma_b_, b_offset_, sigma_ind_raw_: \
    log_joint_ed(x=tf.cast(x, dtype),
                   n_total=tf.cast(n_total, dtype),
                   K=K,
                   predictions=tf.cast(y, dtype),
                   alpha=alpha_,
                   mu_b=mu_b_,
                   sigma_b=sigma_b_,
                   b_offset=b_offset_,
                   sigma_ind_raw=sigma_ind_raw_,
                   )

alpha_size = [K]
beta_size = [D, K]

# MCMC starting values
params_ed = [tf.zeros(alpha_size, name='init_alpha', dtype=dtype),
             # tf.random.normal(alpha_size, 0, 1, name='init_alpha'),
             tf.zeros(1, name="init_mu_b", dtype=dtype),
             tf.ones(1, name="init_sigma_b", dtype=dtype),
             tf.zeros(beta_size, name='init_b_offset', dtype=dtype),
             # tf.random.normal(beta_size, 0, 1, name='init_b_offset'),
             tf.zeros(beta_size, name='init_sigma_ind_raw', dtype=dtype),
             ]

print(target_log_prob_fn_ed(params_ed[0], params_ed[1], params_ed[2], params_ed[3], params_ed[4]))














#%%

num_schools = 8  # number of schools
treatment_effects = np.array(
    [28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)  # treatment effects
treatment_stddevs = np.array(
    [15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)  # treatment SE

#%%

model_seq = tfd.JointDistributionSequential([
  tfd.Normal(loc=0., scale=10., name="avg_effect"),  # `mu` above
  tfd.Normal(loc=5., scale=1., name="avg_stddev"),  # `log(tau)` above
  tfd.Independent(tfd.Normal(loc=tf.zeros(num_schools),
                             scale=tf.ones(num_schools),
                             name="school_effects_standard"),  # `theta_prime`
                  reinterpreted_batch_ndims=1),
  lambda school_effects_standard, avg_stddev, avg_effect: (
      tfd.Independent(tfd.Normal(loc=(avg_effect[..., tf.newaxis] +
                                      tf.exp(avg_stddev[..., tf.newaxis]) *
                                      school_effects_standard),  # `theta` above
                                 scale=treatment_stddevs),
                      name="treatment_effects",  # `y` above
                      reinterpreted_batch_ndims=1))
])

def target_log_prob_fn_seq(avg_effect, avg_stddev, school_effects_standard):
  """Unnormalized target density as a function of states."""
  return model_seq.log_prob((
      avg_effect, avg_stddev, school_effects_standard, treatment_effects))

print(model_seq.log_prob([tf.zeros([], name='init_avg_effect'),
                     tf.zeros([], name='init_avg_stddev'),
                     tf.ones([num_schools], name='init_school_effects_standard'),
                     treatment_effects
      ]))
print(model_seq.sample())

#%%
model_named = tfd.JointDistributionNamed(dict(
  avg_effect=tfd.Normal(loc=0., scale=10., name="avg_effect"),  # `mu` above
  avg_stddev=tfd.Normal(loc=5., scale=1., name="avg_stddev"),  # `log(tau)` above
  school_effects_standard=tfd.Independent(tfd.Normal(loc=tf.zeros(num_schools),
                             scale=tf.ones(num_schools),
                             name="school_effects_standard"),  # `theta_prime`
                  reinterpreted_batch_ndims=1),
  treatment_effects=lambda school_effects_standard, avg_stddev, avg_effect: (
      tfd.Independent(tfd.Normal(loc=(avg_effect[..., tf.newaxis] +
                                      tf.exp(avg_stddev[..., tf.newaxis]) *
                                      school_effects_standard),  # `theta` above
                                 scale=treatment_stddevs),
                      name="treatment_effects",  # `y` above
                      reinterpreted_batch_ndims=1))
))

def target_log_prob_fn_named(avg_effect, avg_stddev, school_effects_standard):
  """Unnormalized target density as a function of states."""
  return model_named.log_prob((
      avg_effect, avg_stddev, school_effects_standard, treatment_effects))

print(model_named.log_prob(dict(avg_effect=tf.zeros([], name='init_avg_effect'),
                     avg_stddev=tf.zeros([], name='init_avg_stddev'),
                     school_effects_standard=tf.ones([num_schools], name='init_school_effects_standard'),
                     treatment_effects=treatment_effects
      )))
print(model_named.sample())

#%%
num_results = 5000
num_burnin_steps = 3000

# Improve performance by tracing the sampler using `tf.function`
# and compiling it using XLA.
@tf.function
def do_sampling_named():
  return tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=dict(avg_effect=tf.zeros([], name='init_avg_effect'),
                         avg_stddev=tf.zeros([], name='init_avg_stddev'),
                         school_effects_standard=tf.ones([num_schools], name='init_school_effects_standard'),
                         ),
      #current_state=(
      #    tf.zeros([], name='init_avg_effect'),
      #    tf.zeros([], name='init_avg_stddev'),
      #    tf.ones([num_schools], name='init_school_effects_standard'),
      #),
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn_named,
          step_size=0.4,
          num_leapfrog_steps=3))

states_named, kernel_results_named = do_sampling_named()

#%%
@tf.function
def do_sampling_seq():
  return tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=[
          tf.zeros([], name='init_avg_effect'),
          tf.zeros([], name='init_avg_stddev'),
          tf.ones([num_schools], name='init_school_effects_standard'),
      ],
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn_seq,
          step_size=0.4,
          num_leapfrog_steps=3))

states_seq, kernel_results_seq = do_sampling_seq()

#%%
print(states_seq)

#%%

avg_effect, avg_stddev, school_effects_standard = states

school_effects_samples = (
    avg_effect[:, np.newaxis] +
    np.exp(avg_stddev)[:, np.newaxis] * school_effects_standard)

num_accepted = np.sum(kernel_results.is_accepted)
print('Acceptance rate: {}'.format(num_accepted / num_results))

#%%
current_state = [
          tf.zeros([], name='init_avg_effect'),
          tf.zeros([], name='init_avg_stddev'),
          tf.ones([num_schools], name='init_school_effects_standard'),
      ]

current_state_2 = dict(avg_effect=tf.zeros([], name='init_avg_effect'),
                         avg_stddev=tf.zeros([], name='init_avg_stddev'),
                         school_effects_standard=tf.ones([num_schools], name='init_school_effects_standard'),
                         )

[tf.convert_to_tensor(value=x) for x in current_state_2]

#%%
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

init_state = current_state_2

if not (mcmc_util.is_list_like(init_state) or isinstance(init_state, dict)):
    init_state = [init_state]
print(init_state)

if isinstance(init_state, dict):
    init_state = {k: tf.convert_to_tensor(value=v) for k, v in init_state.items()}
else:
    init_state = [tf.convert_to_tensor(value=x) for x in init_state]
print(init_state)


#%%
# minimal example for github issue
b = np.random.normal(0., 1.)

model_seq = tfd.JointDistributionSequential([
    tfd.Normal(loc=0., scale=1., name="a"),
    lambda a: tfd.Normal(loc=a, scale=1., name="b")
])


def target_log_prob_fn_seq(a):
    return model_seq.log_prob((a, b))


init_seq = [tf.zeros([], name="init_a")]


model_name = tfd.JointDistributionNamed(dict(
    a=tfd.Normal(loc=0., scale=1., name="a"),
    b=lambda a: tfd.Normal(loc=a, scale=1., name="b")
))


def target_log_prob_fn_name(a):
    return model_name.log_prob((a, b))


init_name = dict(a=tf.zeros([], name="init_a"))

num_results = 5000
num_burnin_steps = 3000


@tf.function
def sample_mcmc(init, target_log_prob_fn):
    return tfp.mcmc.sample_chain(
          num_results=num_results,
          num_burnin_steps=num_burnin_steps,
          current_state=init,
          kernel=tfp.mcmc.HamiltonianMonteCarlo(
              target_log_prob_fn=target_log_prob_fn,
              step_size=0.01,
              num_leapfrog_steps=3))


states_seq, kernel_results_seq = sample_mcmc(init_seq, target_log_prob_fn_seq)
print(states_seq)

#%%
states_name, kernel_results_name = sample_mcmc(init_name, target_log_prob_fn_name)
print(states_seq)




