from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental import edward2 as ed

from util import result_classes as res
from model import dirichlet_models as mod

tfd = tfp.distributions
tfb = tfp.bijectors

#%%
# Testing
from util import compositional_analysis_generation_toolbox as gen

n = 2

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

model = mod.NoBaselineModelNoEdward(x, y)
params_mcmc = model.sample(num_results=int(12), n_burnin=10)
print(params_mcmc)

#%%
model_2 = mod.NoBaselineModel(x, y)
params_mcmc_2 = model_2.sample(num_results=int(12), n_burnin=10)
print(params_mcmc_2)


#%%
D = x.shape[1]
K = y.shape[1]
N = y.shape[0]
dtype = tf.float32
beta_size = [D,K]
alpha_size = [1,K]
#tf.random.set_seed(5678)

test_model = tfd.JointDistributionNamed(dict(
    alpha_=tfd.Independent(
        tfd.Normal(
            loc=tf.zeros(alpha_size),
            scale=tf.ones(alpha_size) * 5,
            name="alpha"),
        reinterpreted_batch_ndims=2),

    mu_b=tfd.Independent(
        tfd.Normal(loc=tf.zeros(1, dtype=dtype),
                   scale=tf.ones(1, dtype=dtype),
                   name="mu_b"),
        reinterpreted_batch_ndims=1),

    sigma_b=tfd.Independent(
        tfd.HalfCauchy(tf.zeros(1, dtype=dtype),
                       tf.ones(1, dtype=dtype),
                       name="sigma_b"),
        reinterpreted_batch_ndims=1),

    b_offset=tfd.Independent(
        tfd.Normal(
            loc=tf.zeros([D, K], dtype=dtype),
            scale=tf.ones([D, K], dtype=dtype),
            name="b_offset"),
        reinterpreted_batch_ndims=2),

    b_raw=lambda mu_b, sigma_b, b_offset: tfd.Independent(
        tfd.Deterministic(
            mu_b[..., tf.newaxis]
            + sigma_b[..., tf.newaxis]
            * b_offset,
            name="b_raw"),
        reinterpreted_batch_ndims=2),

    # Spike-and-slab
    sigma_ind_raw=tfd.Independent(
        tfd.Normal(
            loc=tf.zeros(shape=[D, K], dtype=dtype),
            scale=tf.ones(shape=[D, K], dtype=dtype),
            name='sigma_ind_raw'),
        reinterpreted_batch_ndims=2),

    ind=lambda sigma_ind_raw: tfd.Independent(
        tfd.Deterministic(
            tf.exp(sigma_ind_raw*50) / (1 + tf.exp(sigma_ind_raw*50)),
            name="ind"),
        reinterpreted_batch_ndims=2),

    # Betas
    beta_=lambda ind, b_raw: tfd.Independent(
        tfd.Deterministic(
            ind*b_raw,
            name="beta"),
        reinterpreted_batch_ndims=2),

    # concentration
    concentration_=lambda alpha_, beta_: tfd.Independent(
        tfd.Deterministic(
            tf.exp(alpha_ + tf.matmul(tf.cast(x, dtype), beta_)),
            name="concentration_"),
        reinterpreted_batch_ndims=2),

    # Cell count prediction via DirMult
    predictions=lambda concentration_: tfd.Independent(
        tfd.DirichletMultinomial(
            total_count=tf.cast(n_total, dtype),
            concentration=concentration_,
            name="predictions"),
        reinterpreted_batch_ndims=1)

))

params = dict(alpha_=tf.zeros(alpha_size, name='init_alpha', dtype=dtype),
              mu_b=tf.zeros(1, name="init_mu_b", dtype=dtype),
              sigma_b=tf.ones(1, name="init_sigma_b", dtype=dtype),
              b_offset=tf.zeros(beta_size, name='init_b_offset', dtype=dtype),
              #b_raw=tf.zeros(beta_size, name='init_b_raw'),
              sigma_ind_raw=tf.zeros(beta_size, name='init_sigma_ind_raw', dtype=dtype),
              #ind=tf.ones(beta_size, name='init_ind', dtype=dtype)*0.5,
              #beta=tf.zeros(beta_size, name='init_beta'),
              #predictions=tf.ones([N, K], name='init_predictions')
              )

#%%
test_sam = test_model.sample()
print([x.shape for x in test_sam.values()])
print(test_model.log_prob(params))

#%%
import matplotlib.pyplot as plt

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
print(b)
#%%

model_seq = tfd.JointDistributionSequential([
    tfd.Normal(loc=0., scale=1., name="a"),
    lambda a_: tfd.Normal(loc=a_, scale=1., name="b")])


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

states_name, kernel_results_name = sample_mcmc(init_name, target_log_prob_fn_name)
print(states_seq)




