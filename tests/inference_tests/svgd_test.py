
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import time
import functools

tfd = tfp.distributions


#%%

# Implementation of svgd. Credits to:
# https://github.com/janhuenermann/svgd,
# https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/tree/master/python


class SVGD():

    def __init__(self, model, latent_distribution):
        self.model = model
        self.latent_distribution = latent_distribution
        self.part_history = {}

        tf.random.set_seed(1234)

    def pairwise_distance(self, x):
        # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
        norm = tf.reduce_sum(x * x, 1)
        norm = tf.reshape(norm, [-1, 1])
        return norm - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(norm)

    def kernel_fn(self, x, h='median'):
        pdist = self.pairwise_distance(x)

        if h == 'median':
            # https://stackoverflow.com/questions/43824665/tensorflow-median-value
            lower = tfp.stats.percentile(pdist, 50.0, interpolation='lower')
            higher = tfp.stats.percentile(pdist, 50.0, interpolation='higher')

            median = (lower + higher) / 2.
            median = tf.cast(median, tf.float32)

            h = tf.sqrt(0.5 * median / tf.math.log(x.shape[0] + 1.))
            tf.stop_gradient(h)

        return tf.exp(-pdist / h ** 2 / 2)

    # @tf.function
    def svgd(self, samples, log_prob_fn):
        num_particles, dim = samples.shape

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(samples)

            kernel = self.kernel_fn(samples)

            log_prob = -log_prob_fn(samples)

            val_grad = tfp.math.value_and_gradient(log_prob_fn, samples, use_gradient_tape=True)

        kernel_grad = tape.gradient(kernel, samples)
        log_prob_grad = tape.gradient(log_prob, samples)

        # kernel_grad = tf.gradients(kernel, samples)
        # log_prob_grad = tf.gradients(log_prob, samples)

        tf.print(f"dlnp: {log_prob_grad}")
        tf.print(f"kxy: {kernel}")
        tf.print(f"dxkxy: {kernel_grad}")

        return (tf.matmul(kernel, log_prob_grad) + kernel_grad) / num_particles

    # @tf.function
    def run_svgd_one_step(self, particles):
        grads = self.svgd(particles, self.target_density_log)
        tf.print(f"dtheta: {grads}")
        return grads

    def target_density(self, x):
        return self.model.prob(x)

    def target_density_log(self, x):
        return self.model.log_prob(x)

    def run_svgd_adagrad(self, n_steps, n_particles, step_size=1e-3, record_step=50, alpha=0.9):

        start = time.time()

        # initialize
        particles = tf.cast(self.latent_distribution.sample(sample_shape=n_particles, name="particles"), tf.float32)
        print(particles.numpy())

        self.part_history[0] = particles

        # run
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for i in range(n_steps):

            grads = self.run_svgd_one_step(particles)

            # adagrad
            if i == 0:
                historical_grad = historical_grad + grads ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grads ** 2)
            adj_grad = tf.divide(grads, fudge_factor + tf.sqrt(historical_grad))
            print(f"adj_grad: {adj_grad}")
            particles = particles + step_size * adj_grad

            if (i+1) % record_step == 0:
                label = f'timestep {i + 1}/{n_steps}'
                print(label)
                self.part_history[i+1] = particles

        print(f"SVGD finished! {time.time()-start}s")

    def run_svgd(self, n_steps, n_particles, step_size=1e-3, record_step=50):

        start = time.time()

        # initialize
        #particles = tf.cast(self.latent_distribution.sample(sample_shape=n_particles, name="particles"), tf.float32)
        particles = tf.cast(tf.Variable(np.array([[-1.2650278, -0.8826601], [0.45325747, 1.1934614]])), tf.float32)
        print(particles.numpy())
        self.part_history[0] = particles

        # run
        # gradient descent
        for i in range(n_steps):

            grads = self.run_svgd_one_step(particles)
            particles = particles + step_size * grads

            if (i + 1) % record_step == 0:
                label = f'timestep {i + 1}/{n_steps}'
                print(label)
                self.part_history[i + 1] = particles

        print(f"SVGD finished! {time.time()-start}s")


#%%

def density2image(model, size, extent):
    grid_x = tf.linspace(start=extent[0], stop=extent[1], num=size)
    grid_y = tf.linspace(start=extent[3], stop=extent[2], num=size)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    grid_x, grid_y = tf.reshape(grid_x, shape=(size, size, 1)), tf.reshape(grid_y, shape=(size, size, 1))
    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.reshape(grid, shape=(size ** 2, 2))

    p = model(grid)
    return tf.reshape(p, shape=(size, size)).numpy()


def plot_svgd_2d(svgd_obj):
    n_plots = len(svgd_obj.part_history)
    nrows = int(np.ceil(np.sqrt(n_plots)))
    ncols = int(np.ceil(n_plots / nrows))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    fig.set_tight_layout(True)

    n = 0

    x_lower = np.min([[x[0] for x in p] for p in svgd_obj.part_history.values()])
    x_higher = np.max([[x[0] for x in p] for p in svgd_obj.part_history.values()])
    y_lower = np.min([[x[1] for x in p] for p in svgd_obj.part_history.values()])
    y_higher = np.max([[x[1] for x in p] for p in svgd_obj.part_history.values()])
    extent = [x_lower, x_higher, y_lower, y_higher]

    for i, dat in svgd_obj.part_history.items():
        dat = dat.numpy()

        ax_ = ax.flat[n]

        ax_.imshow(density2image(svgd_obj.target_density, size=100, extent=extent), extent=extent)
        ax_.grid(zorder=10, color='#cccccc', alpha=0.5, linestyle='-.', linewidth=0.7)

        ax_.plot(dat[..., 0], dat[..., 1], '.', c='w', alpha=0.75, markersize=3)

        ax_.set_xlabel('timestep {0}'.format(i))

        n += 1

    plt.show()


#%%
"""
# Target distribution
toy_dist = tfd.Mixture(
    cat=tfd.Categorical(probs=[0.5, 0.5]),
    components=[
        tfd.MultivariateNormalDiag(loc=tf.constant([-1., +1]), scale_diag=tf.constant([0.2, 0.2])),
        tfd.MultivariateNormalDiag(loc=tf.constant([+1., -1]), scale_diag=tf.constant([0.2, 0.2]))
    ])

# Initial distribution
latent_distribution = tfd.Normal(
    loc=tf.zeros(shape=[2]),
    scale=2 * tf.ones(shape=[2]))

#%%
opt = SVGD(toy_dist, latent_distribution)
#%%
opt.run_svgd(n_steps=100, n_particles=500, record_step=50)
#%%
plot_svgd_2d(opt)

#%%
opt2 = SVGD(toy_dist, latent_distribution)
# %%
opt2.run_svgd_adagrad(n_steps=1000, n_particles=500, record_step=50)

#%%
plot_svgd_2d(opt2)


#%%

# Target distribution
toy_dist = tfd.Mixture(
    cat=tfd.Categorical(probs=[1/3, 2/3]),
    components=[
        tfd.Normal(loc=tf.constant(-2.), scale=tf.constant(1.)),
        tfd.Normal(loc=tf.constant(2.), scale=tf.constant(1.))
    ])

# Initial distribution
latent_distribution = tfd.Normal(
    loc=-10*tf.ones(shape=[1]),
    scale=tf.ones(shape=[1]))

#%%
opt = SVGD(toy_dist, latent_distribution)
#%%
opt.run_svgd_adagrad(n_steps=2000, n_particles=100, record_step=50)

#%%
import seaborn as sns

def plot_svgd_1d(svgd_obj):
    n_plots = len(svgd_obj.part_history)
    nrows = int(np.ceil(np.sqrt(n_plots)))
    ncols = int(np.ceil(n_plots / nrows))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    fig.set_tight_layout(True)

    n = 0

    for i, dat in svgd_obj.part_history.items():
        dat = [x[0] for x in dat.numpy()]

        ax_ = ax.flat[n]

        target_samples = svgd_obj.model.sample(1000)
        sns.kdeplot(target_samples, ax=ax_, color="green", linestyle="--")
        sns.kdeplot(dat, ax=ax_, color="red", linestyle="-")

        ax_.set_xlabel('timestep {0}'.format(i))

        n += 1

    plt.show()
#%%
plot_svgd_1d(opt)"""



#%%

if __name__ == '__main__':
    A = np.array([[0.2260, 0.1652],[0.1652, 0.6779]]).astype("float32")
    mu = np.array([-0.6871, 0.8010]).astype("float32")

    # Target distribution
    toy_dist2 = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=A)

    # Initial distribution
    latent_distribution2 = tfd.Normal(loc=np.zeros([2]), scale=np.ones([2]))

    opt2 = SVGD(toy_dist2, latent_distribution2)
    opt2.run_svgd_adagrad(n_steps=5, n_particles=2, record_step=1)

    #plot_svgd_2d(opt2)
