""""
This filw defines multiple models
for statistical analysis of compositional changes


:authors: Benjamin Schubert
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
from tensorflow_probability.python.edward2 import interception
from tensorflow_probability.python.edward2.program_transformations import _get_function_inputs

from util.Result import MCMCResult, MAPResult

tfd = tfp.distributions
tfb = tfp.bijectors


class CompositionDE:
    """"
    implements statistical model and
    test statistics for compositional differential change analysis

    """

    def __init__(self, X, y, n_total, Z=None, dtype=tf.float32):
        """

        :param X: Numpy Design NxD matrix of independent variables of interest
        :param Z: Numpy Design NxB matrix of confounders
        :param y: Numpy NxK Matrix of dependent variables
        :param n_total: Numpy Nx1 Vector of total observed counts
        """
        self.X = tf.cast(X, dtype)
        self.y = tf.cast(y, dtype)
        self.n_total = tf.cast(n_total, dtype)
        self.Z = Z if Z is None else tf.cast(Z, dtype)
        self.global_confounder = Z is not None
        self.dtype = dtype

        N, D = X.shape
        K = y.shape[1]

        # Check input data
        if N != y.shape[0]:
            raise ValueError("Wrong input dimensions X[{},:] != y[{},:]".format(X.shape[0], y.shape[0]))

        if N != len(n_total):
            raise ValueError("Wrong input dimensions X[{},:] != n_total[{}]".format(X.shape[0], len(n_total)))

        if self.global_confounder and Z.shape[0] != N:
            raise ValueError("Wrong input dimensions X[{},:] != Z[{},:]".format(X.shape[0], Z.shape[0]))

        # Model definition
        def define_model(X,n_total, K, Z=Z):
            N,D = X.shape
            sigma_alpha = ed.HalfCauchy(tf.zeros([K], dtype=dtype), tf.ones([K], dtype=dtype)*5, name="sigma_alpha")
            sigma_beta = ed.HalfCauchy(tf.zeros([K], dtype=dtype), tf.ones([K], dtype=dtype)*5, name="sigma_beta")
            nu = ed.HalfCauchy(tf.zeros([D], dtype=dtype), tf.ones([D], dtype=dtype), name="nu")

            alpha = ed.Normal(loc=tf.zeros([K], dtype=dtype), scale=sigma_alpha, name="alpha")
            beta = ed.Normal(loc=tf.zeros([D, K], dtype=dtype), scale=tf.tensordot(nu, sigma_beta, axes=0), name="beta")

            if Z is not None:
                B = Z.shape[1]
                sigma_gamma = ed.HalfCauchy(tf.zeros([B], dtype=dtype), tf.ones([B], dtype=dtype)*5, name="sigma_gamma")
                gamma = ed.Normal(loc=tf.zeros([B], dtype=dtype), scale=sigma_gamma, name="gamma")
                concentration_ = tf.exp(alpha + tf.matmul(X, beta) + tf.matmul(X, gamma))
            else:
                concentration_ = tf.exp(alpha + tf.matmul(X, beta))

            # Likelihood
            predictions = ed.DirichletMultinomial(n_total, concentration=concentration_, name="predictions")
            return predictions


        # Joint posterior distribution
        self.log_joint = ed.make_log_joint_fn(define_model)
        # Function to compute log posterior probability

        if self.global_confounder:
            self.target_log_prob_fn = lambda alpha_, beta_, gamma_, \
                                             sigma_alpha_, sigma_beta_, \
                                             sigma_gamma_, nu_: self.log_joint(X=self.X,
                                                                               n_total=self.n_total,
                                                                               K=K,
                                                                               Z=self.Z,
                                                                               predictions=self.y,
                                                                               alpha=alpha_,
                                                                               beta=beta_,
                                                                               sigma_alpha=sigma_alpha_,
                                                                               sigma_beta=sigma_beta_,
                                                                               nu=nu_,
                                                                               gamma=gamma_,
                                                                               sigma_gamma=sigma_gamma_
                                                                               )
        else:
            self.target_log_prob_fn = lambda alpha_, beta_,  \
                                             sigma_alpha_, sigma_beta_, \
                                             nu_: self.log_joint(X=self.X,
                                                                 n_total=self.n_total,
                                                                 K=K,
                                                                 Z=self.Z,
                                                                 predictions=self.y,
                                                                 alpha=alpha_,
                                                                 beta=beta_,
                                                                 sigma_alpha=sigma_alpha_,
                                                                 sigma_beta=sigma_beta_,
                                                                 nu=nu_)

    def sample(self, n_iterations=int(10e3), n_burn=int(5e3), n_leapfrog=10, n_chains=2):
        """
        HMC sampling of the model

        :param n_iterations: number of HMC iterations
        :param n_burning: number of burn-in iterations
        :param n_leapfrog: number of leap-frog steps per iteration
        :param n_chains: number of MCMC chains (current supports only sampling from one chain)
        :return: dict of parameters
        """
        # TODO: Add support for multi-chain sampling (currently Jan 30th 2019 not supported for Edward2 models)
        N,D = self.X.shape
        K = self.y.shape[1]
        dtype = self.dtype
        target_log_prob_fn = self.target_log_prob_fn

        alpha_size = [K]
        sigma_alpha_size = [K]
        beta_size = [D, K]
        sigma_beta_size = [K]
        nu_size = [D]

        param_names = ["alpha", "beta", "sigma_alpha", "sigma_beta", "nu"]
        init = [tf.zeros(alpha_size, name="init_alpha", dtype=dtype),
                tf.zeros(beta_size, name="init_beta", dtype=dtype),
                tf.ones(sigma_alpha_size, name="init_sigma_alpha", dtype=dtype),
                tf.ones(sigma_beta_size, name="init_sigma_beta", dtype=dtype),
                tf.ones(nu_size, name="init_nu", dtype=dtype),
                ]
        unconstraining_bijectors = [
            tfb.Identity(),
            tfb.Identity(),
            tfb.Softplus(),
            tfb.Softplus(),
            tfb.Softplus(),
            ]

        if self.global_confounder:
            B = self.Z.shape[1]
            gamma_size = [B]
            sigma_gamma_size = [B]
            param_names = [*param_names, "gamma", "sigma_gamma"]
            init = [*init,
                    tf.zeros(gamma_size, name="init_gamma", dtype=dtype),
                    tf.ones(sigma_gamma_size, name="init_sigma_gamma", dtype=dtype),
                    ]
            unconstraining_bijectors = [*unconstraining_bijectors,
                                        tfb.Identity(),
                                        tfb.Softplus()]

        # Define step size variable for automatic adjustment during HMC sampling
        # target acceptance rate is around 0.75 per default
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            step_size = tf.get_variable(
                name='step_size',
                initializer=tf.constant(0.5, dtype=tf.float32),
                trainable=False,
                use_resource=True
            )

        # HMC transition kernel

        kernel = tfp.mcmc.TransformedTransitionKernel(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(int(0.8*n_burn)),
                num_leapfrog_steps=n_leapfrog,
                state_gradients_are_stopped=True),
            bijector=unconstraining_bijectors)

        # kernel = tfp.mcmc.HamiltonianMonteCarlo(
        #         target_log_prob_fn=target_log_prob_fn,
        #         step_size=step_size,
        #         step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(int(0.8*n_burn)),
        #         num_leapfrog_steps=n_leapfrog,
        #         state_gradients_are_stopped=True)

        # Define the chain states
        states, kernel_results = tfp.mcmc.sample_chain(
            num_results=n_iterations,
            num_burnin_steps=n_burn,
            kernel=kernel,
            current_state=init)

        # Initialize any created variables for preconditions
        init_g = tf.global_variables_initializer()

        # Run the chain
        with tf.Session() as sess:
            sess.run(init_g)
            states = sess.run([*states])

        return MCMCResult(int(N), dict(zip(param_names, states)))

    def find_MAP(self, n_iterations=3000, optimizer=tf.train.AdamOptimizer(learning_rate=0.01)):
        """
        Returns the MAP estimate of the model
        :param n_iterations: Max number of gradient steps
        :param optimizer: Tensorflow Optimizer class
        :return: dict of point estimate parameters
        """

        N,D = self.X.shape
        K = self.y.shape[1]
        dtype = self.dtype
        target_log_prob_fn = self.target_log_prob_fn

        alpha_size = [K]
        sigma_alpha_size = [K]
        beta_size = [D, K]
        sigma_beta_size = [K]
        nu_size = [D]
        loss_history = [np.inf]

        param_names = ["alpha", "beta", "sigma_alpha", "sigma_beta", "nu"]
        params = [tf.Variable(tf.ones(alpha_size, dtype=dtype), name="alpha"),
                tf.Variable(tf.ones(beta_size, dtype=dtype), name="beta"),
                tf.Variable(tf.ones(sigma_alpha_size, dtype=dtype), name="sigma_alpha", constraint=tfb.Softplus()),
                tf.Variable(tf.ones(sigma_beta_size, dtype=dtype), name="sigma_beta", constraint=tfb.Softplus()),
                tf.Variable(tf.ones(nu_size, dtype=dtype), name="nu", constraint=tfb.Softplus()),
                ]
        if self.global_confounder:
            B = self.Z.shape[1]
            gamma_size = [B]
            sigma_gamma_size = [B]
            param_names = [*param_names, "gamma", "sigma_gamma"]
            params = [*params,
                    tf.Variable(tf.ones(gamma_size, dtype=dtype), name="gamma"),
                    tf.Variable(tf.ones(sigma_gamma_size, dtype=dtype),
                                name="init_sigma_gamma", constraint=tfb.Softplus()),
                    ]

        loss = -target_log_prob_fn(*params)
        train_op = optimizer.minimize(loss)

        # Initialize any created variables for preconditions
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for t in range(n_iterations):
                sess.run(train_op)

                if t % 5 == 0 or t == n_iterations - 1:
                    state_ = sess.run([*params, loss])
                    loss_history.append(state_[-1])
                    #if np.abs(state_[-1] - loss_history[-2]) < 10e-6:
                    #    print("converged")
                    #    break
                    print("Iteration: {:>4}  Loss: {:.3f}: Diff:{:.3f}".format(t,  state_[-1],
                                                                               np.abs(state_[-1] - loss_history[-2])))

            # get final MAP estimats
            paramsMAP = {param_name:sess.run(param) for param_name, param in zip(param_names, params)
                         if not (param_name.startswith("sigma") or param_name.startswith("nu"))}

            # calculate observed Fischer information matrix and approximate sd of parameters
            hessians = tf.hessians(loss, [param for param_name, param in zip(param_names, params)
                                          if not (param_name.startswith("sigma") or param_name.startswith("nu"))])
            sds = {}
            for param_name, hessian in zip(param_names, hessians):
                # get hessian of parameters at MAP
                if len(hessian.shape)>2:
                    # TODO: not sure this is correct for Beta matrix parameters.
                    # TODO: All variance components have nan
                    hessian = tf.reshape(hessian, [hessian.shape[0]*hessian.shape[1], -1])
                FIM = tf.linalg.inv(hessian)
                sd = tf.math.sqrt(tf.diag_part(FIM))
                sds["sd_{}".format(param_name)] = sess.run(sd)

        return MAPResult(int(N), {"params":paramsMAP, "sds":sds})


if __name__ == "__main__":
    from scipy.special import softmax
    import numpy as np
    import arviz as az
    import pandas as pd

    # Settings
    D = 2  # number of dimensions
    N = 100  # number of datapoints to generate
    K = 10
    n_total = [1000]*N

    noise_std_true = 1.0

    # Generate data
    b_true = np.random.randn(K).astype(np.float32)  # bias (alpha)
    w_true = np.random.randn(D, K).astype(np.float32)  # weights (beta)
    x = np.random.randn(N, D).astype(np.float32)
    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)
    concentration = softmax(np.matmul(x, w_true) + b_true + noise)
    y = np.zeros([N, K], dtype=np.float32)
    for i in range(N):
        y[i, :] = np.random.multinomial(n_total[i], concentration[i, :])

    # Initialize model
    model = CompositionDE(x, y, n_total)
    #params = model.sample()
    #print(params)
    params = model.find_MAP()

    pd.set_option('display.max_columns', None)

    print(params)
    #print(params.raw_params)



    #dataset = az.convert_to_inference_data(params)
    #az.plot_trace(dataset)

