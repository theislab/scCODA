"""
File contains functions for different
simulation scenarios

:authors: Benjamin Schubert
"""
import numpy as np


def generate_simple_data(n=10, m=2, k=2, m_r=1, k_r=1):
    """
    generate simple data without correlation structures

    :param n: number of samples
    :param m: number of covariates
    :param k: number of components
    :param m_r: number of significant independent variables
    :param k_r: number of affected components
    :return: (np.array:total counts,
              np.array:matrix of independent variables,
              np.array:observed counts,
              np.array:index of significant variables,
              np.array:index of affected components)

    """
    X_cov = np.eye(m)
    X = np.random.multivariate_normal(np.zeros(m), cov=X_cov, size=n)

    m_r_idx = np.random.choice(range(m), size=m_r, replace=False)
    k_r_idx = np.random.choice(range(k), size=k_r, replace=False)

    # slope and intercepts
    alphas = np.random.uniform(-1, 1, size=k)
    betas = np.zeros((m, k))

    for i in m_r_idx:
        for j in k_r_idx:
            betas[i, j] = 2. # strong effect

    gamma = np.exp(alphas + np.matmul(X, betas))

    # sample total number of reads:
    n_total = np.random.randint(20000, 50000, size=n)
    y = np.zeros((n, k))

    for i in range(n):
        pi = np.random.dirichlet(gamma[i, :])
        y[i, :] = np.random.multinomial(n_total[i], pi)

    return X, y, None, n_total, m_r_idx, k_r_idx


def generate_correlated_covariates_data(n=10, m=2, k=2, m_r=1, k_r=1, rho=0.1):
    """
    generate data with correlation structures on covariates following Tibshirani et al (1996).

    Covaraits are drawn from MvNormal(0, Cov) Cov_ij = rho^|i-j|

    :param n: number of samples
    :param m: number of independent variables
    :param k: number of components
    :param m_r: number of significant independent variables
    :param k_r: number of affected components
    :param rho: strength of correlation between neighbouring covariates
    :return: (np.array:total counts,
              np.array:matrix of independent variables,
              np.array:observed counts,
              np.array:index of significant variables,
              np.array:index of affected components)

    """
    X_cov =np.array([[rho**np.abs(i-j) for j in range(k)] for i in range(k)])
    X = np.random.multivariate_normal(np.zeros(m), cov=X_cov, size=n)

    m_r_idx = np.random.choice(range(m), size=m_r, replace=False)
    k_r_idx = np.random.choice(range(k), size=k_r, replace=False)

    # slope and intercepts
    alphas = np.random.uniform(-1, 1, size=k)
    betas = np.zeros((m, k))

    for i in m_r_idx:
        for j in k_r_idx:
            betas[i, j] = 2. # strong effect

    gamma = np.exp(alphas + np.matmul(X, betas))

    # sample total number of reads:
    n_total = np.random.randint(3000, 50000, size=n)
    y = np.zeros((n, k))

    for i in range(n):
        pi = np.random.dirichlet(gamma[i, :])
        y[i, :] = np.random.multinomial(n_total[i], pi)

    return X, y, None, n_total, m_r_idx, k_r_idx


def scRNA_realistic_data(n=10, m=2, k=2, m_r=1, k_r=1, rho=0.1):
    """
    cell proportions taken 10x 10k PBMC healthy donors
    """
    n_total_pos = np.random.randint(15000, 30000, size=int(n/2))
    n_total_neg = np.random.randint(1000, 8000, size=int(n/2))
    cell_proportions = [0.148, 0.096, 0.094, 0.094, 0.083, 0.075, 0.074,
                        0.057, 0.051, 0.042, 0.037, 0.033, 0.031, 0.025, 0.025, 0.019, 0.016]
    y = np.zeros((n, len(cell_proportions)))
    X = np.zeros((n, 1))

    for i in range(0, int(n/2)):
        y[i, :] = np.random.multinomial(n_total_pos[i], cell_proportions)
        X[i] = 1.0
        y[int(n/2)+i, :] = np.random.multinomial(n_total_neg[i], cell_proportions)

    return X, y, None, np.array([*n_total_pos, *n_total_neg]), [], []



def scRNA_celltype_variation_without_covariat_effect(n=10, m=2, k=2, m_r=1, k_r=1, rho=0.1):
    """
        generate data that varie across individuals with/and without covariate effects
    """







def run_simulator(model, data_generator, inference_method="MAP", alpha=0.05, itr=100):
    """
     function that runs a simulation given a model class and data_generator function
     and returns the min_auc, macro_auc, and weighted auc

    :param model: Class of model to use
    :param data_generator: a function that generates the data
    :param inference_method: The method to use for inference MAP or MCMC
    :param alpha: the alpha level of significance
    :param itr: number of simulations
    :return: (min_auc, macro_auc, weighted_auc)
    """

    from sklearn import metrics

    min_auc = []
    macro_auc = []
    weighted_auc = []

    ground_trueth = []
    predictions = []

    for _ in range(itr):
        X, y, Z, n_total, m_r_idx, k_r_idx = data_generator()
        print(m_r_idx, k_r_idx)
        true = np.zeros(y.shape[1]*X.shape[1])
        true_idx = np.ravel_multi_index([m_r_idx, k_r_idx], (X.shape[1],y.shape[1]))
        true[true_idx] = 1
        ground_trueth.append(true)
        m = model(X, y, n_total, Z=Z)

        if inference_method.lower() == "map":
            result = m.find_MAP().summary(varnames=["beta"])
            pred = list(map(int, result["Pr(>|z|)"] < alpha))
            predictions.append(pred)
        else:
            result = m.sample().summary(varnames=["beta"])
            pred = list(map(int, result["Pr(>|z|)"] < alpha))
            predictions.append(pred)
    print(ground_trueth)
    ground_trueth = np.array(ground_trueth)
    predictions = np.array(predictions)
    print(ground_trueth)
    print(predictions)
    return (metrics.f1_score(ground_trueth, predictions, average="micro"),
            metrics.f1_score(ground_trueth, predictions, average="macro"),
            metrics.f1_score(ground_trueth, predictions, average="weighted"))




if __name__ == "__main__":
    from model.model import CompositionDE
    from scipy.special import softmax
    #simulator = lambda : generate_simple_data(n=100, m=1, k=5, m_r=1, k_r=2)

    X,y,Z,n_total,_,_ = scRNA_realistic_data(n=10)
    m = CompositionDE(X, y, n_total, Z=Z)
    r = m.find_MAP().summary(varnames=["alpha"])
    t = np.exp(r["mean"])
    print(t/t.sum())

    #print(run_simulator(CompositionDE, simulator, itr=2))



