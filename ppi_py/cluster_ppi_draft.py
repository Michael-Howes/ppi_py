import numpy as np
import pandas as pd
from ppi_py import cluster_ppi
# import statsmodels.stats.sandwich_covariance as sw
from utils.statistics_utils import cov_cluster
# from ppi_py.utils import cov_cluster

n = 10
N = 20
d = 2
grads = np.random.randn(n,d)
grads_hat = np.random.randn(n,d)
grads_hat_unlabeled = np.random.randn(N,d)
hessian = np.eye(d)
groups = np.arange(n)
groups_unlabeled = np.arange(n,n+N)

sigma = cluster_ppi._get_grad_covariance_matrix(
    grads, grads_hat, grads_hat_unlabeled, groups, groups_unlabeled
)

print(sigma)


def generate_cluster(
    theta,
    bias,
    ave_cluster_size,
    prob_labeled,
    rho,
    ppi_correlation,
    seed,
):
    rng = np.random.default_rng(seed)
    cluster_size = rng.poisson(ave_cluster_size) + 1

    sigma = (1 - rho) * np.eye(cluster_size) + rho * np.ones(
        (cluster_size, cluster_size)
    )
    mean = np.full(cluster_size, theta)

    Y = rng.multivariate_normal(
        mean=mean, cov=sigma, size=1
    )
    noise_sd = np.sqrt(1 - ppi_correlation**2)
    Yhat = bias + (
        ppi_correlation * Y
        + rng.normal(loc=0.0, scale=noise_sd, size=Y.shape)
    )

    labels = rng.uniform(size = cluster_size) < prob_labeled
    Y[0,~labels] = np.nan
    data = {
        "Y" : Y[0,labels],
        "Yhat" : Yhat[0,labels],
        "Yhat_unlabeled" : Yhat[0,~labels]
    }
    return data
    
def generate_data(
        theta,
        bias,
        ave_cluster_size,
        a_labeled,
        b_labeled,
        rho,
        ppi_correlation,
        num_clusters,
        seed=0
):
    Ys = []
    Yhats = []
    groups = []
    labels = []
    rng = np.random.default_rng(seed)
    for i in range(num_clusters):
        prob_labeled = rng.beta(a_labeled, b_labeled)
        data = generate_cluster(
            theta,
            bias,
            ave_cluster_size,
            prob_labeled,
            rho,
            ppi_correlation,
            seed = seed + i
        )
        Ys.append(data['Y'])
        Yhats.append(data['Yhat'])
        groups.append(np.repeat(i, len(data['Y'])))
        labels.append(data["labeled"])

    df = pd.DataFrame({
        "Y" : np.concatenate(Ys),
        "Yhat" : np.concatenate(Yhats),
        "group" : np.concatenate(groups),
        "label" : np.concatenate(labels)
    })
    return df


def ppi_mean_se3(
        Y,
        Yhat,
        Yhat_unlabeled,
        group,
        group_unlabeled,
        lam = 1.0
):
    n = len(Y)
    N = len(Yhat_unlabeled)

   

    combined_data = np.vstack(
        [np.concatenate([(Y - Y.mean()) / n, np.zeros(N)]),
        np.concatenate([(Yhat - Yhat.mean()) / n, np.zeros(N)]),
        np.concatenate([np.zeros(n), (Yhat_unlabeled - Yhat_unlabeled.mean())/N])]
    ).T
    combined_group = np.concatenate([group, group_unlabeled])


    cov = cov_cluster(combined_data, combined_group)

    theta_hat = Y.mean() - lam * Yhat.mean() + lam * Yhat_unlabeled.mean()
    var_rectifier = cov[0,0] - 2 * lam *cov[0,1] + lam**2 * cov[1,1]
    var_imputed = lam**2*cov[2,2]
    cov_rectifier_imputed = lam*cov[0,2] - lam**2 * cov[1,2]
    var = var_rectifier + var_imputed + 2 * cov_rectifier_imputed


    return theta_hat, (var)**0.5

    

theta = 1
bias = 10
ave_cluster_size = 2
a_labeled = 0.1
b_labeled = 0.1
rho = 0.4
ppi_correlation = 0.8
num_groups = 20
seed0=0
df = generate_data(
        theta,
        bias,
        ave_cluster_size,
        a_labeled,
        b_labeled,
        rho,
        ppi_correlation,
        num_groups,
        seed=seed0
    )
Y = df.loc[df['label'], "Y"].to_numpy()
Yhat = df.loc[df['label'], "Yhat"].to_numpy()
Yhat_unlabeled = df.loc[~df['label'], "Yhat"].to_numpy()
group = df.loc[df['label'], "group"].to_numpy()
group_unlabeled = df.loc[~df['label'], "group"].to_numpy()
lam = 0.5

print(ppi_mean_se3(Y, Yhat, Yhat_unlabeled, group, group_unlabeled, lam))

estimators = []
ses = []
estimators2 = []
ses2 = []
reps = 10
seed0 = reps*101
for i in range(reps):
    seed = (seed0+i)*num_groups
    df = generate_data(
        theta,
        bias,
        ave_cluster_size,
        a_labeled,
        b_labeled,
        rho,
        ppi_correlation,
        num_groups,
        seed=seed
    )

    Y = df.loc[df['label'], "Y"].to_numpy()
    Yhat = df.loc[df['label'], "Yhat"].to_numpy()
    Yhat_unlabeled = df.loc[~df['label'], "Yhat"].to_numpy()
    group = df.loc[df['label'], "group"].to_numpy()
    group_unlabeled = df.loc[~df['label'], "group"].to_numpy()

    theta_hat2, theta_se2 = ppi_mean_se3(Y, Yhat, Yhat_unlabeled, group, group_unlabeled)
    estimators2.append(theta_hat2)
    ses2.append(theta_se2)

estimators = np.array(estimators)
ses = np.array(ses)

estimators2 = np.array(estimators2)
ses2 = np.array(ses2)


print(f"Mean of theta_hat: {estimators.mean()}")
print(f"STD of theta_hat: {estimators.std()}")

print(f"Mean of  ses: {ses.mean()}")


print(f"\nMean of theta_hat2: {estimators2.mean()}")
print(f"STD of theta_hat2: {estimators2.std()}")

print(f"Mean of  ses2: {ses2.mean()}")

print(estimators2)
print(ses2)