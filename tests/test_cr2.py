import numpy as np
from ppi_py.cluster_ppi import (
    ppi_mean_pointestimate_cluster,
    ppi_mean_ci_cluster,
    _calc_lam_opt,
    _get_grad_covariance_matrix,
)
from ppi_py.ppi import (
    ppi_mean_pointestimate,
    ppi_mean_ci,
    _calc_lam_glm,
)


def generate_cluster(
    theta,
    bias,
    ave_cluster_size,
    a_labeled,
    b_labeled,
    rho,
    ppi_correlation,
    seed,
    covariance_type="equi",
):
    rng = np.random.default_rng(seed)
    cluster_size = rng.poisson(ave_cluster_size) + 1
    if covariance_type == "equi":
        sigma = (1 - rho) * np.eye(cluster_size) + rho * np.ones(
            (cluster_size, cluster_size)
        )
    elif covariance_type == "AR":
        sigma = rho ** np.abs(
            np.arange(cluster_size)[:, None] - np.arange(cluster_size)[None:,]
        )
    else:
        raise ValueError(
            "Invalid value for covariance_type. Must be either 'equi' or 'AR'."
        )

    mean = np.arange(1, cluster_size+1) * theta / ((cluster_size+1)/2)

    Y = rng.multivariate_normal(mean=mean, cov=sigma, size=1)
    noise_sd = np.sqrt(1 - ppi_correlation**2)
    Yhat = bias + (
        ppi_correlation * Y + rng.normal(loc=0.0, scale=noise_sd, size=Y.shape)
    )
    prob_labeled = rng.beta(a_labeled, b_labeled)

    labels = rng.uniform(size=cluster_size) < prob_labeled
    Y[0, ~labels] = np.nan
    data = {
        "Y": Y[0, labels],
        "Yhat": Yhat[0, labels],
        "Yhat_unlabeled": Yhat[0, ~labels],
    }
    return data


def generate_data_clustered(
    theta,
    bias,
    ave_cluster_size,
    a_labeled,
    b_labeled,
    rho,
    ppi_correlation,
    num_clusters,
    covariance_type="equi",
    seed=0,
):
    Ys = []
    Yhats = []
    Yhats_unlabeled = []
    groups = []
    groups_unlabeled = []
    rng = np.random.default_rng(seed)
    for i in range(num_clusters):
        data = generate_cluster(
            theta,
            bias,
            ave_cluster_size,
            a_labeled,
            b_labeled,
            rho,
            ppi_correlation,
            seed=seed + i,
            covariance_type=covariance_type,
        )
        Ys.append(data["Y"])
        Yhats.append(data["Yhat"])
        Yhats_unlabeled.append(data["Yhat_unlabeled"])
        groups.append(np.repeat(i, len(data["Y"])))
        groups_unlabeled.append(np.repeat(i, len(data["Yhat_unlabeled"])))

    combined_data = {
        "Y": np.concatenate(Ys),
        "Yhat": np.concatenate(Yhats),
        "Yhat_unlabeled": np.concatenate(Yhats_unlabeled),
        "group": np.concatenate(groups),
        "group_unlabeled": np.concatenate(groups_unlabeled),
    }
    return combined_data



theta = 1
bias = 2
ave_cluster_size = 20
a_labeled = 0.01
b_labeled = 0.1
rho = 0.5
ppi_correlation = 0.5
num_clusters = 100

reps = 1000
ppis = np.zeros(reps)

for i in range(reps):
    data = generate_data_clustered(
        theta=theta,
        bias=bias,
        ave_cluster_size=ave_cluster_size,
        a_labeled=a_labeled,
        b_labeled=b_labeled,
        rho=rho,
        ppi_correlation=ppi_correlation,
        num_clusters=num_clusters,
        seed=i,
    )
    Y=data["Y"]
    Yhat=data["Yhat"]
    n = len(Y)
    Yhat_unlabeled=data["Yhat_unlabeled"]
    N = len(Yhat_unlabeled)
    group=data["group"]
    group_unlabeled=data["group_unlabeled"]
    ppis[i] = ppi_mean_pointestimate_cluster(
        Y, Yhat, Yhat_unlabeled, group = group, group_unlabeled = group_unlabeled
    )
seed = 3
data = generate_data_clustered(
    theta=theta,
    bias=bias,
    ave_cluster_size=ave_cluster_size,
    a_labeled=a_labeled,
    b_labeled=b_labeled,
    rho=rho,
    ppi_correlation=ppi_correlation,
    num_clusters=num_clusters,
    seed=seed,
)

Y=data["Y"]
Yhat=data["Yhat"]
n = len(Y)
Yhat_unlabeled=data["Yhat_unlabeled"]
N = len(Yhat_unlabeled)
group=data["group"]
group_unlabeled=data["group_unlabeled"]
ppi_pointest = (Yhat_unlabeled).mean(0) + ((Y - Yhat)).mean(0)
inv_hessian = np.eye(1)
residuals = (Y - Y.mean(0))
Yhat_mean = np.concatenate([Yhat, Yhat_unlabeled]).mean()
residuals_hat = (Yhat - Yhat.mean(0))
residuals_hat_unlabeled = (Yhat_unlabeled - Yhat_unlabeled.mean())
X = np.ones(n)
X_unlabeled = np.ones(N)
grads_cov = _get_grad_covariance_matrix(
    residuals = residuals,
    residuals_hat = residuals_hat,
    residuals_hat_unlabeled = residuals_hat_unlabeled,
    X=X,
    X_unlabeled=X_unlabeled,
    group=group,
    group_unlabeled=group_unlabeled,
    cr2 = False
)
grads_cov_cr2 = _get_grad_covariance_matrix(
    residuals = residuals,
    residuals_hat = residuals_hat,
    residuals_hat_unlabeled = residuals_hat_unlabeled,
    X=X,
    X_unlabeled=X_unlabeled,
    group=group,
    group_unlabeled=group_unlabeled,
    cr2 = True
)
print("No cr2:", grads_cov)
print("cr2:", grads_cov_cr2)
print("Rel:", np.sqrt(grads_cov_cr2/grads_cov))

ci = ppi_mean_ci_cluster(
    Y, Yhat, Yhat_unlabeled, alpha = 0.05, group = group, group_unlabeled = group_unlabeled
)
ci_cr2 = ppi_mean_ci_cluster(
    Y, Yhat, Yhat_unlabeled, alpha = 0.05, group = group, group_unlabeled = group_unlabeled, cr2 = True
)
print("No cr2:", (ci[1]-ci[0])/1.959964/2)
print("cr2:", (ci_cr2[1]-ci_cr2[0])/1.959964/2)
print("Emperical:", ppis.std())