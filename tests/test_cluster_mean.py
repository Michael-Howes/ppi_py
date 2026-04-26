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

    mean = np.full(cluster_size, theta)

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


def test_ppi_mean_pointestimate_cluster():
    seed = 0
    epsilon = 0.05

    theta = 1
    bias = 2
    ave_cluster_size = 5
    a_labeled = 0.1
    b_labeled = 0.1
    rho = 0.5
    ppi_correlation = 0.5
    num_clusters = 1000

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

    theta_hat = ppi_mean_pointestimate_cluster(
        Y=data["Y"],
        Yhat=data["Yhat"],
        Yhat_unlabeled=data["Yhat_unlabeled"],
        group=data["group"],
        group_unlabeled=data["group_unlabeled"],
    )

    assert np.abs(theta - theta_hat) < epsilon


def test_calc_lam_opt():
    seed = 1234
    rng = np.random.default_rng(seed)
    epsilon = 0.001

    n = 300
    N = 300
    Y = rng.normal(0, 1, n)
    Yhat = Y + rng.normal(2, 1, n)
    Yhat_unlabeled = rng.normal(2, 2**0.5, N)
    grads = Y - Y.mean()
    grads_hat = Yhat - Y.mean()
    grads_hat_unlabeled = Yhat_unlabeled - Y.mean()
    inv_hessian = np.eye(1)

    lam_ppi = _calc_lam_glm(
        grads,
        grads_hat,
        grads_hat_unlabeled,
        inv_hessian,
    )

    grads_cov = _get_grad_covariance_matrix(
        grads, grads_hat, grads_hat_unlabeled, group=None, group_unlabeled=None
    )
    print(grads_cov)
    lam_cluster = _calc_lam_opt(
        grads_cov,
        inv_hessian,
    )
    print(lam_ppi, lam_cluster)
    assert np.abs(lam_ppi - lam_cluster) <= epsilon


def test_ppi_mean_pointestimate_cluster_groups_none():
    seed = 0
    rng = np.random.default_rng(seed)
    epsilon = 0.01

    n = 200
    N = 2000
    Y = rng.normal(1, 1, n)
    Yhat = Y + rng.normal(2, 1, n)
    Yhat_unlabeled = rng.normal(3, 2**0.5, N)

    theta_ppi = ppi_mean_pointestimate(
        Y,
        Yhat,
        Yhat_unlabeled,
    )

    theta_ppi_cluster = ppi_mean_pointestimate_cluster(Y, Yhat, Yhat_unlabeled)
    print(theta_ppi, theta_ppi_cluster)
    assert np.abs(theta_ppi - theta_ppi_cluster) <= epsilon


def test_ppi_mean_ci_groups_none():
    seed = 0
    rng = np.random.default_rng(seed)
    epsilon = 0.01

    n = 200
    N = 2000
    Y = rng.normal(1, 1, n)
    Yhat = Y + rng.normal(2, 1, n)
    Yhat_unlabeled = rng.normal(3, 2**0.5, N)

    ci_ppi = ppi_mean_ci(
        Y,
        Yhat,
        Yhat_unlabeled,
    )

    ci_cluster = ppi_mean_ci_cluster(
        Y,
        Yhat,
        Yhat_unlabeled,
    )
    print("PPI:", ci_ppi)
    print("Cluster:", ci_cluster)
    assert np.abs(ci_ppi[0] - ci_cluster[0]) < epsilon
    assert np.abs(ci_ppi[1] - ci_cluster[1]) < epsilon


def test_ppi_mean_cluster_coverage():
    seed = 1234
    epsilon_cluster = 0.02
    epsilon_ppi = 0.1
    alphas = np.array([0.05, 0.1, 0.2])
    reps = 1000

    theta = 1
    bias = 2
    ave_cluster_size = 8
    a_labeled = 1
    b_labeled = 1
    rho = 0.5
    ppi_correlation = 0.5
    num_clusters = 400
    covariance_type = "AR"

    includeds_cluster = np.zeros_like(alphas, dtype = int)
    includeds_ppi = np.zeros_like(alphas, dtype = int)

    for i in range(reps):
        seed += num_clusters
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
            covariance_type=covariance_type,
        )
        ci_cluster = ppi_mean_ci_cluster(
            Y=data["Y"],
            Yhat=data["Yhat"],
            Yhat_unlabeled=data["Yhat_unlabeled"],
            group=data["group"],
            group_unlabeled=data["group_unlabeled"],
            alpha=alphas,
        )

        includeds_cluster += ( 
            (ci_cluster[0] <= theta)*  (ci_cluster[1] >= theta)
        )

        ci_ppi = ppi_mean_ci(
            Y=data["Y"],
            Yhat=data["Yhat"],
            Yhat_unlabeled=data["Yhat_unlabeled"],
            alpha=alphas,
        )

        includeds_ppi += ( 
            (ci_ppi[0] <= theta)*  (ci_ppi[1] >= theta)
        )

    # Cluster ci has correct coverage.
    print(includeds_cluster / reps)
    assert np.abs(includeds_cluster / reps - (1 - alphas)).max() <= epsilon_cluster

    # Regular ppi does not have correct coverage for clustered data.
    print(includeds_ppi / reps)
    assert (includeds_ppi / reps < (1 - alphas) - epsilon_ppi).all()

