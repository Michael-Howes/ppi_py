import numpy as np
from ppi_py.cluster_ppi import (
    ppi_ols_pointestimate_cluster,
    ppi_ols_ci_cluster,
)
from ppi_py.ppi import (
    ppi_ols_pointestimate,
    ppi_ols_ci,
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
    d = len(theta)  # Get dimension from theta
    
    if covariance_type == "equi":
        sigma = (1 - rho) * np.eye(cluster_size) + rho * np.ones(
            (cluster_size, cluster_size)
        )
    elif covariance_type == "AR":
        sigma = rho ** np.abs(
            np.arange(cluster_size)[:, None] - np.arange(cluster_size)[None, :]
        )
    else:
        raise ValueError(
            "Invalid value for covariance_type. Must be either 'equi' or 'AR'."
        )

    # Generate X as cluster_size x d array (covariates)
    mean = np.zeros(cluster_size)
    X = rng.multivariate_normal(mean=mean, cov=sigma, size=d).T
    
    # Generate error terms with cluster correlation structure
    
    e = rng.multivariate_normal(mean=mean, cov=sigma, size=1)[0]
    
    # Generate Y = X @ theta + e
    Y = X @ theta + e
    
    # Generate Yhat with correlation structure
    noise_sd = np.sqrt(1 - ppi_correlation**2)
    ehat = ppi_correlation * e + rng.normal(loc=0.0, scale=noise_sd, size=e.shape)
    
    # Yhat = X @ (theta + bias) + ehat
    Yhat = X @ (theta + bias) + ehat
    
    prob_labeled = rng.beta(a_labeled, b_labeled)
    labels = rng.uniform(size=cluster_size) < prob_labeled
    
    data = {
        "Y": Y[labels],
        "X": X[labels, :],
        "Yhat": Yhat[labels],
        "X_unlabeled": X[~labels, :],
        "Yhat_unlabeled": Yhat[~labels],
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
    Xs = []
    Yhats = []
    Xs_unlabeled = []
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
        Xs.append(data["X"])
        Yhats.append(data["Yhat"])
        Xs_unlabeled.append(data["X_unlabeled"])
        Yhats_unlabeled.append(data["Yhat_unlabeled"])
        groups.append(np.repeat(i, len(data["Y"])))
        groups_unlabeled.append(np.repeat(i, len(data["Yhat_unlabeled"])))

    combined_data = {
        "Y": np.concatenate(Ys),
        "X": np.concatenate(Xs, axis=0),
        "Yhat": np.concatenate(Yhats),
        "X_unlabeled": np.concatenate(Xs_unlabeled, axis=0),
        "Yhat_unlabeled": np.concatenate(Yhats_unlabeled),
        "group": np.concatenate(groups),
        "group_unlabeled": np.concatenate(groups_unlabeled),
    }
    return combined_data

def test_ppi_ols_cluster():
    seed = 0
    epsilon = 0.05

    d = 3
    theta = np.ones(d)
    bias = np.ones(d)
    ave_cluster_size = 5
    a_labeled = 0.1
    b_labeled = 0.1
    rho = 0.5
    ppi_correlation = 0.8
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

    theta_hat = ppi_ols_pointestimate_cluster(
        X=data["X"],
        Y=data["Y"],
        Yhat=data["Yhat"],
        X_unlabeled = data["X_unlabeled"],
        Yhat_unlabeled=data["Yhat_unlabeled"],
        group=data["group"],
        group_unlabeled=data["group_unlabeled"],
    )
    print(theta_hat)
    assert np.abs(theta - theta_hat).max() <= epsilon

def test_ppi_ols_cluster_groups_none():
    seed = 0
    rng = np.random.default_rng(seed)
    epsilon = 0.001

    d = 3
    n = 100
    N = 1000
    theta = np.ones(d)
    bias = 2*np.ones(d)
    ppi_correlation = 0.5
    
    X = rng.normal(size = (n, d))
    X_unlabeled = rng.normal(size = (N,d))
    
    e = rng.normal(size = n)
    ehat = ppi_correlation * e + np.sqrt(1-ppi_correlation**2) * rng.normal(size = n)
    ehat_unlabeled = rng.normal(size = N)

    Y = X @ theta + e
    Yhat = X @ (theta + bias) + ehat
    Yhat_unlabeled = X_unlabeled @ (theta + bias) + ehat_unlabeled

    theta_ppi = ppi_ols_pointestimate(
        X, Y, Yhat, X_unlabeled, Yhat_unlabeled
    )

    theta_cluster = ppi_ols_pointestimate_cluster(
        X, Y, Yhat, X_unlabeled, Yhat_unlabeled
    )
    print(theta_ppi)
    print(theta_cluster)
    assert np.abs(theta_ppi - theta_cluster).max() <= epsilon

def test_ppi_ols_ci_groups_none():
    seed = 0
    rng = np.random.default_rng(seed)
    epsilon = 0.01

    d = 3
    n = 100
    N = 1000
    theta = np.ones(d)
    bias = 2*np.ones(d)
    ppi_correlation = 0.5
    
    X = rng.normal(size = (n, d))
    X_unlabeled = rng.normal(size = (N,d))
    
    e = rng.normal(size = n)
    ehat = ppi_correlation * e + np.sqrt(1-ppi_correlation**2) * rng.normal(size = n)
    ehat_unlabeled = rng.normal(size = N)

    Y = X @ theta + e
    Yhat = X @ (theta + bias) + ehat
    Yhat_unlabeled = X_unlabeled @ (theta + bias) + ehat_unlabeled

    ci_ppi = ppi_ols_ci(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
    )

    ci_cluster = ppi_ols_ci_cluster(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
    )
    print(ci_ppi)
    print(ci_cluster)
    assert np.abs(ci_ppi[0] - ci_cluster[0]).max() <= epsilon
    assert np.abs(ci_ppi[1] - ci_cluster[1]).max() <= epsilon

def test_ppi_ols_cluster_coverage():
    seed = 0
    epsilon_cluster = 0.01
    error_ppi = 0.1
    alphas = np.array([0.05, 0.1, 0.2])
    reps = 1000

    d = 3
    theta = np.ones(d)
    bias = 2*np.ones(d)
    ave_cluster_size = 10
    a_labeled = 1
    b_labeled = 1
    rho = 0.5
    ppi_correlation = 0.5
    num_clusters = 1000
    covariance_type = "equi"

    includeds_cluster = np.zeros((d, len(alphas)))
    includeds_ppi = np.zeros((d, len(alphas)))

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
        ci_cluster = ppi_ols_ci_cluster(
            X=data["X"],
            Y=data["Y"],
            Yhat=data["Yhat"],
            X_unlabeled=data["X_unlabeled"],
            Yhat_unlabeled=data["Yhat_unlabeled"],
            group=data["group"],
            group_unlabeled=data["group_unlabeled"],
            alpha=alphas,
        )

    
        includeds_cluster += (ci_cluster[0] <= theta) * (ci_cluster[1] >= theta)

        ci_ppi = ppi_ols_ci(
            X=data["X"],
            Y=data["Y"],
            Yhat=data["Yhat"],
            X_unlabeled=data["X_unlabeled"],
            Yhat_unlabeled=data["Yhat_unlabeled"],
            alpha=alphas,
        )

        includeds_ppi += (ci_ppi[0] <= theta) * (ci_ppi[1] >= theta)
        
    # Cluster ci has correct coverage.
    print(includeds_cluster / reps)
    print(includeds_ppi / reps)
    assert (np.abs(includeds_cluster / reps - (1 - alphas)) <= epsilon_cluster).all()

    # Regular ppi does not have correct coverage for clustered data.
    assert (includeds_ppi / reps < (1 - alphas) - error_ppi).all()
