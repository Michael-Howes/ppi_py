from . import ppi
import numpy as np
from numba import njit
from scipy.stats import norm, binom
from scipy.optimize import brentq, minimize
from scipy.linalg import sqrtm, inv
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.weightstats import (
    _zconfint_generic,
    _zstat_generic,
    _zstat_generic2,
)
from sklearn.linear_model import LogisticRegression, PoissonRegressor
import warnings

warnings.simplefilter("ignore")
from .utils import (
    construct_weight_vector,
    safe_expit,
    safe_log1pexp,
    compute_cdf,
    compute_cdf_diff,
    dataframe_decorator,
    linfty_dkw,
    linfty_binom,
    form_discrete_distribution,
    reshape_to_2d,
    bootstrap,
    cov_cluster,
)

"""
    MEAN ESTIMATION

"""
def _ppi_cov(
    grads_cov,
    inv_hessian,
    lam,
):
    d = inv_hessian.shape[0]
    meat = grads_cov[:d,:d] - lam * (grads_cov[:d,d:] + grads_cov[d:,:d]) + lam**2 * grads_cov[d:, d:]
    return inv_hessian @ meat @ inv_hessian

# def _ppi_cov(
#     grads_cov,
#     inv_hessian,
#     lam,
#     n,
#     N,
# ):
#     d = inv_hessian.shape[0]
#     var_rectifier = (
#         grads_cov[:d, :d]
#         + lam**2 * grads_cov[d : (2 * d), d : (2 * d)]
#         - lam * (grads_cov[:d, d : (2 * d)] + grads_cov[d : (2 * d), :d])
#     )
#     var_imputed = lam**2 * grads_cov[2 * d :, 2 * d :]
#     cov_rectifier_imputed = lam * (
#         grads_cov[:d, 2 * d :] + grads_cov[2 * d :, :d]
#     ) - lam**2 * (grads_cov[d : (2 * d), 2 * d :] + grads_cov[2 * d :, d : (2 * d)])
#     meat = var_rectifier / n**2 + var_imputed / N**2 + cov_rectifier_imputed / n / N
#     return inv_hessian @ meat @ inv_hessian


def ppi_mean_pointestimate_cluster(
    Y,
    Yhat,
    Yhat_unlabeled,
    lam=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    group=None,
    group_unlabeled=None,
    lam_optim_mode="overall",
    cr2=False,
):
    """Computes the prediction-powered point estimate of the d-dimensional mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the dimension of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set. Defaults to all ones vector.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set. Defaults to all ones vector.
        group (ndarray, optional): Cluster indicators for the labeled data set. Defaults to separate clusters
        for each data point if None.
        group_unlabeled (ndarray, optional): Cluster indicators for the labeled data set. Defaults to
        separate clusters for each data point if None.

    Returns:
        float or ndarray: Prediction-powered point estimate of the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled)
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = Yhat.shape[1]

    w = construct_weight_vector(n, w, vectorized=True)
    w_unlabeled = construct_weight_vector(N, w_unlabeled, vectorized=True)

    if lam is None:
        ppi_pointest = (w_unlabeled * Yhat_unlabeled).mean(0) + (w * (Y - Yhat)).mean(0)
        inv_hessian = np.eye(d)
        residuals = w*(Y - (w*Y).mean(0))
        residuals_hat = w*(Yhat - (w*Yhat).mean(0))
        residuals_hat_unlabeled = (
            w_unlabeled*(Yhat_unlabeled - (w_unlabeled*Yhat_unlabeled).mean(0))
        )
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
            cr2 = cr2
        )
        lam = _calc_lam_opt(
            grads_cov,
            inv_hessian,
            coord=coord,
            optim_mode=lam_optim_mode,
        )
        return ppi_mean_pointestimate_cluster(
            Y,
            Yhat,
            Yhat_unlabeled,
            lam=lam,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
            group=group,
            group_unlabeled = group_unlabeled,
            cr2=cr2
        )
    else:
        return (w_unlabeled * lam * Yhat_unlabeled).mean(axis=0) + (
            w * (Y - lam * Yhat)
        ).mean(axis=0).squeeze()


def ppi_mean_ci_cluster(
    Y,
    Yhat,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lam=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    group=None,
    group_unlabeled=None,
    lam_optim_mode="overall",
    cr2=False,
):
    """Computes the prediction-powered confidence interval for a d-dimensional mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = Y.shape[1] if len(Y.shape) > 1 else 1

    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled)

    w = construct_weight_vector(n, w, vectorized=True)
    w_unlabeled = construct_weight_vector(N, w_unlabeled, vectorized=True)

    if lam is None:
        ppi_pointest = ppi_mean_pointestimate_cluster(
            Y,
            Yhat,
            Yhat_unlabeled,
            lam=1,
            w=w,
            w_unlabeled=w_unlabeled,
            group=group,
            group_unlabeled=group_unlabeled,
        )
        residuals = w*(Y - (w*Y).mean(0))
        residuals_hat = w*(Yhat - (w*Yhat).mean(0))
        residuals_hat_unlabeled = (
            w_unlabeled*(Yhat_unlabeled - (w_unlabeled*Yhat_unlabeled).mean(0))
        )
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
            cr2 = cr2
        )
        inv_hessian = np.eye(d)
        lam = _calc_lam_opt(
            grads_cov,
            inv_hessian,
            coord=coord,
            optim_mode=lam_optim_mode,
        )
        return ppi_mean_ci_cluster(
            Y,
            Yhat,
            Yhat_unlabeled,
            alpha=alpha,
            lam=lam,
            alternative=alternative,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
            group=group,
            group_unlabeled=group_unlabeled,
            lam_optim_mode=lam_optim_mode,
            cr2=cr2
        )

    ppi_pointest = ppi_mean_pointestimate_cluster(
        Y,
        Yhat,
        Yhat_unlabeled,
        lam=lam,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
        group=group,
        group_unlabeled=group_unlabeled,
        lam_optim_mode=lam_optim_mode,
        cr2=cr2
    )
    residuals = w*(Y - (w*Y).mean(0))
    residuals_hat = w*(Yhat - (w*Yhat).mean(0))
    residuals_hat_unlabeled = (
        w_unlabeled*(Yhat_unlabeled - (w_unlabeled*Yhat_unlabeled).mean(0))
    )
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
        cr2 = cr2
    )
    inv_hessian = np.eye(d)
    lam = _calc_lam_opt(
        grads_cov,
        inv_hessian,
        coord=coord,
        optim_mode=lam_optim_mode,
    )

    se = np.sqrt(np.diag(_ppi_cov(grads_cov, inv_hessian, lam))) / n

    return _zconfint_generic(
        ppi_pointest,
        se,
        alpha,
        alternative,
    )


"""
    OLS
"""


def ppi_ols_pointestimate_cluster(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    lam=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    group=None,
    group_unlabeled=None,
    cr2=False,
):
    """Computes the prediction-powered point estimate of the OLS coefficients.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        ndarray: Prediction-powered point estimate of the OLS coefficients.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / np.sum(w) * n
    w_unlabeled = (
        np.ones(N) if w_unlabeled is None else w_unlabeled / np.sum(w_unlabeled) * N
    )
    use_unlabeled = lam != 0

    imputed_theta = (
        ppi._wls(X_unlabeled, Yhat_unlabeled, w=w_unlabeled)
        if lam is None
        else ppi._wls(X_unlabeled, lam * Yhat_unlabeled, w=w_unlabeled)
    )
    rectifier = (
        ppi._wls(X, Y - Yhat, w=w) if lam is None else ppi._wls(X, Y - lam * Yhat, w=w)
    )
    ppi_pointest = imputed_theta + rectifier

    if lam is None:
        # resid = lambda M, v: v - M @ np.linalg.lstsq(M, v, rcond=None)[0]
        # residuals = resid(X, Y)
        # residuals_hat = resid(X, Yhat)
        # residuals_hat_unlabeled = resid(X_unlabeled, Yhat_unlabeled)

        residuals = Y - X@ppi_pointest
        residuals_hat = Yhat - X@ppi_pointest
        residuals_hat_unlabeled = Yhat_unlabeled - X_unlabeled@ppi_pointest
        _, _, _, inv_hessian = ppi._ols_get_stats(
                ppi_pointest,
                X.astype(float),
                Y,
                Yhat,
                X_unlabeled.astype(float),
                Yhat_unlabeled,
                w=w,
                w_unlabeled=w_unlabeled,
                use_unlabeled=use_unlabeled,
            )
        grads_cov = _get_grad_covariance_matrix(
            residuals = residuals,
            residuals_hat = residuals_hat,
            residuals_hat_unlabeled = residuals_hat_unlabeled,
            X = X,
            X_unlabeled = X_unlabeled,
            group=group,
            group_unlabeled=group_unlabeled,
            cr2=cr2
        )
        lam = _calc_lam_opt(
            grads_cov,
            inv_hessian,
            coord=coord,
        )
        return ppi_ols_pointestimate_cluster(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            lam=lam,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
            group=group,
            group_unlabeled=group_unlabeled,
            cr2=cr2
        )
    else:
        return ppi_pointest


def ppi_ols_ci_cluster(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lam=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    group=None,
    group_unlabeled=None,
    cr2=False,
):
    """Computes the prediction-powered confidence interval for the OLS coefficients using the PPI++ algorithm from `[ADZ23] <https://arxiv.org/abs/2311.01453>`__.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lam (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lam`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the OLS coefficients.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N) if w_unlabeled is None else w_unlabeled / w_unlabeled.sum() * N
    )
    use_unlabeled = lam != 0  # If lam is 0, revert to classical estimation.

    ppi_pointest = ppi_ols_pointestimate_cluster(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lam=lam,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
        group=group,
        group_unlabeled=group_unlabeled,
        cr2=cr2,
    )

    # resid = lambda M, v: v - M @ np.linalg.lstsq(M, v, rcond=None)[0]
    # residuals = resid(X, Y)
    # residuals_hat = resid(X, Yhat)
    # residuals_hat_unlabeled = resid(X_unlabeled, Yhat_unlabeled)
    residuals = Y - X@ppi_pointest
    residuals_hat = Yhat - X@ppi_pointest
    residuals_hat_unlabeled = Yhat_unlabeled - X_unlabeled@ppi_pointest
    _, _, _, inv_hessian = ppi._ols_get_stats(
            ppi_pointest,
            X.astype(float),
            Y,
            Yhat,
            X_unlabeled.astype(float),
            Yhat_unlabeled,
            w=w,
            w_unlabeled=w_unlabeled,
            use_unlabeled=use_unlabeled,
        )
    grads_cov = _get_grad_covariance_matrix(
        residuals = residuals,
        residuals_hat = residuals_hat,
        residuals_hat_unlabeled = residuals_hat_unlabeled,
        X = X,
        X_unlabeled = X_unlabeled,
        group=group,
        group_unlabeled=group_unlabeled,
        cr2=cr2
    )
    if lam is None:
        lam = _calc_lam_opt(
            grads_cov,
            inv_hessian,
            coord=coord,
            clip=True,
        )
        return ppi_ols_ci_cluster(
            X=X,
            Y=Y,
            Yhat=Yhat,
            X_unlabeled=X_unlabeled,
            Yhat_unlabeled=Yhat_unlabeled,
            alpha=alpha,
            alternative=alternative,
            lam=lam,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
            group=group,
            group_unlabeled=group_unlabeled,
            cr2=cr2
        )

    se = np.sqrt(np.diag(_ppi_cov(grads_cov, inv_hessian, lam))) / n

    return _zconfint_generic(
        ppi_pointest,
        se,
        alpha,
        alternative,
    )


# def _get_grad_covariance_matrix(
#     grads,
#     grads_hat,
#     grads_hat_unlabeled,
#     group=None,
#     group_unlabeled=None,
# ):
#     grads = reshape_to_2d(grads)
#     grads_hat = reshape_to_2d(grads_hat)
#     grads_hat_unlabeled = reshape_to_2d(grads_hat_unlabeled)
#     grads_cent = grads - grads.mean(axis=0)
#     grads_hat_cent = grads_hat - grads_hat.mean(axis=0)
#     grads_hat_unlabeled_cent = grads_hat_unlabeled - grads_hat_unlabeled.mean(axis=0)

#     combined_grads = np.hstack(
#         [
#             np.vstack([grads_cent, np.zeros_like(grads_hat_unlabeled)]),
#             np.vstack([grads_hat_cent, np.zeros_like(grads_hat_unlabeled)]),
#             np.vstack([np.zeros_like(grads), grads_hat_unlabeled_cent]),
#         ]
#     )
#     if (group is not None) and (group_unlabeled is not None):
#         combined_groups = np.concatenate([group, group_unlabeled])
#         covariance = cov_cluster(combined_grads, combined_groups)
#     else:
#         covariance = np.dot(combined_grads.T, combined_grads)
#     return covariance

# def _get_grad_covariance_matrix(
#     grads,
#     grads_hat,
#     grads_hat_unlabeled,
#     group=None,
#     group_unlabeled=None,
# ):
#     n = len(grads)
#     N = len(grads_hat_unlabeled)
#     r = n/N
    
#     grads = reshape_to_2d(grads)
#     grads_hat = reshape_to_2d(grads_hat)
#     grads_hat_unlabeled = reshape_to_2d(grads_hat_unlabeled)
#     # grads_cent = grads - grads.mean(axis=0)
#     # grads_hat_cent = grads_hat - grads_hat.mean(axis=0)
#     # grads_hat_unlabeled_cent = grads_hat_unlabeled - grads_hat_unlabeled.mean(axis=0)

#     combined_grads = np.hstack(
#         [
#             np.vstack([grads, np.zeros_like(grads_hat_unlabeled)]),
#             np.vstack([grads_hat, -r * grads_hat_unlabeled]),
#         ]
#     )
#     if (group is not None) and (group_unlabeled is not None):
#         combined_groups = np.concatenate([group, group_unlabeled])
#         covariance = cov_cluster(combined_grads, combined_groups)
#     else:
#         covariance = np.cov(combined_grads)
#     return covariance



def _cr2_adjustment(X, residual, group):
    """Returns M_g^{-1/2} @ u_g for every cluster g, in the original row order,
    with M_g = I - X_g (X'X)^{-1} X_g'."""
    B_inv = np.linalg.inv(X.T @ X)
    out = np.empty_like(residual)
    for g in np.unique(group):
        idx = group == g
        Xg = X[idx]
        M_g = np.eye(Xg.shape[0]) - Xg @ B_inv @ Xg.T
        sqrt_inv = np.real_if_close(
            inv(sqrtm(M_g))
        )
        out[idx] = sqrt_inv @ residual[idx]
    return out

def _get_grad_covariance_matrix(
    residuals,
    residuals_hat,
    residuals_hat_unlabeled,
    X,
    X_unlabeled,
    group=None,
    group_unlabeled=None,
    cr2 = False
):
    n = len(residuals)
    N = len(residuals_hat_unlabeled)
    r = n/N
    
    residuals = reshape_to_2d(residuals)
    residuals_hat = reshape_to_2d(residuals_hat)
    residuals_hat_unlabeled = reshape_to_2d(residuals_hat_unlabeled)
    X = reshape_to_2d(X)
    X_unlabeled = reshape_to_2d(X_unlabeled)

    if (group is None) or (group_unlabeled is None):
        group, group_unlabeled = np.arange(n), np.arange(n, n + N)  # HC2 analogue
    # joint factorization so labeled/unlabeled ids share one id space
    _, codes = np.unique(np.concatenate([group, group_unlabeled]), return_inverse=True)
    group, group_unlabeled = codes[:n], codes[n:]

    if cr2:
        residuals = _cr2_adjustment(X, residuals, group)
        residuals_hat = _cr2_adjustment(X, residuals_hat, group)
        residuals_hat_unlabeled = _cr2_adjustment(
            X_unlabeled, residuals_hat_unlabeled, group_unlabeled
        )
        grads = X * residuals
        grads_hat = X * residuals_hat
        grads_hat_unlabeled = X_unlabeled * residuals_hat_unlabeled
    else:   
        grads = X * residuals
        grads_hat = X * residuals_hat
        grads_hat_unlabeled = X_unlabeled * residuals_hat_unlabeled

        mean_grad_hat = (grads_hat.mean(axis=0)*r + grads_hat_unlabeled.mean(axis=0))/(r+1)
        grads_hat = grads_hat - mean_grad_hat
        grads_hat_unlabeled = grads_hat_unlabeled - mean_grad_hat
    

    combined_grads = np.hstack(
        [
            np.vstack([grads, np.zeros_like(grads_hat_unlabeled)]),
            np.vstack([grads_hat, -r * grads_hat_unlabeled]),
        ]
    )
    combined_groups = np.concatenate([group, group_unlabeled])
    return cov_cluster(combined_grads, combined_groups)

# def _get_grad_covariance_matrix(
#     grads,
#     grads_hat,
#     grads_hat_unlabeled,
#     group=None,
#     group_unlabeled=None,
# ):
#     n = len(grads)
#     N = len(grads_hat_unlabeled)
#     r = n/N
    
#     grads = reshape_to_2d(grads)
#     grads_hat = reshape_to_2d(grads_hat)
#     grads_hat_unlabeled = reshape_to_2d(grads_hat_unlabeled)
#     mean_grad_hat = (grads_hat.mean(axis=0)*r + grads_hat_unlabeled.mean(axis=0))/(r+1)
#     grads_hat_cent = grads_hat - mean_grad_hat
#     grads_hat_unlabeled_cent = grads_hat_unlabeled - mean_grad_hat

#     combined_grads = np.hstack(
#         [
#             np.vstack([grads, np.zeros_like(grads_hat_unlabeled)]),
#             np.vstack([grads_hat_cent, -r * grads_hat_unlabeled_cent]),
#         ]
#     )
#     if (group is None) or (group_unlabeled is None):
#         group, group_unlabeled = np.arange(n), np.arange(n, n + N) 
#     if (group is not None) and (group_unlabeled is not None):
#         combined_groups = np.concatenate([group, group_unlabeled])
#         covariance = cov_cluster(combined_grads, combined_groups)
#     else:
#         covariance = np.dot(combined_grads.T, combined_grads)
#     return covariance



def _calc_lam_opt(
    grads_cov,
    inv_hessian,
    coord=None,
    clip=False,
    optim_mode="overall",
):
    d = inv_hessian.shape[0]
    vhat = inv_hessian if coord is None else inv_hessian[coord, :]
    numerator_ = grads_cov[:d, d:] + grads_cov[d:,:d] 
    denominator_ = 2 * grads_cov[d:, d:]
    if optim_mode == "overall":
        num = (
            np.trace(vhat @ numerator_ @ vhat)
            if coord is None
            else vhat @ numerator_ @ vhat
        )
        denom = (
            np.trace(vhat @ denominator_ @ vhat)
            if coord is None
            else vhat @ denominator_ @ vhat
        )
        lam = num / denom
        lam = lam.item()
    elif optim_mode == "element":
        num = np.diag(vhat @ numerator_ @ vhat)
        denom = np.diag(vhat @ denominator_ @ vhat)
        lam = num / denom
    else:
        raise ValueError(
            "Invalid value for optim_mode. Must be either 'overall' or 'element'."
        )
    if clip:
        lam = np.clip(lam, 0, 1)
    return lam

# def _calc_lam_opt(
#     grads_cov,
#     inv_hessian,
#     coord=None,
#     clip=False,
#     optim_mode="overall",
# ):
#     d = inv_hessian.shape[0]
#     vhat = inv_hessian if coord is None else inv_hessian[coord, :]
#     numerator_ = (grads_cov[:d, d : (2 * d)] + grads_cov[d : (2 * d), :d]) / n**2 - (
#         grads_cov[:d, 2 * d :] + grads_cov[2 * d :, :d]
#     ) 
#     denominator_ = 2 * (
#         grads_cov[d : (2 * d), d : (2 * d)] / n**2
#         + grads_cov[2 * d :, 2 * d :] / N**2
#         - (grads_cov[d : (2 * d), 2 * d :] + grads_cov[2 * d :, d : (2 * d)]) / n / N
#     )
#     if optim_mode == "overall":
#         num = (
#             np.trace(vhat @ numerator_ @ vhat)
#             if coord is None
#             else vhat @ numerator_ @ vhat
#         )
#         denom = (
#             np.trace(vhat @ denominator_ @ vhat)
#             if coord is None
#             else vhat @ denominator_ @ vhat
#         )
#         lam = num / denom
#         lam = lam.item()
#     elif optim_mode == "element":
#         num = np.diag(vhat @ numerator_ @ vhat)
#         denom = np.diag(vhat @ denominator_ @ vhat)
#         lam = num / denom
#     else:
#         raise ValueError(
#             "Invalid value for optim_mode. Must be either 'overall' or 'element'."
#         )
#     if clip:
#         lam = np.clip(lam, 0, 1)
#     return lam


# def _ppi_cov(
#     grads_cov,
#     inv_hessian,
#     lam,
#     n,
#     N,
# ):
#     d = inv_hessian.shape[0]
#     var_rectifier = (
#         grads_cov[:d, :d]
#         + lam**2 * grads_cov[d : (2 * d), d : (2 * d)]
#         - lam * (grads_cov[:d, d : (2 * d)] + grads_cov[d : (2 * d), :d])
#     )
#     var_imputed = lam**2 * grads_cov[2 * d :, 2 * d :]
#     cov_rectifier_imputed = lam * (
#         grads_cov[:d, 2 * d :] + grads_cov[2 * d :, :d]
#     ) - lam**2 * (grads_cov[d : (2 * d), 2 * d :] + grads_cov[2 * d :, d : (2 * d)])
#     meat = var_rectifier / n**2 + var_imputed / N**2 + cov_rectifier_imputed / n / N
#     return inv_hessian @ meat @ inv_hessian
