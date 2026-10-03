import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jensenshannon

from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import jensenshannon
import numpy as np

def similarity_score(dev_kde_scores, dataset_2):
    """
    Calculate the similarity score between the given KDE scores and a dataset.

    Parameters:
    dev_kde_scores (array-like): The KDE scores of the development dataset.
    dataset_2 (array-like): The dataset to compare against.

    Returns:
    float: The Jensen-Shannon divergence between the two sets of KDE scores.
    """
    generated_kde = KernelDensity(kernel='gaussian', bandwidth="scott").fit(dataset_2)
    kde2_scores = generated_kde.score_samples(dataset_2)
    js_divergence = jensenshannon(np.exp(dev_kde_scores), np.exp(kde2_scores))
    return js_divergence

def kl_divergence(p_mean, p_cov, q_mean, q_cov):
    k = p_mean.shape[0]
    q_cov_inv = np.linalg.inv(q_cov)
    tr = np.trace(np.dot(q_cov_inv, p_cov))
    diff = q_mean - p_mean
    mahalanobis = np.dot(np.dot(diff.T, q_cov_inv), diff)
    det_ratio = np.log(np.linalg.det(q_cov) / np.linalg.det(p_cov))
    return 0.5 * (tr + mahalanobis - k + det_ratio)

def js_divergence(p_mean, p_cov, q_mean, q_cov):
    def kl_divergence(mean1, cov1, mean2, cov2):
        k = mean1.shape[0]
        cov2_inv = np.linalg.inv(cov2)
        tr = np.trace(np.dot(cov2_inv, cov1))
        diff = mean2 - mean1
        mahalanobis = np.dot(np.dot(diff.T, cov2_inv), diff)
        det_ratio = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
        return 0.5 * (tr + mahalanobis - k + det_ratio)

    # Calculate the mixture distribution
    mix_mean = 0.5 * (p_mean + q_mean)
    mix_cov = 0.5 * (p_cov + q_cov)

    # Calculate Jensen-Shannon Divergence
    jsd = 0.5 * (kl_divergence(p_mean, p_cov, mix_mean, mix_cov) +
                 kl_divergence(q_mean, q_cov, mix_mean, mix_cov))

    return jsd