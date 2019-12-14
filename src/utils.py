from math import sqrt

import numpy as np
import scipy.stats
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics import pairwise_distances


def calculate_ppmi(matrix: np.ndarray):
    denominator = (matrix.sum(axis=0) / matrix.sum()).T * (matrix.sum(axis=1) / matrix.sum()).T
    numerator = (matrix.T / matrix.sum(axis=1)).T
    pmi = np.log2(numerator / denominator)

    return np.nan_to_num(pmi.clip(min=0))  # Set zero to negative values


def apply_svd(matrix: np.ndarray, components: int):
    svd = TruncatedSVD(components)
    return svd.fit_transform(matrix)


def apply_nmf(matrix: np.ndarray, components: int):
    nmf = NMF(components)
    return nmf.fit_transform(matrix)


def calculate_cosine_similarities(matrix: np.ndarray):
    return 1 - pairwise_distances(matrix, metric='cosine')


def calculate_pearson_correlation(similarities: np.ndarray, word_pairs: list, mean_human_judgements):
    sum_of_similarities = 0
    for w1, w2, avg_hj in word_pairs:
        sum_of_similarities += similarities[w1, w2]

    mean_of_similarities = sum_of_similarities / len(word_pairs)

    numerator_sum = 0
    denominator1_sum = 0
    denominator2_sum = 0
    for w1, w2, avg_hj in word_pairs:
        numerator_sum += (avg_hj - mean_human_judgements) * (similarities[w1, w2] - mean_of_similarities)
        denominator1_sum += (avg_hj - mean_human_judgements)**2
        denominator2_sum += (similarities[w1, w2] - mean_of_similarities) ** 2

    return numerator_sum / (sqrt(denominator1_sum) * sqrt(denominator2_sum))


def apply_z_test(correlation_1, correlation_2, n):
    z = (np.arctanh(correlation_1) - np.arctanh(correlation_2)) / sqrt(abs((1/(n - 3) + 1/(n - 3))))
    p = scipy.stats.norm.sf(abs(z))
    return z, p
