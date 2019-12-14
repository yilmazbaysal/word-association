import numpy as np
from data_preparer import DataPreparer
from utils import calculate_ppmi, apply_svd, apply_nmf, calculate_cosine_similarities, calculate_pearson_correlation, \
    apply_z_test

np.seterr(divide='ignore', invalid='ignore')  # To ignore error messages


# Prepare data
dp = DataPreparer()
matrix: np.ndarray = dp.read_matrix_file('../dataset/matrix.npz')  # Load matrix
word_pairs, mean_human_judgements = dp.read_word_pairs_file('../dataset/wordPairs.csv')  # Load word pairs

# Apply PPMI weighting
ppmi = calculate_ppmi(matrix)

#
# Calculate and print the results
print()
matrix_svd_result = calculate_cosine_similarities(apply_svd(matrix, components=25))
correlation_svd = calculate_pearson_correlation(matrix_svd_result, word_pairs, mean_human_judgements)
print('No Weighting - SVD:', correlation_svd)

ppmi_svd_result = calculate_cosine_similarities(apply_svd(ppmi, components=250))
correlation_ppmi_svd = calculate_pearson_correlation(ppmi_svd_result, word_pairs, mean_human_judgements)
print('PPMI Weighting - SVD:', correlation_ppmi_svd)

matrix_nmf_result = calculate_cosine_similarities(apply_nmf(matrix, components=25))
correlation_nmf = calculate_pearson_correlation(matrix_nmf_result, word_pairs, mean_human_judgements)
print('No Weighting - NMF:', correlation_nmf)

ppmi_nmf_result = calculate_cosine_similarities(apply_nmf(ppmi, components=200))
correlation_ppmi_nmf = calculate_pearson_correlation(ppmi_nmf_result, word_pairs, mean_human_judgements)
print('PPMI Weighting - NMF:', correlation_ppmi_nmf)
print()


#
# Statistical significance test with Z-Test

# Compare co-occurrence and ppmi on SVD
z_test_1 = apply_z_test(correlation_svd, correlation_ppmi_svd, len(dp.word_pairs))
print('Significance Test between SVD and PPMI-SVD   |   p-value:', z_test_1[1])

# Compare co-occurrence and ppmi on NMF
z_test_2 = apply_z_test(correlation_nmf, correlation_ppmi_nmf, len(dp.word_pairs))
print('Significance Test between NMF and PPMI-NMF   |   p-value:', z_test_2[1])

# Compare SVD and NMF with co-occurrence matrices
z_test_3 = apply_z_test(correlation_svd, correlation_nmf, len(dp.word_pairs))
print('Significance Test between SVD and NMF (co-occurrence)   |   p-value:', z_test_3[1])

# Compare SVD and NMF with PPMI weighted matrices
z_test_4 = apply_z_test(correlation_ppmi_svd, correlation_ppmi_nmf, len(dp.word_pairs))
print('Significance Test between SVD and NMF (PPMI weighted)   |   p-value:', z_test_4[1])
