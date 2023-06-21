import numpy as np
from sklearn.feature_selection import mutual_info_classif


"""
Compute the weighted mutual information between each feature and the
target.
Based on
Silviu Guiaşu,
Weighted entropy,
Reports on Mathematical Physics,
Volume 2, Issue 3,
1971,
Pages 165-179,
ISSN 0034-4877,
https://doi.org/10.1016/0034-4877(71)90002-4.
(https://www.sciencedirect.com/science/article/pii/0034487771900024)
Abstract: Weighted entropy is the measure of information supplied by a 
probablistic experiment whose elementary events are characterized both by their
objective probabilities and by some qualitative (objective or subjective) 
weights. The properties, the axiomatics and the maximum value of the weighted 
entropy are given.
"""


class SelectKBestWeighted:
    def fit(self, X, y, sample_weight):
        self.X_ = X
        self.y_ = y
        self.sample_weight_ = sample_weight
        # Compute the entropy of the target variable
        entropy_y = -np.sum(
            np.multiply(
                np.bincount(y, weights=sample_weight),
                np.log(np.bincount(y, weights=sample_weight)),
            )
        )

        # Compute the mutual information between each feature and the target
        mi = mutual_info_classif(X, y)

        # Compute the weighted entropy of each feature
        entropy_weighted = []
        for i in range(X.shape[1]):
            # Compute the weighted frequency of each unique value of the feature
            freq_weighted = np.bincount(X[:, i], weights=sample_weight)
            freq_weighted = freq_weighted[freq_weighted != 0]

            # Compute the weighted entropy of the feature
            entropy_weighted.append(
                -np.sum(np.multiply(freq_weighted, np.log(freq_weighted)))
                / np.sum(sample_weight)
            )

        # Compute the weighted mutual information between each feature and the target
        mi_weighted = mi * entropy_weighted / entropy_y

        # Return the weighted mutual information scores
        self.mi_weighted_ = mi_weighted
