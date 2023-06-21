import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.feature_selection._univariate_selection import (
    _BaseFilter,
    _clean_nans,
)

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


class SelectKBestWeighted(_BaseFilter):
    def __init__(self, *, k=10):
        super().__init__(score_func=mutual_info_classif)
        self.k = k

    def _check_params(self, X, y):
        if self.k > X.shape[1] or self.k < 1:
            raise ValueError(
                f"k must be between 1 and {X.shape[1]} got {self.k}."
            )

    def _get_support_mask(self):
        check_is_fitted(self)

        if self.k == "all":
            return np.ones(self.scores_.shape, dtype=bool)
        elif self.k == 0:
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = _clean_nans(self.scores_)
            mask = np.zeros(scores.shape, dtype=bool)

            # Request a stable sort. Mergesort takes more memory (~40MB per
            # megafeature on x86-64).
            mask[np.argsort(scores, kind="mergesort")[-self.k :]] = 1
            return mask

    def fit(self, X, y, sample_weight):
        self.X_, self.y_ = check_X_y(X, y)
        self._check_params(X, y)
        self.n_features_in_ = X.shape[1]
        self.sample_weight_ = sample_weight
        # Compute the entropy of the target variable
        entropy_y = -np.sum(
            np.multiply(
                np.bincount(y, weights=sample_weight),
                np.log(np.bincount(y, weights=sample_weight)),
            )
        )

        # Compute the mutual information between each feature and the target
        mi = self.score_func(X, y)

        # Compute the weighted entropy of each feature
        entropy_weighted = []
        for i in range(X.shape[1]):
            # Compute the weighted frequency of each unique value of the
            # feature
            freq_weighted = np.bincount(X[:, i], weights=sample_weight)
            freq_weighted = freq_weighted[freq_weighted != 0]

            # Compute the weighted entropy of the feature
            entropy_weighted.append(
                -np.sum(np.multiply(freq_weighted, np.log(freq_weighted)))
                / np.sum(sample_weight)
            )

        # Compute the weighted mutual information between each feature and
        # the target
        mi_weighted = mi * entropy_weighted / entropy_y

        # Return the weighted mutual information scores
        self.scores_ = mi_weighted
        return self
