#include "FeatureSelect.h"
namespace features {
    SelectKBestWeighted::SelectKBestWeighted(samples_t& samples, labels_t& labels, weights_t& weights, int k, bool nat)
        : samples(samples), labels(labels), weights(weights), k(k), nat(nat)
    {
        if (samples.size() == 0 || samples[0].size() == 0)
            throw invalid_argument("features must be a non-empty matrix");
        if (samples.size() != labels.size())
            throw invalid_argument("number of samples and labels must be equal");
        if (samples.size() != weights.size())
            throw invalid_argument("number of samples and weights must be equal");
        if (k < 1 || k >  static_cast<int>(samples[0].size()))
            throw invalid_argument("k must be between 1 and number of features");
        numFeatures = 0;
        numClasses = 0;
        numSamples = 0;
        fitted = false;
    }
    void SelectKBestWeighted::SelectKBestWeighted::fit()
    {
        auto labelsCopy = labels;
        numFeatures = samples[0].size();
        numSamples = samples.size();
        // compute number of classes
        sort(labelsCopy.begin(), labelsCopy.end());
        auto last = unique(labelsCopy.begin(), labelsCopy.end());
        labelsCopy.erase(last, labelsCopy.end());
        numClasses = labelsCopy.size();
        // compute scores
        scores.reserve(numFeatures);
        for (int i = 0; i < numFeatures; ++i) {
            scores.push_back(MutualInformation(i));
            features.push_back(i);
        }
        // sort & reduce scores and features
        sort(features.begin(), features.end(), [&](int i, int j)
            { return scores[i] > scores[j]; });
        sort(scores.begin(), scores.end(), greater<precision_t>());
        features.resize(k);
        scores.resize(k);
        fitted = true;
    }
    precision_t SelectKBestWeighted::entropyLabel()
    {
        return entropy(labels);
    }
    precision_t SelectKBestWeighted::entropy(const sample_t& data)
    {
        precision_t ventropy = 0, totalWeight = 0;
        score_t counts(numClasses + 1, 0);
        for (auto i = 0; i < static_cast<int>(data.size()); ++i) {
            counts[data[i]] += weights[i];
            totalWeight += weights[i];
        }
        for (auto count : counts) {
            precision_t p = count / totalWeight;
            if (p > 0) {
                if (nat) {
                    ventropy -= p * log(p);
                } else {
                    ventropy -= p * log2(p);
                }
            }
        }
        return ventropy;
    }
    // H(Y|X) = sum_{x in X} p(x) H(Y|X=x)
    precision_t SelectKBestWeighted::conditionalEntropy(const int feature)
    {
        unordered_map<value_t, precision_t> featureCounts;
        unordered_map<value_t, unordered_map<value_t, precision_t>> jointCounts;
        featureCounts.clear();
        jointCounts.clear();
        precision_t totalWeight = 0;
        for (auto i = 0; i < numSamples; i++) {
            featureCounts[samples[i][feature]] += weights[i];
            jointCounts[samples[i][feature]][labels[i]] += weights[i];
            totalWeight += weights[i];
        }
        if (totalWeight == 0)
            throw invalid_argument("Total weight should not be zero");
        precision_t entropy = 0;
        for (auto& [feat, count] : featureCounts) {
            auto p_f = count / totalWeight;
            precision_t entropy_f = 0;
            for (auto& [label, jointCount] : jointCounts[feat]) {
                auto p_l_f = jointCount / count;
                if (p_l_f > 0) {
                    if (nat) {
                        entropy_f -= p_l_f * log(p_l_f);
                    } else {
                        entropy_f -= p_l_f * log2(p_l_f);
                    }
                }
            }
            entropy += p_f * entropy_f;
        }
        return entropy;
    }
    // I(X;Y) = H(Y) - H(Y|X)
    precision_t SelectKBestWeighted::MutualInformation(const int i)
    {
        return entropyLabel() - conditionalEntropy(i);
    }
    score_t SelectKBestWeighted::getScores() const
    {
        if (!fitted)
            throw logic_error("score not fitted");
        return scores;
    }
    //Return the indices of the selected features
    labels_t SelectKBestWeighted::getFeatures() const
    {
        if (!fitted)
            throw logic_error("score not fitted");
        return features;
    }
}
