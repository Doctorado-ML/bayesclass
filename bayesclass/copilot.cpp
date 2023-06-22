#include <unordered_map>
#include <vector>
#include <cmath>

using namespace std;

using value_t = int;
using precision_t = double;
using sample_t = vector<value_t>;
using score_t = vector<precision_t>;

precision_t entropy(const sample_t& data, const vector<precision_t>& weights)
{
    precision_t p;
    precision_t ventropy = 0, totalWeight = 0;
    score_t counts(*max_element(data.begin(), data.end()) + 1, 0);
    for (auto i = 0; i < data.size(); ++i) {
        counts[data[i]] += weights[i];
        totalWeight += weights[i];
    }
    for (auto count : counts) {
        if (count > 0 || count < 0) {
            p = (count) / totalWeight;
            ventropy -= p * log2(p);
        }
    }
    return ventropy;
}

precision_t conditionalEntropy(const sample_t& feature, const sample_t& labels, const vector<precision_t>& weights)
{
    unordered_map<value_t, precision_t> featureCounts;
    unordered_map<value_t, unordered_map<value_t, precision_t>> jointCounts;
    featureCounts.clear();
    jointCounts.clear();
    auto totalWeight = 0.0;
    for (auto i = 0; i < feature.size(); i++) {
        featureCounts[feature[i]] += weights[i];
        jointCounts[feature[i]][labels[i]] += weights[i];
        totalWeight += weights[i];
    }
    precision_t entropy = 0;
    for (auto& [f, count] : featureCounts) {
        auto p_f = count / totalWeight;
        entropy += p_f * ::entropy(jointCounts[f], weights) / ::entropy(feature, weights);
    }
    return entropy;
}