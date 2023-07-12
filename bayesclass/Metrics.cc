#include "Metrics.hpp"
using namespace std;
namespace bayesnet {
    Metrics::Metrics(torch::Tensor& samples, vector<string>& features, string& className, int classNumStates)
        : samples(samples)
        , features(features)
        , className(className)
        , classNumStates(classNumStates)
    {
    }
    Metrics::Metrics(const vector<vector<int>>& vsamples, const vector<int>& labels, const vector<string>& features, const string& className, const int classNumStates)
        : features(features)
        , className(className)
        , classNumStates(classNumStates)
    {
        samples = torch::zeros({ static_cast<int64_t>(vsamples[0].size()), static_cast<int64_t>(vsamples.size() + 1) }, torch::kInt64);
        for (int i = 0; i < vsamples.size(); ++i) {
            samples.index_put_({ "...", i }, torch::tensor(vsamples[i], torch::kInt64));
        }
        samples.index_put_({ "...", -1 }, torch::tensor(labels, torch::kInt64));
    }
    vector<pair<string, string>> Metrics::doCombinations(const vector<string>& source)
    {
        vector<pair<string, string>> result;
        for (int i = 0; i < source.size(); ++i) {
            string temp = source[i];
            for (int j = i + 1; j < source.size(); ++j) {
                result.push_back({ temp, source[j] });
            }
        }
        return result;
    }
    vector<float> Metrics::conditionalEdgeWeights()
    {
        auto result = vector<double>();
        auto source = vector<string>(features);
        source.push_back(className);
        auto combinations = doCombinations(source);
        // Compute class prior
        auto margin = torch::zeros({ classNumStates });
        for (int value = 0; value < classNumStates; ++value) {
            auto mask = samples.index({ "...", -1 }) == value;
            margin[value] = mask.sum().item<float>() / samples.sizes()[0];
        }
        for (auto [first, second] : combinations) {
            int64_t index_first = find(features.begin(), features.end(), first) - features.begin();
            int64_t index_second = find(features.begin(), features.end(), second) - features.begin();
            double accumulated = 0;
            for (int value = 0; value < classNumStates; ++value) {
                auto mask = samples.index({ "...", -1 }) == value;
                auto first_dataset = samples.index({ mask, index_first });
                auto second_dataset = samples.index({ mask, index_second });
                auto mi = mutualInformation(first_dataset, second_dataset);
                auto pb = margin[value].item<float>();
                accumulated += pb * mi;
            }
            result.push_back(accumulated);
        }
        long n_vars = source.size();
        auto matrix = torch::zeros({ n_vars, n_vars });
        auto indices = torch::triu_indices(n_vars, n_vars, 1);
        for (auto i = 0; i < result.size(); ++i) {
            auto x = indices[0][i];
            auto y = indices[1][i];
            matrix[x][y] = result[i];
            matrix[y][x] = result[i];
        }
        std::vector<float> v(matrix.data_ptr<float>(), matrix.data_ptr<float>() + matrix.numel());
        return v;
    }
    double Metrics::entropy(torch::Tensor& feature)
    {
        torch::Tensor counts = feature.bincount();
        int totalWeight = counts.sum().item<int>();
        torch::Tensor probs = counts.to(torch::kFloat) / totalWeight;
        torch::Tensor logProbs = torch::log(probs);
        torch::Tensor entropy = -probs * logProbs;
        return entropy.nansum().item<double>();
    }
    // H(Y|X) = sum_{x in X} p(x) H(Y|X=x)
    double Metrics::conditionalEntropy(torch::Tensor& firstFeature, torch::Tensor& secondFeature)
    {
        int numSamples = firstFeature.sizes()[0];
        torch::Tensor featureCounts = secondFeature.bincount();
        unordered_map<int, unordered_map<int, double>> jointCounts;
        double totalWeight = 0;
        for (auto i = 0; i < numSamples; i++) {
            jointCounts[secondFeature[i].item<int>()][firstFeature[i].item<int>()] += 1;
            totalWeight += 1;
        }
        if (totalWeight == 0)
            throw invalid_argument("Total weight should not be zero");
        double entropyValue = 0;
        for (int value = 0; value < featureCounts.sizes()[0]; ++value) {
            double p_f = featureCounts[value].item<double>() / totalWeight;
            double entropy_f = 0;
            for (auto& [label, jointCount] : jointCounts[value]) {
                double p_l_f = jointCount / featureCounts[value].item<double>();
                if (p_l_f > 0) {
                    entropy_f -= p_l_f * log(p_l_f);
                } else {
                    entropy_f = 0;
                }
            }
            entropyValue += p_f * entropy_f;
        }
        return entropyValue;
    }
    // I(X;Y) = H(Y) - H(Y|X)
    double Metrics::mutualInformation(torch::Tensor& firstFeature, torch::Tensor& secondFeature)
    {
        return entropy(firstFeature) - conditionalEntropy(firstFeature, secondFeature);
    }
}