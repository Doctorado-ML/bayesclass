#include "FeatureSelect.h"
#include <iostream>
namespace features {
    // SelectKBestWeighted::SelectKBestWeighted(samples_t& samples, labels_t& labels, weights_t& weights, int k)
    //     : samples(samples), labels(labels), weights(weights), k(k)
    // {
    //     // if (samples.size() == 0 || samples[0].size() == 0)
    //     //     throw invalid_argument("features must be a non-empty matrix");
    //     // if (samples.size() != labels.size())
    //     //     throw invalid_argument("number of samples and labels must be equal");
    //     // if (samples.size() != weights.size())
    //     //     throw invalid_argument("number of samples and weights must be equal");
    //     // if (k < 1 || k >  static_cast<int>(samples[0].size()))
    //     //     throw invalid_argument("k must be between 1 and number of features");
    //     numFeatures = 0;
    //     numClasses = 0;
    //     numSamples = 0;
    //     fitted = false;
    // }
    SelectKBestWeighted::SelectKBestWeighted(samples_t& samples) : samples(samples) {}
    void SelectKBestWeighted::SelectKBestWeighted::fit()
    {
        // auto labelsCopy = labels;
        numFeatures = 0;//samples[0].size();
        numSamples = samples.size();
        // sort(labelsCopy.begin(), labelsCopy.end());
        // auto last = unique(labelsCopy.begin(), labelsCopy.end());
        // labelsCopy.erase(last, labelsCopy.end());
        // numClasses = labelsCopy.size();
        // score.reserve(numFeatures);
        // for (int i = 0; i < numFeatures; ++i) {
        //     score.push_back(MutualInformation(i));
        // }
        outputValues();
        fitted = true;
    }
    void SelectKBestWeighted::outputValues()
    {
        cout << "numFeatures: " << numFeatures << endl;
        // cout << "numClasses: " << numClasses << endl;
        cout << "numSamples: " << numSamples << endl;
        // cout << "k: " << k << endl;
        // cout << "weights: ";
        // for (auto item : weights)
        //     cout << item << ", ";
        // cout << "end." << endl;
        // cout << "labels: ";
        // for (auto item : labels)
        //     cout << item << ", ";
        // cout << "end." << endl;
        cout << "samples: ";
        for (auto item : samples) {
            // for (auto item2 : item)
            //     cout << item2 << ", ";
            // cout << "end." << endl;
            cout << item << ", ";
        }
        cout << "end." << endl;
    }
    // precision_t SelectKBestWeighted::entropyLabel()
    // {
    //     return entropy(labels);
    // }
    // precision_t SelectKBestWeighted::entropy(const sample_t& data)
    // {
    //     precision_t p;
    //     precision_t ventropy = 0, totalWeight = 0;
    //     score_t counts(numClasses + 1, 0);
    //     for (auto i = 0; i < data.size(); ++i) {
    //         counts[data[i]] += weights[i];
    //         totalWeight += weights[i];
    //     }
    //     for (auto count : counts) {
    //         p = count / totalWeight;
    //         ventropy -= p * log2(p);
    //     }
    //     return ventropy;
    // }
    // // H(Y|X) = sum_{x in X} p(x) H(Y|X=x)
    // precision_t SelectKBestWeighted::conditionalEntropy(const int feature)
    // {
    //     unordered_map<value_t, precision_t> featureCounts;
    //     unordered_map<value_t, unordered_map<value_t, precision_t>> jointCounts;
    //     featureCounts.clear();
    //     jointCounts.clear();
    //     auto totalWeight = 0;
    //     for (auto i = 0; i < numSamples; i++) {
    //         featureCounts[samples[feature][i]] += weights[i];
    //         jointCounts[samples[feature][i]][labels[i]] += weights[i];
    //         totalWeight += weights[i];
    //     }
    //     precision_t entropy = 0;
    //     for (auto& [f, count] : featureCounts) {
    //         auto p_f = count / totalWeight;
    //         precision_t entropy_f = 0;
    //         for (auto& [l, jointCount] : jointCounts[f]) {
    //             auto p_l_f = jointCount / totalWeight;
    //             entropy_f -= p_l_f * log2(p_l_f);
    //         }
    //         entropy += p_f * entropy_f;
    //     }
    //     return entropy;
    // }

    // // I(X;Y) = H(Y) - H(Y|X)
    // precision_t SelectKBestWeighted::MutualInformation(const int i)
    // {
    //     // return entropyLabel() - conditionalEntropy(i);
    //     return 25 / (i + 1);
    // }
    score_t SelectKBestWeighted::getScore() const
    {
        if (!fitted)
            throw logic_error("score not fitted");
        return score;
    }
}

// using namespace std;

// int main()
// {
//     vector<vector<int>> samples = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
//     vector<int> labels = { 1, 2, 1 };
//     vector<float> weights = { 0.1, 0.7, 0.2 };
//     int k = 3;
//     auto metric = features::SelectKBestWeighted(samples, labels, weights, k);
//     metric.fit();
//     cout << "score: ";
//     for (auto item : metric.getScore())
//         cout << item << ", ";
//     cout << "end." << endl;
//     return 0;
// }
