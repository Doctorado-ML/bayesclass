#ifndef BAYESNET_METRICS_H
#define BAYESNET_METRICS_H
#include <torch/torch.h>
#include <vector>
#include <string>
namespace bayesnet {
    using namespace std;
    using namespace torch;
    class Metrics {
    private:
        Tensor samples;
        vector<string> features;
        string className;
        int classNumStates;
    public:
        Metrics() = default;
        Metrics(Tensor&, vector<string>&, string&, int);
        Metrics(const vector<vector<int>>&, const vector<int>&, const vector<string>&, const string&, const int);
        double entropy(Tensor&);
        double conditionalEntropy(Tensor&, Tensor&);
        double mutualInformation(Tensor&, Tensor&);
        vector<float> conditionalEdgeWeights();
        Tensor conditionalEdge();
        vector<pair<string, string>> doCombinations(const vector<string>&);
        vector<pair<int, int>> maximumSpanningTree(vector<string> features, Tensor& weights, int root);
    };
}
#endif