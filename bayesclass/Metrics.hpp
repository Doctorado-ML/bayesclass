#ifndef BAYESNET_METRICS_H
#define BAYESNET_METRICS_H
#include <torch/torch.h>
#include <vector>
#include <string>
using namespace std;
namespace bayesnet {
    class Metrics {
    private:
        torch::Tensor samples;
        vector<string> features;
        string className;
        int classNumStates;
        vector<pair<string, string>> doCombinations(const vector<string>&);
        double entropy(torch::Tensor&);
        double conditionalEntropy(torch::Tensor&, torch::Tensor&);
    public:
        double mutualInformation(torch::Tensor&, torch::Tensor&);
        Metrics(torch::Tensor&, vector<string>&, string&, int);
        Metrics(const vector<vector<int>>&, const vector<int>&, const vector<string>&, const string&, const int);
        vector<float> conditionalEdgeWeights();
        torch::Tensor conditionalEdge();
    };
}
#endif