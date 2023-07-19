#ifndef CLASSIFIERS_H
#define CLASSIFIERS_H
#include <torch/torch.h>
#include "Network.h"
#include "Metrics.hpp"
using namespace std;
using namespace torch;

namespace bayesnet {
    class BaseClassifier {
    private:
        bool fitted;
        BaseClassifier& build(vector<string>& features, string className, map<string, vector<int>>& states);
    protected:
        Network model;
        int m, n; // m: number of samples, n: number of features
        Tensor X;
        vector<vector<int>> Xv;
        Tensor y;
        vector<int> yv;
        Tensor dataset;
        Metrics metrics;
        vector<string> features;
        string className;
        map<string, vector<int>> states;
        void checkFitParameters();
        virtual void train() = 0;
    public:
        BaseClassifier(Network model);
        virtual ~BaseClassifier() = default;
        BaseClassifier& fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states);
        void addNodes();
        int getNumberOfNodes();
        int getNumberOfEdges();
        Tensor predict(Tensor& X);
        vector<int> predict(vector<vector<int>>& X);
        float score(Tensor& X, Tensor& y);
        float score(vector<vector<int>>& X, vector<int>& y);
        vector<string> show();
        virtual vector<string> graph(string title) = 0;
    };
}
#endif





