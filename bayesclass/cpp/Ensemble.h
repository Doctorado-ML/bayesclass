#ifndef ENSEMBLE_H
#define ENSEMBLE_H
#include <torch/torch.h>
#include "BaseClassifier.h"
#include "Metrics.hpp"
#include "utils.h"
using namespace std;
using namespace torch;

namespace bayesnet {
    class Ensemble {
    private:
        bool fitted;
        long n_models;
        Ensemble& build(vector<string>& features, string className, map<string, vector<int>>& states);
    protected:
        vector<unique_ptr<BaseClassifier>> models;
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
        void virtual train() = 0;
        vector<int> voting(Tensor& y_pred);
    public:
        Ensemble();
        virtual ~Ensemble() = default;
        Ensemble& fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states);
        Tensor predict(Tensor& X);
        vector<int> predict(vector<vector<int>>& X);
        float score(Tensor& X, Tensor& y);
        float score(vector<vector<int>>& X, vector<int>& y);
        vector<string> show();
        vector<string> graph(string title);
    };
}
#endif
