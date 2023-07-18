#include "Ensemble.h"

namespace bayesnet {
    using namespace std;
    using namespace torch;

    Ensemble::Ensemble() : m(0), n(0), n_models(0), metrics(Metrics()), fitted(false) {}
    Ensemble& Ensemble::build(vector<string>& features, string className, map<string, vector<int>>& states)
    {
        dataset = cat({ X, y.view({y.size(0), 1}) }, 1);
        this->features = features;
        this->className = className;
        this->states = states;
        auto n_classes = states[className].size();
        metrics = Metrics(dataset, features, className, n_classes);
        // Build models
        train();
        // Train models
        n_models = models.size();
        for (auto i = 0; i < n_models; ++i) {
            models[i]->fit(Xv, yv, features, className, states);
        }
        fitted = true;
        return *this;
    }
    Ensemble& Ensemble::fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = torch::zeros({ static_cast<int64_t>(X[0].size()), static_cast<int64_t>(X.size()) }, kInt64);
        Xv = X;
        for (int i = 0; i < X.size(); ++i) {
            this->X.index_put_({ "...", i }, torch::tensor(X[i], kInt64));
        }
        this->y = torch::tensor(y, kInt64);
        yv = y;
        return build(features, className, states);
    }
    Tensor Ensemble::predict(Tensor& X)
    {
        if (!fitted) {
            throw logic_error("Ensemble has not been fitted");
        }
        Tensor y_pred = torch::zeros({ X.size(0), n_models }, kInt64);
        for (auto i = 0; i < n_models; ++i) {
            y_pred.index_put_({ "...", i }, models[i]->predict(X));
        }
        return torch::tensor(voting(y_pred));
    }
    vector<int> Ensemble::voting(Tensor& y_pred)
    {
        auto y_pred_ = y_pred.accessor<int64_t, 2>();
        vector<int> y_pred_final;
        for (int i = 0; i < y_pred.size(0); ++i) {
            vector<float> votes(states[className].size(), 0);
            for (int j = 0; j < y_pred.size(1); ++j) {
                votes[y_pred_[i][j]] += 1;
            }
            auto indices = argsort(votes);
            y_pred_final.push_back(indices[0]);
        }
        return y_pred_final;
    }
    vector<int> Ensemble::predict(vector<vector<int>>& X)
    {
        if (!fitted) {
            throw logic_error("Ensemble has not been fitted");
        }
        long m_ = X[0].size();
        long n_ = X.size();
        vector<vector<int>> Xd(n_, vector<int>(m_, 0));
        for (auto i = 0; i < n_; i++) {
            Xd[i] = vector<int>(X[i].begin(), X[i].end());
        }
        Tensor y_pred = torch::zeros({ m_, n_models }, kInt64);
        for (auto i = 0; i < n_models; ++i) {
            y_pred.index_put_({ "...", i }, torch::tensor(models[i]->predict(Xd), kInt64));
        }
        return voting(y_pred);
    }
    float Ensemble::score(vector<vector<int>>& X, vector<int>& y)
    {
        if (!fitted) {
            throw logic_error("Ensemble has not been fitted");
        }
        auto y_pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y_pred.size(); ++i) {
            if (y_pred[i] == y[i]) {
                correct++;
            }
        }
        return (double)correct / y_pred.size();

    }
    vector<string> Ensemble::show()
    {
        auto result = vector<string>();
        for (auto i = 0; i < n_models; ++i) {
            auto res = models[i]->show();
            result.insert(result.end(), res.begin(), res.end());
        }
        return result;
    }
    vector<string> Ensemble::graph(string title)
    {
        auto result = vector<string>();
        for (auto i = 0; i < n_models; ++i) {
            auto res = models[i]->graph(title + "_" + to_string(i));
            result.insert(result.end(), res.begin(), res.end());
        }
        return result;
    }
}