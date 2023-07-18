#include "BaseClassifier.h"
#include "utils.h"

namespace bayesnet {
    using namespace std;
    using namespace torch;

    BaseClassifier::BaseClassifier(Network model) : model(model), m(0), n(0), metrics(Metrics()), fitted(false) {}
    BaseClassifier& BaseClassifier::build(vector<string>& features, string className, map<string, vector<int>>& states)
    {

        dataset = torch::cat({ X, y.view({y.size(0), 1}) }, 1);
        this->features = features;
        this->className = className;
        this->states = states;
        checkFitParameters();
        auto n_classes = states[className].size();
        metrics = Metrics(dataset, features, className, n_classes);
        train();
        model.fit(Xv, yv, features, className);
        fitted = true;
        return *this;
    }
    BaseClassifier& BaseClassifier::fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states)
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
    void BaseClassifier::checkFitParameters()
    {
        auto sizes = X.sizes();
        m = sizes[0];
        n = sizes[1];
        if (m != y.size(0)) {
            throw invalid_argument("X and y must have the same number of samples");
        }
        if (n != features.size()) {
            throw invalid_argument("X and features must have the same number of features");
        }
        if (states.find(className) == states.end()) {
            throw invalid_argument("className not found in states");
        }
        for (auto feature : features) {
            if (states.find(feature) == states.end()) {
                throw invalid_argument("feature [" + feature + "] not found in states");
            }
        }
    }

    Tensor BaseClassifier::predict(Tensor& X)
    {
        if (!fitted) {
            throw logic_error("Classifier has not been fitted");
        }
        auto m_ = X.size(0);
        auto n_ = X.size(1);
        vector<vector<int>> Xd(n_, vector<int>(m_, 0));
        for (auto i = 0; i < n_; i++) {
            auto temp = X.index({ "...", i });
            Xd[i] = vector<int>(temp.data_ptr<int>(), temp.data_ptr<int>() + m_);
        }
        auto yp = model.predict(Xd);
        auto ypred = torch::tensor(yp, torch::kInt64);
        return ypred;
    }
    vector<int> BaseClassifier::predict(vector<vector<int>>& X)
    {
        if (!fitted) {
            throw logic_error("Classifier has not been fitted");
        }
        auto m_ = X[0].size();
        auto n_ = X.size();
        vector<vector<int>> Xd(n_, vector<int>(m_, 0));
        for (auto i = 0; i < n_; i++) {
            Xd[i] = vector<int>(X[i].begin(), X[i].end());
        }
        auto yp = model.predict(Xd);
        return yp;
    }
    float BaseClassifier::score(Tensor& X, Tensor& y)
    {
        if (!fitted) {
            throw logic_error("Classifier has not been fitted");
        }
        Tensor y_pred = predict(X);
        return (y_pred == y).sum().item<float>() / y.size(0);
    }
    float BaseClassifier::score(vector<vector<int>>& X, vector<int>& y)
    {
        if (!fitted) {
            throw logic_error("Classifier has not been fitted");
        }
        auto m_ = X[0].size();
        auto n_ = X.size();
        vector<vector<int>> Xd(n_, vector<int>(m_, 0));
        for (auto i = 0; i < n_; i++) {
            Xd[i] = vector<int>(X[i].begin(), X[i].end());
        }
        return model.score(Xd, y);
    }
    vector<string> BaseClassifier::show()
    {
        return model.show();
    }
    void BaseClassifier::addNodes()
    {
        // Add all nodes to the network
        for (auto feature : features) {
            model.addNode(feature, states[feature].size());
        }
        model.addNode(className, states[className].size());
    }
}