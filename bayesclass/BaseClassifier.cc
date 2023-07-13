#include "BaseClassifier.h"

namespace bayesnet {
    using namespace std;
    using namespace torch;

    BaseClassifier::BaseClassifier(Network model) : model(model), m(0), n(0) {}
    BaseClassifier& BaseClassifier::build(vector<string>& features, string className, map<string, vector<int>>& states)
    {

        dataset = torch::cat({ X, y.view({150, 1}) }, 1);
        this->features = features;
        this->className = className;
        this->states = states;
        checkFitParameters();
        train();
        return *this;
    }
    BaseClassifier& BaseClassifier::fit(Tensor& X, Tensor& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = X;
        this->y = y;
        return build(features, className, states);
    }
    BaseClassifier& BaseClassifier::fit(vector<vector<int>>& X, vector<int>& y, vector<string>& features, string className, map<string, vector<int>>& states)
    {
        this->X = torch::zeros({ static_cast<int64_t>(X[0].size()), static_cast<int64_t>(X.size()) }, kInt64);
        for (int i = 0; i < X.size(); ++i) {
            this->X.index_put_({ "...", i }, torch::tensor(X[i], kInt64));
        }
        this->y = torch::tensor(y, kInt64);
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
    vector<vector<int>> tensorToVector(const torch::Tensor& tensor)
    {
        // convert mxn tensor to nxm vector
        vector<vector<int>> result;
        auto tensor_accessor = tensor.accessor<int, 2>();

        // Iterate over columns and rows of the tensor
        for (int j = 0; j < tensor.size(1); ++j) {
            vector<int> column;
            for (int i = 0; i < tensor.size(0); ++i) {
                column.push_back(tensor_accessor[i][j]);
            }
            result.push_back(column);
        }

        return result;
    }
    Tensor BaseClassifier::predict(Tensor& X)
    {
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
    float BaseClassifier::score(Tensor& X, Tensor& y)
    {
        Tensor y_pred = predict(X);
        return (y_pred == y).sum().item<float>() / y.size(0);
    }
    void BaseClassifier::show()
    {
        model.show();
    }
}