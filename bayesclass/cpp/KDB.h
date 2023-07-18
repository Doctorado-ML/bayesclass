#ifndef KDB_H
#define KDB_H
#include "BaseClassifier.h"
#include "utils.h"
namespace bayesnet {
    using namespace std;
    using namespace torch;
    class KDB : public BaseClassifier {
    private:
        int k;
        float theta;
        void add_m_edges(int idx, vector<int>& S, Tensor& weights);
    protected:
        void train() override;
    public:
        KDB(int k, float theta = 0.03);
        vector<string> graph(string name = "KDB") override;
    };
}
#endif