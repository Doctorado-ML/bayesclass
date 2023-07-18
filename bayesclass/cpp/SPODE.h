#ifndef SPODE_H
#define SPODE_H
#include "BaseClassifier.h"
namespace bayesnet {
    class SPODE : public BaseClassifier {
    private:
        int root;
    protected:
        void train() override;
    public:
        SPODE(int root);
        vector<string> graph(string name = "SPODE") override;
    };
}
#endif