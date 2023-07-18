#ifndef TAN_H
#define TAN_H
#include "BaseClassifier.h"
namespace bayesnet {
    using namespace std;
    using namespace torch;
    class TAN : public BaseClassifier {
    private:
    protected:
        void train() override;
    public:
        TAN();
        vector<string> graph(string name = "TAN") override;
    };
}
#endif