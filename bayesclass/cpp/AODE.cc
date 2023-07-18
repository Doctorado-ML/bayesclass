#include "AODE.h"

namespace bayesnet {
    AODE::AODE() : Ensemble() {}
    void AODE::train()
    {
        models.clear();
        for (int i = 0; i < features.size(); ++i) {
            models.push_back(std::make_unique<SPODE>(i));
        }
    }
    vector<string> AODE::graph(string title)
    {
        return Ensemble::graph(title);
    }
}