#include "FeatureTest.h"
#include <iostream>
namespace featuresTest {
    SelectKBest::SelectKBest(vector<int>& samples) : samples(samples) {}
    SelectKBest::SelectKBest() = default;
    SelectKBest::~SelectKBest() = default;
    void SelectKBest::SelectKBest::fit()
    {
        numFeatures = 0;
        numSamples = samples.size();
        outputValues();
        fitted = true;
    }
    void SelectKBest::outputValues()
    {
        cout << "numFeatures: " << numFeatures << endl;
        // cout << "numClasses: " << numClasses << endl;
        cout << "numSamples: " << numSamples << endl;
        // cout << "k: " << k << endl;
        // cout << "weights: ";
        // for (auto item : weights)
        //     cout << item << ", ";
        // cout << "end." << endl;
        // cout << "labels: ";
        // for (auto item : labels)
        //     cout << item << ", ";
        // cout << "end." << endl;
        cout << "samples: ";
        for (auto item : samples) {
            // for (auto item2 : item)
            //     cout << item2 << ", ";
            // cout << "end." << endl;
            cout << item << ", ";
        }
        cout << "end." << endl;
    }
}