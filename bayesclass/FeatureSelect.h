#ifndef SELECT_K_BEST_WEIGHTED_H
#define SELECT_K_BEST_WEIGHTED_H
#include <map>
#include <vector>
#include <string>
using namespace std;
namespace features {
    typedef float precision_t;
    typedef int value_t;
    typedef vector<value_t> sample_t;
    typedef vector<sample_t> samples_t;
    typedef vector<value_t> labels_t;
    typedef vector<precision_t> score_t, weights_t;

    class SelectKBestWeighted {
    private:
        const samples_t samples;
        const labels_t labels;
        const weights_t weights;
        const int k;
        bool nat; // use natural log or log2
        int numFeatures, numClasses, numSamples;
        bool fitted;
        score_t scores; // scores of the features
        labels_t features; // indices of the selected features
        precision_t entropyLabel();
        precision_t entropy(const sample_t&);
        precision_t conditionalEntropy(const int);
        precision_t MutualInformation(const int);
    public:
        SelectKBestWeighted(samples_t&, labels_t&, weights_t&, int, bool);
        void fit();
        score_t getScores() const;
        labels_t getFeatures() const; //Return the indices of the selected features
        static inline string version() { return "0.1.0"; };
    };
}
#endif