#ifndef SELECT_K_BEST_TEST_H
#define SELECT_K_BEST_TEST_H
#include <map>
#include <vector>
#include <string>
using namespace std;
namespace featuresTest {
    typedef float precision_t;
    typedef int value_t;
    typedef vector<value_t> sample_t;
    // typedef vector<sample_t> samples_t;
    typedef vector<value_t> samples_t;
    typedef vector<value_t> labels_t;
    typedef vector<precision_t> score_t, weights_t;

    class SelectKBest {
    private:
        vector<int>& samples;
        int numFeatures, numClasses, numSamples;
        bool fitted;
        void outputValues();
    public:
        SelectKBest();
        SelectKBest(vector<int>&);
        ~SelectKBest();
        void fit();
        static inline string version() { return "0.1.0"; };
    };
}
#endif