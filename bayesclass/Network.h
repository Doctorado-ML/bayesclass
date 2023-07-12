#ifndef NETWORK_H
#define NETWORK_H
#include "Node.h"
#include <map>
#include <vector>


namespace bayesnet {
    class Network {
    private:
        map<string, Node*> nodes;
        map<string, vector<int>> dataset;
        float maxThreads;
        int classNumStates;
        vector<string> features;
        string className;
        int laplaceSmoothing;
        torch::Tensor samples;
        bool isCyclic(const std::string&, std::unordered_set<std::string>&, std::unordered_set<std::string>&);
        vector<double> predict_sample(const vector<int>&);
        vector<double> exactInference(map<string, int>&);
        double computeFactor(map<string, int>&);
        double mutual_info(torch::Tensor&, torch::Tensor&);
        double entropy(torch::Tensor&);
        double conditionalEntropy(torch::Tensor&, torch::Tensor&);
        double mutualInformation(torch::Tensor&, torch::Tensor&);
    public:
        Network();
        Network(float, int);
        Network(float);
        Network(Network&);
        ~Network();
        torch::Tensor& getSamples();
        float getmaxThreads();
        void addNode(string, int);
        void addEdge(const string, const string);
        map<string, Node*>& getNodes();
        vector<string> getFeatures();
        int getStates();
        int getClassNumStates();
        string getClassName();
        void fit(const vector<vector<int>>&, const vector<int>&, const vector<string>&, const string&);
        vector<int> predict(const vector<vector<int>>&);
        //Computes the conditional edge weight of variable index u and v conditioned on class_node
        torch::Tensor conditionalEdgeWeight();
        vector<vector<double>> predict_proba(const vector<vector<int>>&);
        double score(const vector<vector<int>>&, const vector<int>&);
        inline string version() { return "0.1.0"; }
    };
}
#endif