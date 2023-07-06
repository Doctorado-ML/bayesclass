#ifndef NODE_H
#define NODE_H
#include <torch/torch.h>
//#include <torch/extension.h>
#include <vector>
#include <string>
namespace bayesnet {
    using namespace std;
    class Node {
    private:
        string name;
        vector<Node*> parents;
        vector<Node*> children;
        int numStates; // number of states of the variable
        torch::Tensor cpTable; // Order of indices is 0-> node variable, 1-> 1st parent, 2-> 2nd parent, ...
        vector<int64_t> dimensions; // dimensions of the cpTable
        vector<string> combinations(const set<string>&);
    public:
        Node(const std::string&, int);
        void addParent(Node*);
        void addChild(Node*);
        void removeParent(Node*);
        void removeChild(Node*);
        string getName() const;
        vector<Node*>& getParents();
        vector<Node*>& getChildren();
        torch::Tensor& getCPT();
        void computeCPT(map<string, vector<int>>&, const int);
        int getNumStates() const;
        void setNumStates(int);
        unsigned minFill();
        float getFactorValue(map<string, int>&);
    };
}
#endif