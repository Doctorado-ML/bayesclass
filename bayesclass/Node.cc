#include "Node.h"

namespace bayesnet {

    Node::Node(const std::string& name, int numStates)
        : name(name), numStates(numStates), cpTable(torch::Tensor()), parents(vector<Node*>()), children(vector<Node*>())
    {
    }

    string Node::getName() const
    {
        return name;
    }

    void Node::addParent(Node* parent)
    {
        parents.push_back(parent);
    }
    void Node::removeParent(Node* parent)
    {
        parents.erase(std::remove(parents.begin(), parents.end(), parent), parents.end());
    }
    void Node::removeChild(Node* child)
    {
        children.erase(std::remove(children.begin(), children.end(), child), children.end());
    }
    void Node::addChild(Node* child)
    {
        children.push_back(child);
    }
    vector<Node*>& Node::getParents()
    {
        return parents;
    }
    vector<Node*>& Node::getChildren()
    {
        return children;
    }
    int Node::getNumStates() const
    {
        return numStates;
    }
    void Node::setNumStates(int numStates)
    {
        this->numStates = numStates;
    }
    torch::Tensor& Node::getCPT()
    {
        return cpTable;
    }
    /*
     The MinFill criterion is a heuristic for variable elimination.
     The variable that minimizes the number of edges that need to be added to the graph to make it triangulated.
     This is done by counting the number of edges that need to be added to the graph if the variable is eliminated.
     The variable with the minimum number of edges is chosen.
     Here this is done computing the length of the combinations of the node neighbors taken 2 by 2.
    */
    unsigned Node::minFill()
    {
        unordered_set<string> neighbors;
        for (auto child : children) {
            neighbors.emplace(child->getName());
        }
        for (auto parent : parents) {
            neighbors.emplace(parent->getName());
        }
        auto source = vector<string>(neighbors.begin(), neighbors.end());
        return combinations(source).size();
    }
    vector<pair<string, string>> Node::combinations(const vector<string>& source)
    {
        vector<pair<string, string>> result;
        for (int i = 0; i < source.size(); ++i) {
            string temp = source[i];
            for (int j = i + 1; j < source.size(); ++j) {
                result.push_back({ temp, source[j] });
            }
        }
        return result;
    }
    void Node::computeCPT(map<string, vector<int>>& dataset, const int laplaceSmoothing)
    {
        // Get dimensions of the CPT
        dimensions.push_back(numStates);
        for (auto father : getParents()) {
            dimensions.push_back(father->getNumStates());
        }
        auto length = dimensions.size();
        // Create a tensor of zeros with the dimensions of the CPT
        cpTable = torch::zeros(dimensions, torch::kFloat) + laplaceSmoothing;
        // Fill table with counts
        for (int n_sample = 0; n_sample < dataset[name].size(); ++n_sample) {
            torch::List<c10::optional<torch::Tensor>> coordinates;
            coordinates.push_back(torch::tensor(dataset[name][n_sample]));
            for (auto father : getParents()) {
                coordinates.push_back(torch::tensor(dataset[father->getName()][n_sample]));
            }
            // Increment the count of the corresponding coordinate
            cpTable.index_put_({ coordinates }, cpTable.index({ coordinates }) + 1);
        }
        // Normalize the counts
        cpTable = cpTable / cpTable.sum(0);
    }
    float Node::getFactorValue(map<string, int>& evidence)
    {
        torch::List<c10::optional<torch::Tensor>> coordinates;
        // following predetermined order of indices in the cpTable (see Node.h)
        coordinates.push_back(torch::tensor(evidence[name]));
        for (auto parent : getParents()) {
            coordinates.push_back(torch::tensor(evidence[parent->getName()]));
        }
        return cpTable.index({ coordinates }).item<float>();
    }
}