#ifndef MST_H
#define MST_H
#include <torch/torch.h>
#include <vector>
#include <string>
namespace bayesnet {
    using namespace std;
    using namespace torch;
    class MST {
    private:
        Tensor weights;
        vector<string> features;
        int root;
    public:
        MST() = default;
        MST(vector<string>& features, Tensor& weights, int root);
        vector<pair<int, int>> maximumSpanningTree();
    };
    class Graph {
    private:
        int V;      // number of nodes in graph
        vector <pair<float, pair<int, int>>> G; // vector for graph
        vector <pair<float, pair<int, int>>> T; // vector for mst
        vector<int> parent;
    public:
        Graph(int V);
        void addEdge(int u, int v, float wt);
        int find_set(int i);
        void union_set(int u, int v);
        void kruskal_algorithm();
        void display_mst();
        vector <pair<float, pair<int, int>>> get_mst() { return T; }
    };
}
#endif