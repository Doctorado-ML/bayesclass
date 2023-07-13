#include <thread>
#include <mutex>
#include "Network.h"
namespace bayesnet {
    Network::Network() : laplaceSmoothing(1), features(vector<string>()), className(""), classNumStates(0), maxThreads(0.8) {}
    Network::Network(float maxT) : laplaceSmoothing(1), features(vector<string>()), className(""), classNumStates(0), maxThreads(maxT) {}
    Network::Network(float maxT, int smoothing) : laplaceSmoothing(smoothing), features(vector<string>()), className(""), classNumStates(0), maxThreads(maxT) {}
    Network::Network(Network& other) : laplaceSmoothing(other.laplaceSmoothing), features(other.features), className(other.className), classNumStates(other.getClassNumStates()), maxThreads(other.getmaxThreads())
    {
        for (auto& pair : other.nodes) {
            nodes[pair.first] = new Node(*pair.second);
        }
    }
    Network::~Network()
    {
        for (auto& pair : nodes) {
            delete pair.second;
        }
    }
    float Network::getmaxThreads()
    {
        return maxThreads;
    }
    torch::Tensor& Network::getSamples()
    {
        return samples;
    }
    void Network::addNode(string name, int numStates)
    {
        if (nodes.find(name) != nodes.end()) {
            // if node exists update its number of states
            nodes[name]->setNumStates(numStates);
            return;
        }
        nodes[name] = new Node(name, numStates);
    }
    vector<string> Network::getFeatures()
    {
        return features;
    }
    int Network::getClassNumStates()
    {
        return classNumStates;
    }
    int Network::getStates()
    {
        int result = 0;
        for (auto node : nodes) {
            result += node.second->getNumStates();
        }
        return result;
    }
    string Network::getClassName()
    {
        return className;
    }
    bool Network::isCyclic(const string& nodeId, unordered_set<string>& visited, unordered_set<string>& recStack)
    {
        if (visited.find(nodeId) == visited.end()) // if node hasn't been visited yet
        {
            visited.insert(nodeId);
            recStack.insert(nodeId);
            for (Node* child : nodes[nodeId]->getChildren()) {
                if (visited.find(child->getName()) == visited.end() && isCyclic(child->getName(), visited, recStack))
                    return true;
                else if (recStack.find(child->getName()) != recStack.end())
                    return true;
            }
        }
        recStack.erase(nodeId); // remove node from recursion stack before function ends
        return false;
    }
    void Network::addEdge(const string parent, const string child)
    {
        if (nodes.find(parent) == nodes.end()) {
            throw invalid_argument("Parent node " + parent + " does not exist");
        }
        if (nodes.find(child) == nodes.end()) {
            throw invalid_argument("Child node " + child + " does not exist");
        }
        // Temporarily add edge to check for cycles
        nodes[parent]->addChild(nodes[child]);
        nodes[child]->addParent(nodes[parent]);
        unordered_set<string> visited;
        unordered_set<string> recStack;
        if (isCyclic(nodes[child]->getName(), visited, recStack)) // if adding this edge forms a cycle
        {
            // remove problematic edge
            nodes[parent]->removeChild(nodes[child]);
            nodes[child]->removeParent(nodes[parent]);
            throw invalid_argument("Adding this edge forms a cycle in the graph.");
        }

    }
    map<string, Node*>& Network::getNodes()
    {
        return nodes;
    }
    void Network::fit(const vector<vector<int>>& input_data, const vector<int>& labels, const vector<string>& featureNames, const string& className)
    {
        features = featureNames;
        this->className = className;
        dataset.clear();

        // Build dataset & tensor of samples
        samples = torch::zeros({ static_cast<int64_t>(input_data[0].size()), static_cast<int64_t>(input_data.size() + 1) }, torch::kInt64);
        for (int i = 0; i < featureNames.size(); ++i) {
            dataset[featureNames[i]] = input_data[i];
            samples.index_put_({ "...", i }, torch::tensor(input_data[i], torch::kInt64));
        }
        dataset[className] = labels;
        samples.index_put_({ "...", -1 }, torch::tensor(labels, torch::kInt64));
        classNumStates = *max_element(labels.begin(), labels.end()) + 1;
        int maxThreadsRunning = static_cast<int>(std::thread::hardware_concurrency() * maxThreads);
        if (maxThreadsRunning < 1) {
            maxThreadsRunning = 1;
        }
        vector<thread> threads;
        mutex mtx;
        condition_variable cv;
        int activeThreads = 0;
        int nextNodeIndex = 0;

        while (nextNodeIndex < nodes.size()) {
            unique_lock<mutex> lock(mtx);
            cv.wait(lock, [&activeThreads, &maxThreadsRunning]() { return activeThreads < maxThreadsRunning; });

            if (nextNodeIndex >= nodes.size()) {
                break;  // No more work remaining
            }

            threads.emplace_back([this, &nextNodeIndex, &mtx, &cv, &activeThreads]() {
                while (true) {
                    unique_lock<mutex> lock(mtx);
                    if (nextNodeIndex >= nodes.size()) {
                        break;  // No more work remaining
                    }
                    auto& pair = *std::next(nodes.begin(), nextNodeIndex);
                    ++nextNodeIndex;
                    lock.unlock();

                    pair.second->computeCPT(dataset, laplaceSmoothing);

                    lock.lock();
                    nodes[pair.first] = pair.second;
                    lock.unlock();
                }
                lock_guard<mutex> lock(mtx);
                --activeThreads;
                cv.notify_one();
                });

            ++activeThreads;
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    vector<int> Network::predict(const vector<vector<int>>& tsamples)
    {
        vector<int> predictions;
        vector<int> sample;
        for (int row = 0; row < tsamples[0].size(); ++row) {
            sample.clear();
            for (int col = 0; col < tsamples.size(); ++col) {
                sample.push_back(tsamples[col][row]);
            }
            vector<double> classProbabilities = predict_sample(sample);
            // Find the class with the maximum posterior probability
            auto maxElem = max_element(classProbabilities.begin(), classProbabilities.end());
            int predictedClass = distance(classProbabilities.begin(), maxElem);
            predictions.push_back(predictedClass);
        }
        return predictions;
    }
    vector<vector<double>> Network::predict_proba(const vector<vector<int>>& tsamples)
    {
        vector<vector<double>> predictions;
        vector<int> sample;
        for (int row = 0; row < tsamples[0].size(); ++row) {
            sample.clear();
            for (int col = 0; col < tsamples.size(); ++col) {
                sample.push_back(tsamples[col][row]);
            }
            predictions.push_back(predict_sample(sample));
        }
        return predictions;
    }
    double Network::score(const vector<vector<int>>& tsamples, const vector<int>& labels)
    {
        vector<int> y_pred = predict(tsamples);
        int correct = 0;
        for (int i = 0; i < y_pred.size(); ++i) {
            if (y_pred[i] == labels[i]) {
                correct++;
            }
        }
        return (double)correct / y_pred.size();
    }
    vector<double> Network::predict_sample(const vector<int>& sample)
    {
        // Ensure the sample size is equal to the number of features
        if (sample.size() != features.size()) {
            throw invalid_argument("Sample size (" + to_string(sample.size()) +
                ") does not match the number of features (" + to_string(features.size()) + ")");
        }
        map<string, int> evidence;
        for (int i = 0; i < sample.size(); ++i) {
            evidence[features[i]] = sample[i];
        }
        return exactInference(evidence);

    }
    double Network::computeFactor(map<string, int>& completeEvidence)
    {
        double result = 1.0;
        for (auto node : getNodes()) {
            result *= node.second->getFactorValue(completeEvidence);
        }
        return result;
    }
    vector<double> Network::exactInference(map<string, int>& evidence)
    {
        vector<double> result(classNumStates, 0.0);
        vector<thread> threads;
        mutex mtx;
        for (int i = 0; i < classNumStates; ++i) {
            threads.emplace_back([this, &result, &evidence, i, &mtx]() {
                auto completeEvidence = map<string, int>(evidence);
                completeEvidence[getClassName()] = i;
                double factor = computeFactor(completeEvidence);
                lock_guard<mutex> lock(mtx);
                result[i] = factor;
                });
        }
        for (auto& thread : threads) {
            thread.join();
        }

        // Normalize result
        double sum = accumulate(result.begin(), result.end(), 0.0);
        for (double& value : result) {
            value /= sum;
        }
        return result;
    }
    void Network::show()
    {
        // Draw the network
        for (auto node : nodes) {
            cout << node.first << " -> ";
            for (auto child : node.second->getChildren()) {
                cout << child->getName() << ", ";
            }
            cout << endl;
        }
    }

}
