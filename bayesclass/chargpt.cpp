double conditionalEntropy(std::vector<int>& classVec, std::vector<int>& featureVec, std::vector<double>& weightsVec)
{
    std::map<int, std::vector<int>> classesPerFeatureValue;
    std::map<int, std::vector<double>> weightsPerFeatureValue;

    for (int i = 0; i < featureVec.size(); i++) {
        classesPerFeatureValue[featureVec[i]].push_back(classVec[i]);
        weightsPerFeatureValue[featureVec[i]].push_back(weightsVec[i]);
    }

    double totalEntropy = 0;
    double totalWeight = 0;
    for (auto& pair : classesPerFeatureValue) {
        double featureValueEntropy = calculateEntropy(pair.second, weightsPerFeatureValue[pair.first]);
        double featureValueWeight = 0;
        for (double weight : weightsPerFeatureValue[pair.first]) {
            featureValueWeight += weight;
        }
        totalEntropy += featureValueWeight * featureValueEntropy;
        totalWeight += featureValueWeight;
    }

    return totalEntropy / totalWeight;
}