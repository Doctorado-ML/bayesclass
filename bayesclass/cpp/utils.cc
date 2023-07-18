#include <torch/torch.h>
#include <vector>
namespace bayesnet {
    using namespace std;
    using namespace torch;
    vector<int> argsort(vector<float>& nums)
    {
        int n = nums.size();
        vector<int> indices(n);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&nums](int i, int j) {return nums[i] > nums[j];});
        return indices;
    }
    vector<vector<int>> tensorToVector(const Tensor& tensor)
    {
        // convert mxn tensor to nxm vector
        vector<vector<int>> result;
        auto tensor_accessor = tensor.accessor<int, 2>();

        // Iterate over columns and rows of the tensor
        for (int j = 0; j < tensor.size(1); ++j) {
            vector<int> column;
            for (int i = 0; i < tensor.size(0); ++i) {
                column.push_back(tensor_accessor[i][j]);
            }
            result.push_back(column);
        }

        return result;
    }
}