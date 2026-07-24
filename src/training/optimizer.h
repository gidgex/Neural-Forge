#pragma once
#include "core/tensor.h"
#include <vector>

namespace nf {

class AdamOptimizer {
public:
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.0f;
    int step_count = 0;

    std::vector<Tensor> m; // First moment
    std::vector<Tensor> v; // Second moment

    AdamOptimizer() = default;
    AdamOptimizer(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float wd = 0.01f);

    void init(const std::vector<Tensor*>& params);
    void step(std::vector<Tensor*>& params, std::vector<Tensor*>& grads);
};

} // namespace nf
