#pragma once
#include "core/tensor.h"

namespace nf {

class LMHead {
public:
    Tensor weight;     // [d_model x vocab_size]
    Tensor bias;       // [1 x vocab_size]
    Tensor weight_grad, bias_grad;
    Tensor cached_input;

    LMHead() = default;
    void init(int d_model, int vocab_size);

    // Forward: [n x d_model] -> [n x vocab_size] (logits)
    Tensor forward(const Tensor& x);

    // Backward: grad_logits [n x vocab_size] -> grad_input [n x d_model]
    Tensor backward(const Tensor& grad_logits);

    void zero_grad();
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
};

} // namespace nf
