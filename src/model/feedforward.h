#pragma once
#include "core/tensor.h"

namespace nf {

class FeedForward {
public:
    Tensor W1, W2;       // W1: [d_model x d_ff], W2: [d_ff x d_model]
    Tensor b1, b2;       // biases
    Tensor W1_grad, W2_grad, b1_grad, b2_grad;

    // Cached
    Tensor cached_input, cached_hidden;

    FeedForward() = default;
    void init(int d_model, int d_ff);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);

    void zero_grad();
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
};

} // namespace nf
