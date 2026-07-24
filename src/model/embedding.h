#pragma once
#include "core/tensor.h"
#include "model/config.h"

namespace nf {

class Embedding {
public:
    Tensor weight; // [vocab_size x d_model]
    Tensor weight_grad;

    Embedding() = default;
    void init(int vocab_size, int d_model);

    // Forward: indices [batch x seq_len] as flat int vector -> [batch*seq_len x d_model]
    Tensor forward(const std::vector<int>& indices, int seq_len) const;

    // Backward: accumulate gradients
    void backward(const std::vector<int>& indices, int seq_len, const Tensor& grad_output);

    void zero_grad();
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
};

class PositionalEncoding {
public:
    Tensor encoding; // [max_seq_len x d_model]

    PositionalEncoding() = default;
    void init(int max_seq_len, int d_model);

    // Add positional encoding to input
    Tensor forward(const Tensor& x, int seq_len) const;
};

} // namespace nf
