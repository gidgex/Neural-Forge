#pragma once
#include "core/tensor.h"
#include "model/config.h"

namespace nf {

class MultiHeadAttention {
public:
    // Weight matrices
    Tensor Wq, Wk, Wv, Wo;        // [d_model x d_model]
    Tensor Wq_grad, Wk_grad, Wv_grad, Wo_grad;

    // Cached for backward pass
    Tensor cached_q, cached_k, cached_v;
    Tensor cached_attn_weights;
    Tensor cached_input;

    int n_heads = 0;
    int d_model = 0;
    int head_dim = 0;

    MultiHeadAttention() = default;
    void init(int d_model, int n_heads);

    // Forward: x [seq_len x d_model] -> [seq_len x d_model]
    Tensor forward(const Tensor& x, int seq_len, bool use_causal_mask = true);

    // Backward: grad_output [seq_len x d_model] -> grad_input [seq_len x d_model]
    Tensor backward(const Tensor& grad_output, int seq_len);

    void zero_grad();
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
};

} // namespace nf
