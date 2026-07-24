#pragma once
#include "model/config.h"
#include "model/embedding.h"
#include "model/attention.h"
#include "model/feedforward.h"
#include "model/lm_head.h"
#include "core/tensor.h"
#include <vector>

namespace nf {

struct LayerNormParams {
    Tensor gamma; // [1 x d_model]
    Tensor beta;
    Tensor gamma_grad, beta_grad;
    Tensor cached_input, cached_norm;

    void init(int d_model);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);
    void zero_grad();
};

struct TransformerBlock {
    LayerNormParams ln1, ln2;
    MultiHeadAttention attn;
    FeedForward ff;
    Tensor cached_residual1, cached_residual2;

    void init(const ModelConfig& config);
    Tensor forward(const Tensor& x, int seq_len);
    Tensor backward(const Tensor& grad_output, int seq_len);
    void zero_grad();
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
};

class Transformer {
public:
    ModelConfig config;
    Embedding token_emb;
    PositionalEncoding pos_enc;
    std::vector<TransformerBlock> layers;
    LayerNormParams final_ln;
    LMHead lm_head;

    // Cached
    std::vector<int> cached_input_ids;
    int cached_seq_len = 0;

    Transformer() = default;
    void init(const ModelConfig& config);

    // Forward: token IDs -> logits [batch*seq_len x vocab_size]
    Tensor forward(const std::vector<int>& input_ids, int seq_len);

    // Backward from logits gradient
    void backward(const Tensor& grad_logits);

    void zero_grad();
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    size_t param_count() const;
};

} // namespace nf
