#include "model/transformer.h"
#include <cmath>
#include <iostream>

namespace nf {

// ---- LayerNormParams ----

void LayerNormParams::init(int d_model) {
    gamma = Tensor::ones(1, d_model);
    beta = Tensor::zeros(1, d_model);
    gamma_grad = Tensor::zeros(1, d_model);
    beta_grad = Tensor::zeros(1, d_model);
}

Tensor LayerNormParams::forward(const Tensor& x) {
    cached_input = x.clone();

    Tensor norm = x.layer_norm();
    cached_norm = norm.clone();

    // Scale and shift: gamma * norm + beta
    Tensor result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; i++) {
        for (size_t j = 0; j < x.cols; j++) {
            result.data[i * x.cols + j] = gamma.data[j] * norm.data[i * x.cols + j] + beta.data[j];
        }
    }
    return result;
}

Tensor LayerNormParams::backward(const Tensor& grad_output) {
    size_t n = grad_output.rows;
    size_t d = grad_output.cols;

    // gamma_grad, beta_grad
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            gamma_grad.data[j] += grad_output.data[i * d + j] * cached_norm.data[i * d + j];
            beta_grad.data[j] += grad_output.data[i * d + j];
        }
    }

    // grad through layer norm (simplified)
    // d_norm = grad_output * gamma
    Tensor d_norm(n, d);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            d_norm.data[i * d + j] = grad_output.data[i * d + j] * gamma.data[j];
        }
    }

    // Layer norm backward
    Tensor grad_input(n, d);
    float inv_d = 1.0f / static_cast<float>(d);

    for (size_t i = 0; i < n; i++) {
        // Compute mean and var of input row
        float mean = 0.0f;
        for (size_t j = 0; j < d; j++) mean += cached_input.data[i * d + j];
        mean *= inv_d;

        float var = 0.0f;
        for (size_t j = 0; j < d; j++) {
            float diff = cached_input.data[i * d + j] - mean;
            var += diff * diff;
        }
        var *= inv_d;
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);

        // Compute dot products needed
        float dot_dn_norm = 0.0f;
        float dot_dn = 0.0f;
        for (size_t j = 0; j < d; j++) {
            dot_dn_norm += d_norm.data[i * d + j] * cached_norm.data[i * d + j];
            dot_dn += d_norm.data[i * d + j];
        }

        for (size_t j = 0; j < d; j++) {
            grad_input.data[i * d + j] = inv_std * inv_d *
                (d * d_norm.data[i * d + j] - dot_dn - cached_norm.data[i * d + j] * dot_dn_norm);
        }
    }

    return grad_input;
}

void LayerNormParams::zero_grad() {
    gamma_grad.fill(0.0f);
    beta_grad.fill(0.0f);
}

// ---- TransformerBlock ----

void TransformerBlock::init(const ModelConfig& config) {
    ln1.init(config.d_model);
    ln2.init(config.d_model);
    attn.init(config.d_model, config.n_heads);
    ff.init(config.d_model, config.d_ff);
}

Tensor TransformerBlock::forward(const Tensor& x, int seq_len) {
    // Pre-norm architecture
    // x1 = x + attn(ln1(x))
    cached_residual1 = x.clone();
    Tensor ln1_out = ln1.forward(x);
    Tensor attn_out = attn.forward(ln1_out, seq_len);
    Tensor x1 = x + attn_out;

    // x2 = x1 + ff(ln2(x1))
    cached_residual2 = x1.clone();
    Tensor ln2_out = ln2.forward(x1);
    Tensor ff_out = ff.forward(ln2_out);
    return x1 + ff_out;
}

Tensor TransformerBlock::backward(const Tensor& grad_output, int seq_len) {
    // Backward through: output = x1 + ff(ln2(x1))
    // grad_x1 = grad_output + grad through ff path
    Tensor grad_ff = ff.backward(grad_output);
    Tensor grad_ln2 = ln2.backward(grad_ff);
    Tensor grad_x1 = grad_output + grad_ln2;

    // Backward through: x1 = x + attn(ln1(x))
    Tensor grad_attn = attn.backward(grad_x1, seq_len);
    Tensor grad_ln1 = ln1.backward(grad_attn);
    Tensor grad_x = grad_x1 + grad_ln1;

    return grad_x;
}

void TransformerBlock::zero_grad() {
    ln1.zero_grad();
    ln2.zero_grad();
    attn.zero_grad();
    ff.zero_grad();
}

std::vector<Tensor*> TransformerBlock::parameters() {
    std::vector<Tensor*> params;
    auto attn_p = attn.parameters();
    auto ff_p = ff.parameters();
    params.insert(params.end(), attn_p.begin(), attn_p.end());
    params.insert(params.end(), ff_p.begin(), ff_p.end());
    params.push_back(&ln1.gamma);
    params.push_back(&ln1.beta);
    params.push_back(&ln2.gamma);
    params.push_back(&ln2.beta);
    return params;
}

std::vector<Tensor*> TransformerBlock::gradients() {
    std::vector<Tensor*> grads;
    auto attn_g = attn.gradients();
    auto ff_g = ff.gradients();
    grads.insert(grads.end(), attn_g.begin(), attn_g.end());
    grads.insert(grads.end(), ff_g.begin(), ff_g.end());
    grads.push_back(&ln1.gamma_grad);
    grads.push_back(&ln1.beta_grad);
    grads.push_back(&ln2.gamma_grad);
    grads.push_back(&ln2.beta_grad);
    return grads;
}

// ---- Transformer ----

void Transformer::init(const ModelConfig& cfg) {
    config = cfg;
    token_emb.init(config.vocab_size, config.d_model);
    pos_enc.init(config.max_seq_len, config.d_model);

    layers.resize(config.n_layers);
    for (auto& layer : layers) {
        layer.init(config);
    }

    final_ln.init(config.d_model);
    lm_head.init(config.d_model, config.vocab_size);

    std::cout << "[Model] Transformer initialized: "
              << config.n_layers << " layers, "
              << config.d_model << " dim, "
              << config.n_heads << " heads, "
              << config.d_ff << " ff_dim, "
              << config.vocab_size << " vocab\n";
    std::cout << "[Model] Parameters: " << param_count() << "\n";
}

Tensor Transformer::forward(const std::vector<int>& input_ids, int seq_len) {
    cached_input_ids = input_ids;
    cached_seq_len = seq_len;

    // Token embeddings
    Tensor x = token_emb.forward(input_ids, seq_len);

    // Add positional encoding
    x = pos_enc.forward(x, seq_len);

    // Transformer layers
    for (auto& layer : layers) {
        x = layer.forward(x, seq_len);
    }

    // Final layer norm
    x = final_ln.forward(x);

    // LM head -> logits
    return lm_head.forward(x);
}

void Transformer::backward(const Tensor& grad_logits) {
    // Backward through LM head
    Tensor grad = lm_head.backward(grad_logits);

    // Final layer norm
    grad = final_ln.backward(grad);

    // Transformer layers (reverse order)
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
        grad = layers[i].backward(grad, cached_seq_len);
    }

    // Embedding backward
    token_emb.backward(cached_input_ids, cached_seq_len, grad);
}

void Transformer::zero_grad() {
    token_emb.zero_grad();
    for (auto& layer : layers) layer.zero_grad();
    final_ln.zero_grad();
    lm_head.zero_grad();
}

std::vector<Tensor*> Transformer::parameters() {
    std::vector<Tensor*> params;
    auto emb_p = token_emb.parameters();
    params.insert(params.end(), emb_p.begin(), emb_p.end());

    for (auto& layer : layers) {
        auto lp = layer.parameters();
        params.insert(params.end(), lp.begin(), lp.end());
    }

    params.push_back(&final_ln.gamma);
    params.push_back(&final_ln.beta);

    auto lm_p = lm_head.parameters();
    params.insert(params.end(), lm_p.begin(), lm_p.end());
    return params;
}

std::vector<Tensor*> Transformer::gradients() {
    std::vector<Tensor*> grads;
    auto emb_g = token_emb.gradients();
    grads.insert(grads.end(), emb_g.begin(), emb_g.end());

    for (auto& layer : layers) {
        auto lg = layer.gradients();
        grads.insert(grads.end(), lg.begin(), lg.end());
    }

    grads.push_back(&final_ln.gamma_grad);
    grads.push_back(&final_ln.beta_grad);

    auto lm_g = lm_head.gradients();
    grads.insert(grads.end(), lm_g.begin(), lm_g.end());
    return grads;
}

size_t Transformer::param_count() const {
    size_t count = 0;
    // Embedding
    count += config.vocab_size * config.d_model;
    // Each layer: 4 attention matrices + 2 FF matrices + 2 biases + 4 LN params
    for (int i = 0; i < config.n_layers; i++) {
        count += 4 * config.d_model * config.d_model;   // Wq, Wk, Wv, Wo
        count += config.d_model * config.d_ff;           // W1
        count += config.d_ff * config.d_model;           // W2
        count += config.d_ff + config.d_model;           // biases
        count += 4 * config.d_model;                     // LN gamma/beta x2
    }
    count += 2 * config.d_model; // final LN
    count += config.d_model * config.vocab_size + config.vocab_size; // LM head
    return count;
}

} // namespace nf
