#include "model/attention.h"
#include <cmath>
#include <limits>

namespace nf {

void MultiHeadAttention::init(int d_model_, int n_heads_) {
    d_model = d_model_;
    n_heads = n_heads_;
    head_dim = d_model / n_heads;

    float std = 0.02f;
    Wq = Tensor::randn(d_model, d_model, 0.0f, std, true);
    Wk = Tensor::randn(d_model, d_model, 0.0f, std, true);
    Wv = Tensor::randn(d_model, d_model, 0.0f, std, true);
    Wo = Tensor::randn(d_model, d_model, 0.0f, std, true);

    Wq_grad = Tensor::zeros(d_model, d_model);
    Wk_grad = Tensor::zeros(d_model, d_model);
    Wv_grad = Tensor::zeros(d_model, d_model);
    Wo_grad = Tensor::zeros(d_model, d_model);
}

Tensor MultiHeadAttention::forward(const Tensor& x, int seq_len, bool use_causal_mask) {
    cached_input = x.clone();
    int total_tokens = static_cast<int>(x.rows);
    int batch_size = total_tokens / seq_len;

    // Project Q, K, V: [total_tokens x d_model]
    cached_q = x.matmul(Wq);
    cached_k = x.matmul(Wk);
    cached_v = x.matmul(Wv);

    // Output accumulator
    Tensor head_concat(total_tokens, d_model);

    // Process each batch and head
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    cached_attn_weights = Tensor(batch_size * n_heads * seq_len, seq_len);

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {
            int h_offset = h * head_dim;

            // Extract this head's Q, K, V for this batch: [seq_len x head_dim]
            Tensor q_h(seq_len, head_dim);
            Tensor k_h(seq_len, head_dim);
            Tensor v_h(seq_len, head_dim);

            for (int t = 0; t < seq_len; t++) {
                int row = b * seq_len + t;
                for (int d = 0; d < head_dim; d++) {
                    q_h.data[t * head_dim + d] = cached_q.data[row * d_model + h_offset + d];
                    k_h.data[t * head_dim + d] = cached_k.data[row * d_model + h_offset + d];
                    v_h.data[t * head_dim + d] = cached_v.data[row * d_model + h_offset + d];
                }
            }

            // Attention scores: Q * K^T / sqrt(d_k) -> [seq_len x seq_len]
            Tensor scores = q_h.matmul(k_h.transpose()) * scale;

            // Causal mask
            if (use_causal_mask) {
                for (int i = 0; i < seq_len; i++) {
                    for (int j = i + 1; j < seq_len; j++) {
                        scores.data[i * seq_len + j] = -1e9f;
                    }
                }
            }

            // Softmax
            Tensor attn = scores.softmax(-1);

            // Store attention weights for backward
            int aw_offset = (b * n_heads + h) * seq_len;
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    cached_attn_weights.data[(aw_offset + i) * seq_len + j] = attn.data[i * seq_len + j];
                }
            }

            // Weighted sum: attn * V -> [seq_len x head_dim]
            Tensor head_out = attn.matmul(v_h);

            // Copy into head_concat
            for (int t = 0; t < seq_len; t++) {
                int row = b * seq_len + t;
                for (int d = 0; d < head_dim; d++) {
                    head_concat.data[row * d_model + h_offset + d] = head_out.data[t * head_dim + d];
                }
            }
        }
    }

    // Output projection
    return head_concat.matmul(Wo);
}

Tensor MultiHeadAttention::backward(const Tensor& grad_output, int seq_len) {
    int total_tokens = static_cast<int>(grad_output.rows);
    int batch_size = total_tokens / seq_len;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // grad through Wo: grad_head_concat = grad_output * Wo^T
    Tensor grad_head_concat = grad_output.matmul(Wo.transpose());

    // Wo_grad += head_concat^T * grad_output (reconstruct head_concat from cached Q,K,V)
    // We need to redo the forward for head_concat — or we store it. Let's recompute.
    // Actually, let's accumulate Wo_grad properly.

    // For Wo_grad, we need the head_concat. Let's reconstruct it.
    Tensor head_concat(total_tokens, d_model);
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {
            int h_offset = h * head_dim;
            int aw_offset = (b * n_heads + h) * seq_len;

            Tensor v_h(seq_len, head_dim);
            for (int t = 0; t < seq_len; t++) {
                int row = b * seq_len + t;
                for (int d = 0; d < head_dim; d++) {
                    v_h.data[t * head_dim + d] = cached_v.data[row * d_model + h_offset + d];
                }
            }

            Tensor attn(seq_len, seq_len);
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    attn.data[i * seq_len + j] = cached_attn_weights.data[(aw_offset + i) * seq_len + j];
                }
            }

            Tensor head_out = attn.matmul(v_h);
            for (int t = 0; t < seq_len; t++) {
                int row = b * seq_len + t;
                for (int d = 0; d < head_dim; d++) {
                    head_concat.data[row * d_model + h_offset + d] = head_out.data[t * head_dim + d];
                }
            }
        }
    }

    // Wo_grad
    Tensor hc_t = head_concat.transpose();
    Tensor wo_g = hc_t.matmul(grad_output);
    for (size_t i = 0; i < Wo_grad.data.size(); i++) Wo_grad.data[i] += wo_g.data[i];

    // Grad through each head
    Tensor grad_q_full = Tensor::zeros(total_tokens, d_model);
    Tensor grad_k_full = Tensor::zeros(total_tokens, d_model);
    Tensor grad_v_full = Tensor::zeros(total_tokens, d_model);

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {
            int h_offset = h * head_dim;
            int aw_offset = (b * n_heads + h) * seq_len;

            // Extract per-head grad
            Tensor grad_hout(seq_len, head_dim);
            for (int t = 0; t < seq_len; t++) {
                int row = b * seq_len + t;
                for (int d = 0; d < head_dim; d++) {
                    grad_hout.data[t * head_dim + d] = grad_head_concat.data[row * d_model + h_offset + d];
                }
            }

            // Extract cached values
            Tensor q_h(seq_len, head_dim), k_h(seq_len, head_dim), v_h(seq_len, head_dim);
            for (int t = 0; t < seq_len; t++) {
                int row = b * seq_len + t;
                for (int d = 0; d < head_dim; d++) {
                    q_h.data[t * head_dim + d] = cached_q.data[row * d_model + h_offset + d];
                    k_h.data[t * head_dim + d] = cached_k.data[row * d_model + h_offset + d];
                    v_h.data[t * head_dim + d] = cached_v.data[row * d_model + h_offset + d];
                }
            }

            Tensor attn(seq_len, seq_len);
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    attn.data[i * seq_len + j] = cached_attn_weights.data[(aw_offset + i) * seq_len + j];
                }
            }

            // grad_v_h = attn^T * grad_hout
            Tensor grad_v_h = attn.transpose().matmul(grad_hout);

            // grad_attn = grad_hout * v_h^T
            Tensor grad_attn = grad_hout.matmul(v_h.transpose());

            // Softmax backward: grad_scores = attn * (grad_attn - sum(grad_attn * attn, axis=-1))
            Tensor grad_scores(seq_len, seq_len);
            for (int i = 0; i < seq_len; i++) {
                float dot = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    dot += grad_attn.data[i * seq_len + j] * attn.data[i * seq_len + j];
                }
                for (int j = 0; j < seq_len; j++) {
                    grad_scores.data[i * seq_len + j] =
                        attn.data[i * seq_len + j] * (grad_attn.data[i * seq_len + j] - dot);
                }
            }
            grad_scores = grad_scores * scale;

            // grad_q_h = grad_scores * k_h
            Tensor grad_q_h = grad_scores.matmul(k_h);
            // grad_k_h = grad_scores^T * q_h
            Tensor grad_k_h = grad_scores.transpose().matmul(q_h);

            // Scatter back
            for (int t = 0; t < seq_len; t++) {
                int row = b * seq_len + t;
                for (int d = 0; d < head_dim; d++) {
                    grad_q_full.data[row * d_model + h_offset + d] += grad_q_h.data[t * head_dim + d];
                    grad_k_full.data[row * d_model + h_offset + d] += grad_k_h.data[t * head_dim + d];
                    grad_v_full.data[row * d_model + h_offset + d] += grad_v_h.data[t * head_dim + d];
                }
            }
        }
    }

    // Weight gradients for Wq, Wk, Wv
    Tensor inp_t = cached_input.transpose();
    Tensor wq_g = inp_t.matmul(grad_q_full);
    Tensor wk_g = inp_t.matmul(grad_k_full);
    Tensor wv_g = inp_t.matmul(grad_v_full);
    for (size_t i = 0; i < Wq_grad.data.size(); i++) {
        Wq_grad.data[i] += wq_g.data[i];
        Wk_grad.data[i] += wk_g.data[i];
        Wv_grad.data[i] += wv_g.data[i];
    }

    // Input gradient
    Tensor grad_input = grad_q_full.matmul(Wq.transpose())
                      + grad_k_full.matmul(Wk.transpose())
                      + grad_v_full.matmul(Wv.transpose());

    return grad_input;
}

void MultiHeadAttention::zero_grad() {
    Wq_grad.fill(0.0f);
    Wk_grad.fill(0.0f);
    Wv_grad.fill(0.0f);
    Wo_grad.fill(0.0f);
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    return {&Wq, &Wk, &Wv, &Wo};
}

std::vector<Tensor*> MultiHeadAttention::gradients() {
    return {&Wq_grad, &Wk_grad, &Wv_grad, &Wo_grad};
}

} // namespace nf
