#include "model/embedding.h"
#include <cmath>

namespace nf {

void Embedding::init(int vocab_size, int d_model) {
    float std = 0.02f;
    weight = Tensor::randn(vocab_size, d_model, 0.0f, std, true);
    weight_grad = Tensor::zeros(vocab_size, d_model);
}

Tensor Embedding::forward(const std::vector<int>& indices, int seq_len) const {
    int batch_tokens = static_cast<int>(indices.size());
    Tensor output(batch_tokens, weight.cols);
    for (int i = 0; i < batch_tokens; i++) {
        int idx = indices[i];
        if (idx >= 0 && idx < static_cast<int>(weight.rows)) {
            for (size_t j = 0; j < weight.cols; j++) {
                output.data[i * weight.cols + j] = weight.data[idx * weight.cols + j];
            }
        }
    }
    return output;
}

void Embedding::backward(const std::vector<int>& indices, int seq_len, const Tensor& grad_output) {
    int batch_tokens = static_cast<int>(indices.size());
    for (int i = 0; i < batch_tokens; i++) {
        int idx = indices[i];
        if (idx >= 0 && idx < static_cast<int>(weight.rows)) {
            for (size_t j = 0; j < weight.cols; j++) {
                weight_grad.data[idx * weight.cols + j] += grad_output.data[i * weight.cols + j];
            }
        }
    }
}

void Embedding::zero_grad() {
    weight_grad.fill(0.0f);
}

std::vector<Tensor*> Embedding::parameters() { return {&weight}; }
std::vector<Tensor*> Embedding::gradients() { return {&weight_grad}; }

void PositionalEncoding::init(int max_seq_len, int d_model) {
    encoding = Tensor(max_seq_len, d_model);
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            float angle = static_cast<float>(pos) / std::pow(10000.0f, (2.0f * (i / 2)) / d_model);
            encoding.data[pos * d_model + i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
}

Tensor PositionalEncoding::forward(const Tensor& x, int seq_len) const {
    Tensor result = x.clone();
    int batch_size = static_cast<int>(x.rows) / seq_len;
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < x.cols; d++) {
                result.data[(b * seq_len + t) * x.cols + d] += encoding.data[t * x.cols + d];
            }
        }
    }
    return result;
}

} // namespace nf
