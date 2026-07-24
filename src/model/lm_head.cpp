#include "model/lm_head.h"

namespace nf {

void LMHead::init(int d_model, int vocab_size) {
    float std = 0.02f;
    weight = Tensor::randn(d_model, vocab_size, 0.0f, std, true);
    bias = Tensor::zeros(1, vocab_size);
    weight_grad = Tensor::zeros(d_model, vocab_size);
    bias_grad = Tensor::zeros(1, vocab_size);
}

Tensor LMHead::forward(const Tensor& x) {
    cached_input = x.clone();
    Tensor logits = x.matmul(weight);
    for (size_t i = 0; i < logits.rows; i++) {
        for (size_t j = 0; j < logits.cols; j++) {
            logits.data[i * logits.cols + j] += bias.data[j];
        }
    }
    return logits;
}

Tensor LMHead::backward(const Tensor& grad_logits) {
    size_t n = grad_logits.rows;

    // bias_grad
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < bias_grad.cols; j++) {
            bias_grad.data[j] += grad_logits.data[i * grad_logits.cols + j];
        }
    }

    // weight_grad += input^T * grad_logits
    Tensor inp_t = cached_input.transpose();
    Tensor wg = inp_t.matmul(grad_logits);
    for (size_t i = 0; i < weight_grad.data.size(); i++) weight_grad.data[i] += wg.data[i];

    // grad_input = grad_logits * weight^T
    return grad_logits.matmul(weight.transpose());
}

void LMHead::zero_grad() {
    weight_grad.fill(0.0f);
    bias_grad.fill(0.0f);
}

std::vector<Tensor*> LMHead::parameters() { return {&weight, &bias}; }
std::vector<Tensor*> LMHead::gradients() { return {&weight_grad, &bias_grad}; }

} // namespace nf
