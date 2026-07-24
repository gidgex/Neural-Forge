#include "model/feedforward.h"
#include <cmath>

namespace nf {

void FeedForward::init(int d_model, int d_ff) {
    float std = 0.02f;
    W1 = Tensor::randn(d_model, d_ff, 0.0f, std, true);
    W2 = Tensor::randn(d_ff, d_model, 0.0f, std, true);
    b1 = Tensor::zeros(1, d_ff);
    b2 = Tensor::zeros(1, d_model);

    W1_grad = Tensor::zeros(d_model, d_ff);
    W2_grad = Tensor::zeros(d_ff, d_model);
    b1_grad = Tensor::zeros(1, d_ff);
    b2_grad = Tensor::zeros(1, d_model);
}

Tensor FeedForward::forward(const Tensor& x) {
    cached_input = x.clone();

    // hidden = GELU(x * W1 + b1)
    Tensor proj = x.matmul(W1);
    // Add bias (broadcast across rows)
    for (size_t i = 0; i < proj.rows; i++) {
        for (size_t j = 0; j < proj.cols; j++) {
            proj.data[i * proj.cols + j] += b1.data[j];
        }
    }
    cached_hidden = proj.gelu();

    // output = hidden * W2 + b2
    Tensor output = cached_hidden.matmul(W2);
    for (size_t i = 0; i < output.rows; i++) {
        for (size_t j = 0; j < output.cols; j++) {
            output.data[i * output.cols + j] += b2.data[j];
        }
    }
    return output;
}

Tensor FeedForward::backward(const Tensor& grad_output) {
    size_t n = grad_output.rows;

    // Grad for b2: sum over batch
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < b2_grad.cols; j++) {
            b2_grad.data[j] += grad_output.data[i * grad_output.cols + j];
        }
    }

    // W2_grad += cached_hidden^T * grad_output
    Tensor ch_t = cached_hidden.transpose();
    Tensor w2g = ch_t.matmul(grad_output);
    for (size_t i = 0; i < W2_grad.data.size(); i++) W2_grad.data[i] += w2g.data[i];

    // grad_hidden = grad_output * W2^T
    Tensor grad_hidden = grad_output.matmul(W2.transpose());

    // GELU backward: d/dx[x * Phi(x)] = Phi(x) + x * phi(x)
    // where Phi = CDF, phi = PDF of standard normal
    // Using tanh approximation derivative
    Tensor grad_pre_gelu(n, cached_hidden.cols);
    Tensor pre_gelu = cached_input.matmul(W1);
    for (size_t i = 0; i < pre_gelu.rows; i++) {
        for (size_t j = 0; j < pre_gelu.cols; j++) {
            pre_gelu.data[i * pre_gelu.cols + j] += b1.data[j];
        }
    }

    const float sqrt_2_over_pi = 0.7978845608f;
    for (size_t i = 0; i < grad_pre_gelu.data.size(); i++) {
        float x = pre_gelu.data[i];
        float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        float tanh_val = std::tanh(inner);
        float cdf = 0.5f * (1.0f + tanh_val);
        float pdf_term = 0.5f * sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x) * (1.0f - tanh_val * tanh_val);
        float gelu_grad = cdf + x * pdf_term;
        grad_pre_gelu.data[i] = grad_hidden.data[i] * gelu_grad;
    }

    // b1_grad
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < b1_grad.cols; j++) {
            b1_grad.data[j] += grad_pre_gelu.data[i * grad_pre_gelu.cols + j];
        }
    }

    // W1_grad += input^T * grad_pre_gelu
    Tensor inp_t = cached_input.transpose();
    Tensor w1g = inp_t.matmul(grad_pre_gelu);
    for (size_t i = 0; i < W1_grad.data.size(); i++) W1_grad.data[i] += w1g.data[i];

    // grad_input = grad_pre_gelu * W1^T
    return grad_pre_gelu.matmul(W1.transpose());
}

void FeedForward::zero_grad() {
    W1_grad.fill(0.0f);
    W2_grad.fill(0.0f);
    b1_grad.fill(0.0f);
    b2_grad.fill(0.0f);
}

std::vector<Tensor*> FeedForward::parameters() {
    return {&W1, &W2, &b1, &b2};
}

std::vector<Tensor*> FeedForward::gradients() {
    return {&W1_grad, &W2_grad, &b1_grad, &b2_grad};
}

} // namespace nf
