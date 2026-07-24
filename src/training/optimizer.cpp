#include "training/optimizer.h"
#include <cmath>

namespace nf {

AdamOptimizer::AdamOptimizer(float lr_, float beta1_, float beta2_, float eps_, float wd_)
    : lr(lr_), beta1(beta1_), beta2(beta2_), epsilon(eps_), weight_decay(wd_) {}

void AdamOptimizer::init(const std::vector<Tensor*>& params) {
    m.clear();
    v.clear();
    for (auto* p : params) {
        m.push_back(Tensor::zeros(p->rows, p->cols));
        v.push_back(Tensor::zeros(p->rows, p->cols));
    }
    step_count = 0;
}

void AdamOptimizer::step(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) {
    step_count++;
    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step_count));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step_count));

    for (size_t i = 0; i < params.size(); i++) {
        Tensor* p = params[i];
        Tensor* g = grads[i];

        for (size_t j = 0; j < p->data.size(); j++) {
            float grad = g->data[j];

            // Weight decay
            if (weight_decay > 0.0f) {
                grad += weight_decay * p->data[j];
            }

            // Update moments
            m[i].data[j] = beta1 * m[i].data[j] + (1.0f - beta1) * grad;
            v[i].data[j] = beta2 * v[i].data[j] + (1.0f - beta2) * grad * grad;

            // Bias correction
            float m_hat = m[i].data[j] / bc1;
            float v_hat = v[i].data[j] / bc2;

            // Update
            p->data[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
}

} // namespace nf
