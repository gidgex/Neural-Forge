#pragma once
#include "core/tensor.h"
#include <vector>

namespace nf {

struct SamplerConfig {
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.9f;
    float repetition_penalty = 1.1f;
};

class Sampler {
public:
    SamplerConfig config;

    Sampler() = default;
    Sampler(const SamplerConfig& cfg) : config(cfg) {}

    // Sample a token ID from logits [1 x vocab_size]
    int sample(const Tensor& logits, const std::vector<int>& recent_tokens = {}) const;

    // Greedy: just take argmax
    int greedy(const Tensor& logits) const;

private:
    void apply_temperature(std::vector<float>& logits) const;
    void apply_repetition_penalty(std::vector<float>& logits, const std::vector<int>& recent) const;
    void apply_top_k(std::vector<float>& logits) const;
    void apply_top_p(std::vector<float>& probs) const;
};

} // namespace nf
