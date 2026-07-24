#include "inference/sampler.h"
#include "core/random.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace nf {

int Sampler::greedy(const Tensor& logits) const {
    int best = 0;
    float best_val = logits.data[0];
    for (size_t i = 1; i < logits.data.size(); i++) {
        if (logits.data[i] > best_val) {
            best_val = logits.data[i];
            best = static_cast<int>(i);
        }
    }
    return best;
}

void Sampler::apply_temperature(std::vector<float>& logits) const {
    if (config.temperature <= 0.0f) return;
    float inv_temp = 1.0f / config.temperature;
    for (auto& l : logits) l *= inv_temp;
}

void Sampler::apply_repetition_penalty(std::vector<float>& logits, const std::vector<int>& recent) const {
    if (config.repetition_penalty <= 1.0f) return;
    for (int id : recent) {
        if (id >= 0 && id < static_cast<int>(logits.size())) {
            if (logits[id] > 0) {
                logits[id] /= config.repetition_penalty;
            } else {
                logits[id] *= config.repetition_penalty;
            }
        }
    }
}

void Sampler::apply_top_k(std::vector<float>& logits) const {
    if (config.top_k <= 0 || config.top_k >= static_cast<int>(logits.size())) return;

    std::vector<std::pair<float, int>> indexed(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        indexed[i] = {logits[i], static_cast<int>(i)};
    }
    std::partial_sort(indexed.begin(), indexed.begin() + config.top_k, indexed.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    float threshold = indexed[config.top_k - 1].first;
    for (size_t i = 0; i < logits.size(); i++) {
        if (logits[i] < threshold) {
            logits[i] = -1e9f;
        }
    }
}

void Sampler::apply_top_p(std::vector<float>& probs) const {
    if (config.top_p >= 1.0f) return;

    std::vector<std::pair<float, int>> indexed(probs.size());
    for (size_t i = 0; i < probs.size(); i++) {
        indexed[i] = {probs[i], static_cast<int>(i)};
    }
    std::sort(indexed.begin(), indexed.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    float cumsum = 0.0f;
    size_t cutoff = indexed.size();
    for (size_t i = 0; i < indexed.size(); i++) {
        cumsum += indexed[i].first;
        if (cumsum >= config.top_p) {
            cutoff = i + 1;
            break;
        }
    }

    // Zero out everything below cutoff
    std::vector<bool> keep(probs.size(), false);
    for (size_t i = 0; i < cutoff; i++) {
        keep[indexed[i].second] = true;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        if (!keep[i]) probs[i] = 0.0f;
    }
}

int Sampler::sample(const Tensor& logits, const std::vector<int>& recent_tokens) const {
    if (config.temperature < 0.01f) {
        return greedy(logits);
    }

    std::vector<float> l(logits.data.begin(), logits.data.end());

    apply_repetition_penalty(l, recent_tokens);
    apply_temperature(l);
    apply_top_k(l);

    // Softmax
    float max_val = *std::max_element(l.begin(), l.end());
    float sum = 0.0f;
    for (auto& v : l) {
        v = std::exp(v - max_val);
        sum += v;
    }
    for (auto& v : l) v /= sum;

    apply_top_p(l);

    // Re-normalize
    sum = std::accumulate(l.begin(), l.end(), 0.0f);
    if (sum > 0.0f) {
        for (auto& v : l) v /= sum;
    }

    // Weighted random sample
    float r = Random::instance().uniform(0.0f, 1.0f);
    float cumsum = 0.0f;
    for (size_t i = 0; i < l.size(); i++) {
        cumsum += l[i];
        if (cumsum >= r) return static_cast<int>(i);
    }

    return static_cast<int>(l.size() - 1);
}

} // namespace nf
