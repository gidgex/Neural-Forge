#pragma once
#include "model/transformer.h"
#include "tokenizer/tokenizer.h"
#include "inference/sampler.h"
#include <string>
#include <functional>
#include <atomic>

namespace nf {

struct GenerateConfig {
    int max_tokens = 256;
    SamplerConfig sampler;
    bool stream = true;
};

using StreamCallback = std::function<void(const std::string& token)>;

class Generator {
public:
    Generator() = default;

    // Generate text given a prompt
    std::string generate(Transformer& model, const Tokenizer& tokenizer,
                         const std::string& prompt, const GenerateConfig& config,
                         StreamCallback on_token = nullptr);

    void stop() { should_stop_ = true; }

private:
    std::atomic<bool> should_stop_{false};
};

} // namespace nf
