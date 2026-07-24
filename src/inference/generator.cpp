#include "inference/generator.h"
#include <iostream>

namespace nf {

std::string Generator::generate(Transformer& model, const Tokenizer& tokenizer,
                                 const std::string& prompt, const GenerateConfig& config,
                                 StreamCallback on_token) {
    should_stop_ = false;

    // Encode prompt
    std::vector<int> tokens = tokenizer.encode(prompt);
    if (tokens.empty()) {
        tokens.push_back(tokenizer.bos_id());
    }

    Sampler sampler(config.sampler);
    std::string generated;
    int max_ctx = model.config.max_seq_len;

    for (int i = 0; i < config.max_tokens && !should_stop_; i++) {
        // Use last max_ctx tokens as context
        int ctx_len = std::min(static_cast<int>(tokens.size()), max_ctx);
        std::vector<int> context(tokens.end() - ctx_len, tokens.end());

        // Forward pass
        Tensor logits = model.forward(context, ctx_len);

        // Get logits for the last position
        int last_row = static_cast<int>(logits.rows) - 1;
        Tensor last_logits(1, logits.cols);
        for (size_t j = 0; j < logits.cols; j++) {
            last_logits.data[j] = logits.data[last_row * logits.cols + j];
        }

        // Build recent tokens for repetition penalty
        int recent_n = std::min(static_cast<int>(tokens.size()), 64);
        std::vector<int> recent(tokens.end() - recent_n, tokens.end());

        // Sample next token
        int next_id = sampler.sample(last_logits, recent);

        // Stop on EOS
        if (next_id == tokenizer.eos_id()) break;

        tokens.push_back(next_id);

        // Decode this token
        std::string tok_str = tokenizer.decode({next_id});
        generated += tok_str;

        if (on_token) {
            on_token(tok_str);
        }
    }

    return generated;
}

} // namespace nf
