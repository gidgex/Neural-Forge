#include "training/dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace nf {

bool Dataset::load_text(const std::string& path, const Tokenizer& tokenizer) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Dataset] Failed to open " << path << "\n";
        return false;
    }
    std::stringstream buf;
    buf << file.rdbuf();
    std::string text = buf.str();
    load_from_string(text, tokenizer);
    return true;
}

void Dataset::load_from_string(const std::string& text, const Tokenizer& tokenizer) {
    tokens = tokenizer.encode(text);
    current_pos = 0;
    std::cout << "[Dataset] Loaded " << tokens.size() << " tokens\n";
}

Batch Dataset::next_batch() {
    Batch batch;
    batch.batch_size = batch_size;
    batch.seq_len = seq_len;
    batch.input_ids.resize(batch_size * seq_len);
    batch.target_ids.resize(batch_size * seq_len);

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            size_t idx = (current_pos + b * seq_len + t) % tokens.size();
            size_t idx_next = (idx + 1) % tokens.size();
            batch.input_ids[b * seq_len + t] = tokens[idx];
            batch.target_ids[b * seq_len + t] = tokens[idx_next];
        }
    }

    current_pos += batch_size * seq_len;
    if (current_pos >= tokens.size()) {
        current_pos = current_pos % tokens.size();
    }

    return batch;
}

bool Dataset::has_more() const {
    return tokens.size() > static_cast<size_t>(seq_len + 1);
}

void Dataset::reset() {
    current_pos = 0;
}

size_t Dataset::num_batches() const {
    if (tokens.empty()) return 0;
    return tokens.size() / (batch_size * seq_len);
}

} // namespace nf
