#pragma once
#include "tokenizer/tokenizer.h"
#include <vector>
#include <string>

namespace nf {

struct Batch {
    std::vector<int> input_ids;  // flattened [batch_size x seq_len]
    std::vector<int> target_ids; // flattened [batch_size x seq_len]
    int batch_size;
    int seq_len;
};

class Dataset {
public:
    std::vector<int> tokens;
    int seq_len = 64;
    int batch_size = 4;
    size_t current_pos = 0;

    Dataset() = default;

    bool load_text(const std::string& path, const Tokenizer& tokenizer);
    void load_from_string(const std::string& text, const Tokenizer& tokenizer);

    Batch next_batch();
    bool has_more() const;
    void reset();
    size_t num_batches() const;
    size_t total_tokens() const { return tokens.size(); }
};

} // namespace nf
