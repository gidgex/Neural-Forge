#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace nf {

struct BPEMerge {
    int left;
    int right;
    int result;
};

class Tokenizer {
public:
    Tokenizer() = default;

    // Train BPE on text corpus
    void train(const std::string& text, int vocab_size = 1000);

    // Encode text to token IDs
    std::vector<int> encode(const std::string& text) const;

    // Decode token IDs to text
    std::string decode(const std::vector<int>& ids) const;

    // Vocabulary info
    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    int pad_id() const { return pad_id_; }
    int eos_id() const { return eos_id_; }
    int bos_id() const { return bos_id_; }
    int unk_id() const { return unk_id_; }

    // Save/load
    void save(const std::string& path) const;
    bool load(const std::string& path);

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    std::vector<BPEMerge> merges_;
    int pad_id_ = 0;
    int unk_id_ = 1;
    int bos_id_ = 2;
    int eos_id_ = 3;

    void init_base_vocab();
    std::vector<int> text_to_base_tokens(const std::string& text) const;
    void apply_merges(std::vector<int>& tokens) const;
};

} // namespace nf
