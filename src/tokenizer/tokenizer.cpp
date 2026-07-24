#include "tokenizer/tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <unordered_set>

namespace nf {

void Tokenizer::init_base_vocab() {
    token_to_id_.clear();
    id_to_token_.clear();

    // Special tokens
    token_to_id_["<pad>"] = 0;  id_to_token_[0] = "<pad>";
    token_to_id_["<unk>"] = 1;  id_to_token_[1] = "<unk>";
    token_to_id_["<bos>"] = 2;  id_to_token_[2] = "<bos>";
    token_to_id_["<eos>"] = 3;  id_to_token_[3] = "<eos>";

    // All single bytes as base vocabulary (IDs 4-259)
    for (int i = 0; i < 256; i++) {
        std::string tok(1, static_cast<char>(i));
        int id = i + 4;
        token_to_id_[tok] = id;
        id_to_token_[id] = tok;
    }
}

std::vector<int> Tokenizer::text_to_base_tokens(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text) {
        tokens.push_back(static_cast<int>(c) + 4);
    }
    return tokens;
}

void Tokenizer::apply_merges(std::vector<int>& tokens) const {
    for (const auto& merge : merges_) {
        size_t i = 0;
        while (i + 1 < tokens.size()) {
            if (tokens[i] == merge.left && tokens[i + 1] == merge.right) {
                tokens[i] = merge.result;
                tokens.erase(tokens.begin() + i + 1);
            } else {
                i++;
            }
        }
    }
}

void Tokenizer::train(const std::string& text, int target_vocab_size) {
    init_base_vocab();
    merges_.clear();

    std::vector<int> tokens = text_to_base_tokens(text);

    int next_id = 260; // 4 special + 256 bytes

    std::cout << "[Tokenizer] Training BPE, target vocab size: " << target_vocab_size << "\n";
    std::cout << "[Tokenizer] Text length: " << text.size() << " chars, " << tokens.size() << " base tokens\n";

    while (next_id < target_vocab_size && tokens.size() > 1) {
        // Count all adjacent pairs
        std::unordered_map<uint64_t, int> pair_counts;
        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            uint64_t key = (static_cast<uint64_t>(tokens[i]) << 32) | static_cast<uint64_t>(tokens[i + 1]);
            pair_counts[key]++;
        }

        if (pair_counts.empty()) break;

        // Find most frequent pair
        uint64_t best_key = 0;
        int best_count = 0;
        for (auto& [key, count] : pair_counts) {
            if (count > best_count) {
                best_count = count;
                best_key = key;
            }
        }

        if (best_count < 2) break;

        int left = static_cast<int>(best_key >> 32);
        int right = static_cast<int>(best_key & 0xFFFFFFFF);

        // Create new token by concatenating the pair's strings
        std::string new_token = id_to_token_[left] + id_to_token_[right];
        token_to_id_[new_token] = next_id;
        id_to_token_[next_id] = new_token;

        merges_.push_back({left, right, next_id});

        // Merge in the token list
        for (size_t i = 0; i + 1 < tokens.size(); ) {
            if (tokens[i] == left && tokens[i + 1] == right) {
                tokens[i] = next_id;
                tokens.erase(tokens.begin() + i + 1);
            } else {
                i++;
            }
        }

        if ((next_id - 260) % 100 == 0) {
            std::cout << "[Tokenizer] Merge " << (next_id - 260)
                      << ": \"" << id_to_token_[left] << "\" + \"" << id_to_token_[right]
                      << "\" -> \"" << new_token << "\" (count=" << best_count
                      << ", tokens remaining=" << tokens.size() << ")\n";
        }

        next_id++;
    }

    std::cout << "[Tokenizer] Training complete. Vocab size: " << vocab_size()
              << ", Merges: " << merges_.size() << "\n";
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens = text_to_base_tokens(text);
    apply_merges(tokens);
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id == pad_id_ || id == bos_id_ || id == eos_id_) continue;
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            result += it->second;
        }
    }
    return result;
}

void Tokenizer::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Tokenizer] Failed to save to " << path << "\n";
        return;
    }

    // Write vocab size
    int vs = vocab_size();
    file.write(reinterpret_cast<const char*>(&vs), sizeof(int));

    // Write each token
    for (int i = 0; i < vs; i++) {
        auto it = id_to_token_.find(i);
        if (it != id_to_token_.end()) {
            int len = static_cast<int>(it->second.size());
            file.write(reinterpret_cast<const char*>(&i), sizeof(int));
            file.write(reinterpret_cast<const char*>(&len), sizeof(int));
            file.write(it->second.data(), len);
        }
    }

    // Write merges
    int num_merges = static_cast<int>(merges_.size());
    file.write(reinterpret_cast<const char*>(&num_merges), sizeof(int));
    for (const auto& m : merges_) {
        file.write(reinterpret_cast<const char*>(&m.left), sizeof(int));
        file.write(reinterpret_cast<const char*>(&m.right), sizeof(int));
        file.write(reinterpret_cast<const char*>(&m.result), sizeof(int));
    }

    std::cout << "[Tokenizer] Saved to " << path << "\n";
}

bool Tokenizer::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    token_to_id_.clear();
    id_to_token_.clear();
    merges_.clear();

    int vs;
    file.read(reinterpret_cast<char*>(&vs), sizeof(int));

    for (int i = 0; i < vs; i++) {
        int id, len;
        file.read(reinterpret_cast<char*>(&id), sizeof(int));
        file.read(reinterpret_cast<char*>(&len), sizeof(int));
        std::string tok(len, '\0');
        file.read(&tok[0], len);
        token_to_id_[tok] = id;
        id_to_token_[id] = tok;
    }

    int num_merges;
    file.read(reinterpret_cast<char*>(&num_merges), sizeof(int));
    merges_.resize(num_merges);
    for (auto& m : merges_) {
        file.read(reinterpret_cast<char*>(&m.left), sizeof(int));
        file.read(reinterpret_cast<char*>(&m.right), sizeof(int));
        file.read(reinterpret_cast<char*>(&m.result), sizeof(int));
    }

    std::cout << "[Tokenizer] Loaded from " << path << " (vocab=" << vocab_size()
              << ", merges=" << merges_.size() << ")\n";
    return true;
}

} // namespace nf
