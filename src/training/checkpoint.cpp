#include "training/checkpoint.h"
#include <fstream>
#include <iostream>

namespace nf {

static void write_tensor(std::ofstream& f, const Tensor& t) {
    size_t r = t.rows, c = t.cols;
    f.write(reinterpret_cast<const char*>(&r), sizeof(size_t));
    f.write(reinterpret_cast<const char*>(&c), sizeof(size_t));
    f.write(reinterpret_cast<const char*>(t.data.data()), t.data.size() * sizeof(float));
}

static Tensor read_tensor(std::ifstream& f) {
    size_t r, c;
    f.read(reinterpret_cast<char*>(&r), sizeof(size_t));
    f.read(reinterpret_cast<char*>(&c), sizeof(size_t));
    Tensor t(r, c);
    f.read(reinterpret_cast<char*>(t.data.data()), t.data.size() * sizeof(float));
    return t;
}

bool Checkpoint::save(const std::string& path, const Transformer& model, const Tokenizer& tokenizer) {
    // Save tokenizer
    tokenizer.save(path + ".tok");

    // Save model
    return save_model_only(path, model);
}

bool Checkpoint::load(const std::string& path, Transformer& model, Tokenizer& tokenizer) {
    if (!tokenizer.load(path + ".tok")) {
        std::cerr << "[Checkpoint] Failed to load tokenizer from " << path << ".tok\n";
        return false;
    }
    return load_model_only(path, model);
}

bool Checkpoint::save_model_only(const std::string& path, const Transformer& model) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[Checkpoint] Failed to save to " << path << "\n";
        return false;
    }

    // Write magic and config
    uint32_t magic = 0x4E465F31; // "NF_1"
    f.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(&model.config), sizeof(ModelConfig));

    // Write all parameters
    // Cast away const for reading parameters (they won't be modified)
    auto& nc_model = const_cast<Transformer&>(model);
    auto params = nc_model.parameters();
    int num_params = static_cast<int>(params.size());
    f.write(reinterpret_cast<const char*>(&num_params), sizeof(int));

    for (auto* p : params) {
        write_tensor(f, *p);
    }

    std::cout << "[Checkpoint] Saved model to " << path << " (" << num_params << " parameter tensors)\n";
    return true;
}

bool Checkpoint::load_model_only(const std::string& path, Transformer& model) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[Checkpoint] Failed to load from " << path << "\n";
        return false;
    }

    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    if (magic != 0x4E465F31) {
        std::cerr << "[Checkpoint] Invalid file format\n";
        return false;
    }

    ModelConfig config;
    f.read(reinterpret_cast<char*>(&config), sizeof(ModelConfig));

    // Re-init model with loaded config
    model.init(config);

    int num_params;
    f.read(reinterpret_cast<char*>(&num_params), sizeof(int));

    auto params = model.parameters();
    if (num_params != static_cast<int>(params.size())) {
        std::cerr << "[Checkpoint] Parameter count mismatch: file=" << num_params
                  << " model=" << params.size() << "\n";
        return false;
    }

    for (auto* p : params) {
        *p = read_tensor(f);
    }

    std::cout << "[Checkpoint] Loaded model from " << path << "\n";
    return true;
}

} // namespace nf
