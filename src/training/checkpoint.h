#pragma once
#include "model/transformer.h"
#include "tokenizer/tokenizer.h"
#include <string>

namespace nf {

class Checkpoint {
public:
    static bool save(const std::string& path, const Transformer& model, const Tokenizer& tokenizer);
    static bool load(const std::string& path, Transformer& model, Tokenizer& tokenizer);

    static bool save_model_only(const std::string& path, const Transformer& model);
    static bool load_model_only(const std::string& path, Transformer& model);
};

} // namespace nf
