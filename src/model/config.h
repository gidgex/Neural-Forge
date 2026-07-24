#pragma once

namespace nf {

struct ModelConfig {
    int vocab_size = 1000;
    int d_model = 128;
    int n_heads = 4;
    int n_layers = 4;
    int d_ff = 512;
    int max_seq_len = 256;
    float dropout = 0.0f;

    int head_dim() const { return d_model / n_heads; }
};

} // namespace nf
