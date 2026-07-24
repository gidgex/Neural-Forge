#pragma once
#include "model/transformer.h"
#include "training/optimizer.h"
#include "training/dataset.h"
#include <string>
#include <functional>
#include <atomic>

namespace nf {

struct TrainConfig {
    int epochs = 10;
    float learning_rate = 3e-4f;
    float weight_decay = 0.01f;
    int batch_size = 4;
    int seq_len = 64;
    int save_every = 5;       // Save checkpoint every N epochs
    int log_every = 10;       // Log every N steps
    std::string save_path = "models/checkpoint";
};

struct TrainStats {
    float loss = 0.0f;
    int epoch = 0;
    int step = 0;
    int total_steps = 0;
    float tokens_per_sec = 0.0f;
};

using TrainCallback = std::function<void(const TrainStats&)>;

class Trainer {
public:
    Trainer() = default;

    void train(Transformer& model, Dataset& dataset, const Tokenizer& tokenizer,
               const TrainConfig& config, TrainCallback callback = nullptr);

    void stop() { should_stop_ = true; }
    bool is_training() const { return is_training_; }

private:
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> is_training_{false};

    float compute_loss_and_grad(Transformer& model, const Batch& batch);
};

} // namespace nf
