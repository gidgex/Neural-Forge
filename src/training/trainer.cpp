#include "training/trainer.h"
#include "training/checkpoint.h"
#include <cmath>
#include <chrono>
#include <iostream>

namespace nf {

float Trainer::compute_loss_and_grad(Transformer& model, const Batch& batch) {
    // Forward pass
    Tensor logits = model.forward(batch.input_ids, batch.seq_len);

    int n = static_cast<int>(logits.rows);
    int vocab_size = static_cast<int>(logits.cols);

    // Cross-entropy loss with softmax
    Tensor probs = logits.softmax(-1);

    float loss = 0.0f;
    Tensor grad_logits(n, vocab_size);

    for (int i = 0; i < n; i++) {
        int target = batch.target_ids[i];
        if (target >= 0 && target < vocab_size) {
            float p = std::max(probs.data[i * vocab_size + target], 1e-10f);
            loss -= std::log(p);

            // Gradient of cross-entropy with softmax: probs - one_hot(target)
            for (int j = 0; j < vocab_size; j++) {
                grad_logits.data[i * vocab_size + j] = probs.data[i * vocab_size + j];
            }
            grad_logits.data[i * vocab_size + target] -= 1.0f;

            // Scale by 1/n
            for (int j = 0; j < vocab_size; j++) {
                grad_logits.data[i * vocab_size + j] /= static_cast<float>(n);
            }
        }
    }

    loss /= static_cast<float>(n);

    // Backward pass
    model.backward(grad_logits);

    return loss;
}

void Trainer::train(Transformer& model, Dataset& dataset, const Tokenizer& tokenizer,
                    const TrainConfig& config, TrainCallback callback) {
    should_stop_ = false;
    is_training_ = true;

    dataset.batch_size = config.batch_size;
    dataset.seq_len = config.seq_len;

    AdamOptimizer optimizer(config.learning_rate, 0.9f, 0.999f, 1e-8f, config.weight_decay);
    auto params = model.parameters();
    auto grads = model.gradients();
    optimizer.init(params);

    int total_steps = config.epochs * static_cast<int>(dataset.num_batches());
    int step = 0;

    std::cout << "\n[Training] Starting training\n";
    std::cout << "[Training] Epochs: " << config.epochs
              << ", Batch size: " << config.batch_size
              << ", Seq len: " << config.seq_len << "\n";
    std::cout << "[Training] Total tokens: " << dataset.total_tokens()
              << ", Batches/epoch: " << dataset.num_batches()
              << ", Total steps: " << total_steps << "\n\n";

    for (int epoch = 0; epoch < config.epochs && !should_stop_; epoch++) {
        dataset.reset();
        float epoch_loss = 0.0f;
        int epoch_steps = 0;

        auto epoch_start = std::chrono::high_resolution_clock::now();

        size_t batches_per_epoch = dataset.num_batches();
        for (size_t b = 0; b < batches_per_epoch && !should_stop_; b++) {
            auto step_start = std::chrono::high_resolution_clock::now();

            model.zero_grad();
            Batch batch = dataset.next_batch();
            float loss = compute_loss_and_grad(model, batch);

            // Get fresh pointers (they don't change but be safe)
            params = model.parameters();
            grads = model.gradients();
            optimizer.step(params, grads);

            epoch_loss += loss;
            epoch_steps++;
            step++;

            auto step_end = std::chrono::high_resolution_clock::now();
            float step_time = std::chrono::duration<float>(step_end - step_start).count();
            float tps = static_cast<float>(config.batch_size * config.seq_len) / step_time;

            if (step % config.log_every == 0 || b == 0) {
                TrainStats stats;
                stats.loss = loss;
                stats.epoch = epoch;
                stats.step = step;
                stats.total_steps = total_steps;
                stats.tokens_per_sec = tps;

                std::cout << "[Step " << step << "/" << total_steps
                          << "] Epoch " << (epoch + 1) << "/" << config.epochs
                          << " | Loss: " << loss
                          << " | " << tps << " tok/s\n";

                if (callback) callback(stats);
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        float epoch_time = std::chrono::duration<float>(epoch_end - epoch_start).count();

        float avg_loss = epoch_steps > 0 ? epoch_loss / epoch_steps : 0.0f;
        std::cout << "\n[Epoch " << (epoch + 1) << "/" << config.epochs
                  << "] Avg loss: " << avg_loss
                  << " | Time: " << epoch_time << "s\n\n";

        // Save checkpoint
        if (config.save_every > 0 && (epoch + 1) % config.save_every == 0) {
            std::string ckpt_path = config.save_path + "_epoch" + std::to_string(epoch + 1) + ".bin";
            Checkpoint::save(ckpt_path, model, tokenizer);
        }
    }

    // Final save
    if (!should_stop_) {
        std::string final_path = config.save_path + "_final.bin";
        Checkpoint::save(final_path, model, tokenizer);
        std::cout << "[Training] Complete! Final model saved to " << final_path << "\n";
    }

    is_training_ = false;
}

} // namespace nf
