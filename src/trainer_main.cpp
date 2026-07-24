#include "model/transformer.h"
#include "tokenizer/tokenizer.h"
#include "training/trainer.h"
#include "training/dataset.h"
#include "training/checkpoint.h"
#include "inference/generator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

void print_usage() {
    std::cout << "NeuralForge AI Trainer\n";
    std::cout << "=====================\n\n";
    std::cout << "Usage:\n";
    std::cout << "  ai_trainer --data <file.txt> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --data <path>       Training text file (required)\n";
    std::cout << "  --epochs <n>        Number of epochs (default: 10)\n";
    std::cout << "  --lr <float>        Learning rate (default: 3e-4)\n";
    std::cout << "  --batch <n>         Batch size (default: 4)\n";
    std::cout << "  --seq-len <n>       Sequence length (default: 64)\n";
    std::cout << "  --layers <n>        Number of transformer layers (default: 4)\n";
    std::cout << "  --dim <n>           Model dimension (default: 128)\n";
    std::cout << "  --heads <n>         Number of attention heads (default: 4)\n";
    std::cout << "  --ff-dim <n>        Feed-forward dimension (default: 512)\n";
    std::cout << "  --vocab <n>         Vocabulary size (default: 1000)\n";
    std::cout << "  --output <path>     Output model path (default: models/checkpoint)\n";
    std::cout << "  --chat              Enter chat mode after training\n";
    std::cout << "  --load <path>       Load existing model instead of training\n";
}

int main(int argc, char* argv[]) {
    std::string data_path;
    std::string output_path = "models/checkpoint";
    std::string load_path;
    bool do_chat = false;

    nf::ModelConfig model_config;
    nf::TrainConfig train_config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) data_path = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) train_config.epochs = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) train_config.learning_rate = std::stof(argv[++i]);
        else if (arg == "--batch" && i + 1 < argc) train_config.batch_size = std::stoi(argv[++i]);
        else if (arg == "--seq-len" && i + 1 < argc) train_config.seq_len = std::stoi(argv[++i]);
        else if (arg == "--layers" && i + 1 < argc) model_config.n_layers = std::stoi(argv[++i]);
        else if (arg == "--dim" && i + 1 < argc) model_config.d_model = std::stoi(argv[++i]);
        else if (arg == "--heads" && i + 1 < argc) model_config.n_heads = std::stoi(argv[++i]);
        else if (arg == "--ff-dim" && i + 1 < argc) model_config.d_ff = std::stoi(argv[++i]);
        else if (arg == "--vocab" && i + 1 < argc) model_config.vocab_size = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--load" && i + 1 < argc) load_path = argv[++i];
        else if (arg == "--chat") do_chat = true;
        else if (arg == "--help" || arg == "-h") { print_usage(); return 0; }
    }

    nf::Transformer model;
    nf::Tokenizer tokenizer;

    if (!load_path.empty()) {
        // Load existing model
        std::cout << "Loading model from " << load_path << "...\n";
        if (!nf::Checkpoint::load(load_path, model, tokenizer)) {
            std::cerr << "Failed to load model\n";
            return 1;
        }
        do_chat = true;
    } else {
        // Train new model
        if (data_path.empty()) {
            print_usage();
            return 1;
        }

        // Read training data
        std::ifstream file(data_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << data_path << "\n";
            return 1;
        }
        std::stringstream buf;
        buf << file.rdbuf();
        std::string text = buf.str();

        std::cout << "Training tokenizer...\n";
        tokenizer.train(text, model_config.vocab_size);
        model_config.vocab_size = tokenizer.vocab_size();

        std::cout << "Initializing model...\n";
        model.init(model_config);

        nf::Dataset dataset;
        dataset.seq_len = train_config.seq_len;
        dataset.batch_size = train_config.batch_size;
        dataset.load_from_string(text, tokenizer);

        train_config.save_path = output_path;

        nf::Trainer trainer;
        trainer.train(model, dataset, tokenizer, train_config);
    }

    // Chat mode
    if (do_chat) {
        std::cout << "\n=== Chat Mode ===\n";
        std::cout << "Type your message and press Enter. Type 'quit' to exit.\n\n";

        nf::Generator generator;
        nf::GenerateConfig gen_config;
        gen_config.max_tokens = 200;

        while (true) {
            std::cout << "You: ";
            std::string input;
            std::getline(std::cin, input);
            if (input == "quit" || input == "exit") break;
            if (input.empty()) continue;

            std::cout << "AI: ";
            std::string response = generator.generate(model, tokenizer, input, gen_config,
                [](const std::string& token) {
                    std::cout << token;
                    std::cout.flush();
                });
            std::cout << "\n\n";
        }
    }

    return 0;
}
