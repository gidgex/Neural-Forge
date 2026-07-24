#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>

#include "app/chat_ui.h"
#include "app/sidebar.h"
#include "model/transformer.h"
#include "tokenizer/tokenizer.h"
#include "training/trainer.h"
#include "training/dataset.h"
#include "inference/generator.h"

namespace nf {

class MainWindow {
public:
    MainWindow();
    ~MainWindow();

    bool create(HINSTANCE hInstance, int nCmdShow);
    int run();

    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

private:
    HWND hwnd_ = nullptr;
    HWND input_edit_ = nullptr;
    HWND send_btn_ = nullptr;
    HINSTANCE hInstance_ = nullptr;

    ChatUI chat_;
    Sidebar sidebar_;

    Transformer model_;
    Tokenizer tokenizer_;
    Dataset dataset_;
    Trainer trainer_;
    Generator generator_;

    bool model_initialized_ = false;
    bool data_loaded_ = false;
    std::string data_path_;

    std::mutex ui_mutex_;
    std::atomic<bool> generating_{false};

    void on_paint();
    void on_resize();
    void on_send();
    void on_mouse_wheel(int delta);
    void on_sidebar_click(int x, int y);

    void load_training_data();
    void load_model();
    void start_training();
    void stop_training();
    void fine_tune_model();
    void generate_response(const std::string& prompt);

    void init_default_model();
    std::string open_file_dialog(const char* filter);

    // Custom messages
    static const UINT WM_UPDATE_UI = WM_USER + 1;
    static const UINT WM_GENERATION_DONE = WM_USER + 2;
};

} // namespace nf
