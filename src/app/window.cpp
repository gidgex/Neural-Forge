#include "app/window.h"
#include "app/theme.h"
#include "training/checkpoint.h"
#include <commdlg.h>
#include <commctrl.h>
#include <sstream>
#include <fstream>
#include <iostream>

#pragma comment(lib, "comctl32.lib")

namespace nf {

static MainWindow* g_window = nullptr;

MainWindow::MainWindow() {
    g_window = this;
}

MainWindow::~MainWindow() {
    trainer_.stop();
    generator_.stop();
    g_window = nullptr;
}

bool MainWindow::create(HINSTANCE hInstance, int nCmdShow) {
    hInstance_ = hInstance;

    INITCOMMONCONTROLSEX icex = { sizeof(icex), ICC_STANDARD_CLASSES };
    InitCommonControlsEx(&icex);

    WNDCLASSEXA wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = CreateSolidBrush(g_theme.bg_primary);
    wc.lpszClassName = "NeuralForgeWnd";
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    RegisterClassExA(&wc);

    hwnd_ = CreateWindowExA(
        0, "NeuralForgeWnd", "NeuralForge AI  //  Quantum Neural Engine",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1280, 860,
        NULL, NULL, hInstance, NULL
    );

    if (!hwnd_) return false;

    // Create input edit - dark styled
    input_edit_ = CreateWindowExA(
        0, "EDIT", "",
        WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_AUTOVSCROLL | WS_TABSTOP,
        0, 0, 100, 40,
        hwnd_, (HMENU)101, hInstance, NULL
    );

    // Create send button
    send_btn_ = CreateWindowExA(
        0, "BUTTON", ">>",
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        0, 0, 60, 40,
        hwnd_, (HMENU)102, hInstance, NULL
    );

    // Set default config from preset
    const PresetInfo& preset = GetPreset(PRESET_TINY);
    sidebar_.model_config = preset.config;
    sidebar_.current_preset = PRESET_TINY;
    sidebar_.sampler_config = SamplerConfig();
    sidebar_.train_config = TrainConfig();

    ShowWindow(hwnd_, nCmdShow);
    UpdateWindow(hwnd_);
    on_resize();

    return true;
}

int MainWindow::run() {
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        if (msg.message == WM_KEYDOWN && msg.wParam == VK_RETURN) {
            if (GetFocus() == input_edit_ && !(GetKeyState(VK_SHIFT) & 0x8000)) {
                on_send();
                continue;
            }
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return (int)msg.wParam;
}

LRESULT CALLBACK MainWindow::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (!g_window) return DefWindowProcA(hwnd, msg, wParam, lParam);

    switch (msg) {
    case WM_PAINT:
        g_window->on_paint();
        return 0;
    case WM_SIZE:
        g_window->on_resize();
        return 0;
    case WM_MOUSEWHEEL: {
        int delta = GET_WHEEL_DELTA_WPARAM(wParam) / WHEEL_DELTA;
        g_window->on_mouse_wheel(delta);
        return 0;
    }
    case WM_LBUTTONDOWN: {
        int x = LOWORD(lParam);
        int y = HIWORD(lParam);
        if (x < g_theme.sidebar_width) {
            g_window->on_sidebar_click(x, y);
        }
        return 0;
    }
    case WM_COMMAND:
        if (LOWORD(wParam) == 102 && HIWORD(wParam) == BN_CLICKED) {
            g_window->on_send();
        }
        return 0;
    case WM_UPDATE_UI:
        InvalidateRect(hwnd, NULL, FALSE);
        return 0;
    case WM_GENERATION_DONE:
        g_window->generating_ = false;
        InvalidateRect(hwnd, NULL, FALSE);
        return 0;
    case WM_CTLCOLOREDIT: {
        HDC hdcEdit = (HDC)wParam;
        SetTextColor(hdcEdit, g_theme.text_primary);
        SetBkColor(hdcEdit, g_theme.bg_input);
        static HBRUSH hBrush = CreateSolidBrush(g_theme.bg_input);
        return (LRESULT)hBrush;
    }
    case WM_CTLCOLORBTN: {
        HDC hdcBtn = (HDC)wParam;
        SetTextColor(hdcBtn, g_theme.neon_cyan);
        SetBkColor(hdcBtn, g_theme.bg_tertiary);
        static HBRUSH btnBrush = CreateSolidBrush(g_theme.bg_tertiary);
        return (LRESULT)btnBrush;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_ERASEBKGND:
        return 1;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

void MainWindow::on_paint() {
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hwnd_, &ps);

    RECT client;
    GetClientRect(hwnd_, &client);

    // Double buffer
    HDC memDC = CreateCompatibleDC(hdc);
    HBITMAP memBM = CreateCompatibleBitmap(hdc, client.right, client.bottom);
    SelectObject(memDC, memBM);

    // Fill deep void background
    HBRUSH bg_brush = CreateSolidBrush(g_theme.bg_primary);
    FillRect(memDC, &client, bg_brush);
    DeleteObject(bg_brush);

    // Sidebar
    RECT sidebar_rect = {0, 0, g_theme.sidebar_width, client.bottom};
    sidebar_.paint(memDC, sidebar_rect);

    // Chat area
    RECT chat_rect = {
        g_theme.sidebar_width,
        0,
        client.right,
        client.bottom - g_theme.input_height
    };

    // Header bar
    RECT header = {chat_rect.left, 0, chat_rect.right, g_theme.header_height};
    HBRUSH hdr_bg = CreateSolidBrush(g_theme.bg_secondary);
    FillRect(memDC, &header, hdr_bg);
    DeleteObject(hdr_bg);

    SetBkMode(memDC, TRANSPARENT);

    // Header neon text — large and readable
    HFONT hdr_font = CreateFontA(g_theme.font_size_header, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                  DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
    HFONT old_font = (HFONT)SelectObject(memDC, hdr_font);
    SetTextColor(memDC, g_theme.neon_cyan);
    int hdr_y = (g_theme.header_height - g_theme.font_size_header) / 2;
    TextOutA(memDC, chat_rect.left + g_theme.padding, hdr_y,
             "\xC4\xC4 CHAT INTERFACE", 17);

    if (generating_) {
        SetTextColor(memDC, g_theme.neon_magenta);
        const char* gen_text = "\xB0\xB1\xB2 GENERATING \xB2\xB1\xB0";
        TextOutA(memDC, chat_rect.right - 200, hdr_y, gen_text, (int)strlen(gen_text));
    } else if (model_initialized_) {
        SetTextColor(memDC, g_theme.neon_green);
        TextOutA(memDC, chat_rect.right - 130, hdr_y, "\xFE ONLINE", 8);
    }

    SelectObject(memDC, old_font);
    DeleteObject(hdr_font);

    // Neon line under header
    DrawNeonLine(memDC, chat_rect.left, g_theme.header_height, chat_rect.right, g_theme.header_height, g_theme.neon_cyan);

    chat_rect.top = g_theme.header_height + 2;

    std::lock_guard<std::mutex> lock(ui_mutex_);
    chat_.paint(memDC, chat_rect);

    // Input area
    RECT input_area = {
        g_theme.sidebar_width,
        client.bottom - g_theme.input_height,
        client.right,
        client.bottom
    };
    HBRUSH inp_bg = CreateSolidBrush(g_theme.bg_secondary);
    FillRect(memDC, &input_area, inp_bg);
    DeleteObject(inp_bg);

    // Neon line above input
    DrawNeonLine(memDC, input_area.left, input_area.top, input_area.right, input_area.top, g_theme.neon_purple);

    // Blit
    BitBlt(hdc, 0, 0, client.right, client.bottom, memDC, 0, 0, SRCCOPY);
    DeleteObject(memBM);
    DeleteDC(memDC);

    EndPaint(hwnd_, &ps);
}

void MainWindow::on_resize() {
    RECT client;
    GetClientRect(hwnd_, &client);

    int input_y = client.bottom - g_theme.input_height + 14;
    int input_w = client.right - g_theme.sidebar_width - 80 - g_theme.padding * 3;

    MoveWindow(input_edit_,
        g_theme.sidebar_width + g_theme.padding,
        input_y,
        input_w,
        g_theme.input_height - 28,
        TRUE);

    MoveWindow(send_btn_,
        g_theme.sidebar_width + g_theme.padding + input_w + g_theme.padding,
        input_y,
        60,
        g_theme.input_height - 28,
        TRUE);

    InvalidateRect(hwnd_, NULL, FALSE);
}

void MainWindow::on_send() {
    if (generating_) return;

    char buf[4096] = {};
    GetWindowTextA(input_edit_, buf, sizeof(buf));
    std::string text(buf);
    if (text.empty()) return;
    SetWindowTextA(input_edit_, "");

    {
        std::lock_guard<std::mutex> lock(ui_mutex_);
        chat_.add_message(text, true);
    }

    if (!model_initialized_) {
        std::lock_guard<std::mutex> lock(ui_mutex_);
        chat_.add_message("[SYSTEM] No model loaded. Load training data and train, or load an existing model checkpoint.", false);
        InvalidateRect(hwnd_, NULL, FALSE);
        return;
    }

    generate_response(text);
}

void MainWindow::on_mouse_wheel(int delta) {
    POINT pt;
    GetCursorPos(&pt);
    ScreenToClient(hwnd_, &pt);
    if (pt.x > g_theme.sidebar_width) {
        std::lock_guard<std::mutex> lock(ui_mutex_);
        chat_.scroll(delta);
        InvalidateRect(hwnd_, NULL, FALSE);
    }
}

void MainWindow::on_sidebar_click(int x, int y) {
    RECT client;
    GetClientRect(hwnd_, &client);
    RECT sidebar_rect = {0, 0, g_theme.sidebar_width, client.bottom};

    int action = sidebar_.handle_click(x, y, sidebar_rect);

    switch (action) {
    case Sidebar::BTN_NEW_CHAT:
        chat_.clear();
        InvalidateRect(hwnd_, NULL, FALSE);
        break;
    case Sidebar::BTN_LOAD_DATA:
        load_training_data();
        break;
    case Sidebar::BTN_LOAD_MODEL:
        load_model();
        break;
    case Sidebar::BTN_TRAIN:
        start_training();
        break;
    case Sidebar::BTN_STOP:
        stop_training();
        break;
    case Sidebar::BTN_FINETUNE:
        fine_tune_model();
        break;
    case Sidebar::BTN_PRESET_NEXT: {
        int next = (sidebar_.current_preset + 1) % PRESET_COUNT;
        sidebar_.current_preset = (ModelPreset)next;
        const PresetInfo& p = GetPreset(sidebar_.current_preset);
        sidebar_.model_config = p.config;
        InvalidateRect(hwnd_, NULL, FALSE);
        break;
    }
    case Sidebar::BTN_PRESET_PREV: {
        int prev = (sidebar_.current_preset - 1 + PRESET_COUNT) % PRESET_COUNT;
        sidebar_.current_preset = (ModelPreset)prev;
        const PresetInfo& p = GetPreset(sidebar_.current_preset);
        sidebar_.model_config = p.config;
        InvalidateRect(hwnd_, NULL, FALSE);
        break;
    }
    }
}

void MainWindow::load_training_data() {
    std::string path = open_file_dialog("Text Files\0*.txt\0All Files\0*.*\0");
    if (path.empty()) return;

    data_path_ = path;

    std::thread([this]() {
        std::ifstream file(data_path_);
        if (!file.is_open()) {
            sidebar_.model_status = "FAILED TO OPEN FILE";
            PostMessage(hwnd_, WM_UPDATE_UI, 0, 0);
            return;
        }
        std::stringstream buf;
        buf << file.rdbuf();
        std::string text = buf.str();

        sidebar_.model_status = "TRAINING TOKENIZER...";
        PostMessage(hwnd_, WM_UPDATE_UI, 0, 0);

        tokenizer_.train(text, sidebar_.model_config.vocab_size);
        sidebar_.model_config.vocab_size = tokenizer_.vocab_size();

        dataset_.seq_len = sidebar_.train_config.seq_len;
        dataset_.batch_size = sidebar_.train_config.batch_size;
        dataset_.load_from_string(text, tokenizer_);

        data_loaded_ = true;
        sidebar_.data_loaded = true;

        char status[128];
        snprintf(status, sizeof(status), "DATA LOADED [%zu tokens]", dataset_.total_tokens());
        sidebar_.model_status = status;
        PostMessage(hwnd_, WM_UPDATE_UI, 0, 0);
    }).detach();
}

void MainWindow::load_model() {
    std::string path = open_file_dialog("Model Files\0*.bin\0All Files\0*.*\0");
    if (path.empty()) return;

    sidebar_.model_status = "LOADING MODEL...";
    InvalidateRect(hwnd_, NULL, FALSE);

    std::thread([this, path]() {
        if (Checkpoint::load(path, model_, tokenizer_)) {
            model_initialized_ = true;
            sidebar_.model_loaded = true;
            sidebar_.model_config = model_.config;
            sidebar_.param_count = model_.param_count();

            char status[128];
            snprintf(status, sizeof(status), "MODEL ONLINE [%zuM params]", model_.param_count() / 1000000);
            if (model_.param_count() < 1000000) {
                snprintf(status, sizeof(status), "MODEL ONLINE [%zuK params]", model_.param_count() / 1000);
            }
            sidebar_.model_status = status;
        } else {
            sidebar_.model_status = "LOAD FAILED";
        }
        PostMessage(hwnd_, WM_UPDATE_UI, 0, 0);
    }).detach();
}

void MainWindow::init_default_model() {
    model_.init(sidebar_.model_config);
    model_initialized_ = true;
    sidebar_.model_loaded = true;
    sidebar_.param_count = model_.param_count();
}

void MainWindow::start_training() {
    if (!data_loaded_) {
        sidebar_.train_status = "Load training data first!";
        InvalidateRect(hwnd_, NULL, FALSE);
        return;
    }

    if (sidebar_.is_training) return;

    if (!model_initialized_) {
        init_default_model();
    }

    sidebar_.is_training = true;
    sidebar_.train_status = "Initializing...";

    std::thread([this]() {
        TrainConfig cfg = sidebar_.train_config;
        cfg.save_path = "models/checkpoint";

        trainer_.train(model_, dataset_, tokenizer_, cfg,
            [this](const TrainStats& stats) {
                char buf[256];
                snprintf(buf, sizeof(buf), "E%d S%d/%d\nLoss: %.4f | %.0f t/s",
                         stats.epoch + 1, stats.step, stats.total_steps, stats.loss, stats.tokens_per_sec);
                sidebar_.train_status = buf;
                sidebar_.last_loss = stats.loss;
                PostMessage(hwnd_, WM_UPDATE_UI, 0, 0);
            });

        sidebar_.is_training = false;
        sidebar_.model_loaded = true;
        sidebar_.param_count = model_.param_count();

        char status[128];
        snprintf(status, sizeof(status), "TRAINED [%zuK params]", model_.param_count() / 1000);
        sidebar_.model_status = status;
        sidebar_.train_status = "Training complete!";
        PostMessage(hwnd_, WM_UPDATE_UI, 0, 0);
    }).detach();
}

void MainWindow::stop_training() {
    trainer_.stop();
    sidebar_.train_status = "Stopping...";
    InvalidateRect(hwnd_, NULL, FALSE);
}

void MainWindow::fine_tune_model() {
    if (!data_loaded_) {
        sidebar_.train_status = "Load data first, then fine-tune";
        InvalidateRect(hwnd_, NULL, FALSE);
        return;
    }

    if (!model_initialized_) {
        // Try to load a model first
        std::string path = open_file_dialog("Model Files\0*.bin\0All Files\0*.*\0");
        if (path.empty()) return;

        sidebar_.model_status = "LOADING FOR FINE-TUNE...";
        InvalidateRect(hwnd_, NULL, FALSE);

        // Load synchronously for simplicity
        if (!Checkpoint::load(path, model_, tokenizer_)) {
            sidebar_.model_status = "LOAD FAILED";
            InvalidateRect(hwnd_, NULL, FALSE);
            return;
        }
        model_initialized_ = true;
        sidebar_.model_loaded = true;
        sidebar_.model_config = model_.config;

        // Re-tokenize data with loaded tokenizer
        std::ifstream file(data_path_);
        std::stringstream buf;
        buf << file.rdbuf();
        dataset_.load_from_string(buf.str(), tokenizer_);
    }

    // Now start training (fine-tuning) with lower learning rate
    sidebar_.is_training = true;
    sidebar_.train_status = "Fine-tuning...";

    std::thread([this]() {
        TrainConfig cfg = sidebar_.train_config;
        cfg.learning_rate *= 0.1f; // Lower LR for fine-tuning
        cfg.save_path = "models/finetuned";

        trainer_.train(model_, dataset_, tokenizer_, cfg,
            [this](const TrainStats& stats) {
                char buf[256];
                snprintf(buf, sizeof(buf), "[FT] E%d S%d/%d\nLoss: %.4f | %.0f t/s",
                         stats.epoch + 1, stats.step, stats.total_steps, stats.loss, stats.tokens_per_sec);
                sidebar_.train_status = buf;
                PostMessage(hwnd_, WM_UPDATE_UI, 0, 0);
            });

        sidebar_.is_training = false;
        sidebar_.train_status = "Fine-tuning complete!";

        char status[128];
        snprintf(status, sizeof(status), "FINE-TUNED [%zuK params]", model_.param_count() / 1000);
        sidebar_.model_status = status;
        PostMessage(hwnd_, WM_UPDATE_UI, 0, 0);
    }).detach();
}

void MainWindow::generate_response(const std::string& prompt) {
    generating_ = true;

    {
        std::lock_guard<std::mutex> lock(ui_mutex_);
        chat_.start_ai_message();
    }
    InvalidateRect(hwnd_, NULL, FALSE);

    std::thread([this, prompt]() {
        GenerateConfig cfg;
        cfg.sampler = sidebar_.sampler_config;
        cfg.max_tokens = 256;

        generator_.generate(model_, tokenizer_, prompt, cfg,
            [this](const std::string& token) {
                std::lock_guard<std::mutex> lock(ui_mutex_);
                chat_.append_to_last(token);
                PostMessage(hwnd_, WM_UPDATE_UI, 0, 0);
            });

        {
            std::lock_guard<std::mutex> lock(ui_mutex_);
            if (!chat_.messages.empty()) {
                chat_.messages.back().is_streaming = false;
                if (chat_.messages.back().text.empty()) {
                    chat_.messages.back().text = "[No output. Train the model longer for better results.]";
                }
            }
        }
        PostMessage(hwnd_, WM_GENERATION_DONE, 0, 0);
    }).detach();
}

std::string MainWindow::open_file_dialog(const char* filter) {
    char filename[MAX_PATH] = {};
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = hwnd_;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
    return "";
}

} // namespace nf
