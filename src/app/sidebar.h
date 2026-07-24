#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "inference/sampler.h"
#include "model/config.h"
#include "training/trainer.h"
#include <string>
#include <vector>

namespace nf {

enum ModelPreset {
    PRESET_TINY = 0,    // ~100K params - fast training, basic patterns
    PRESET_SMALL,       // ~1M params - decent quality
    PRESET_MEDIUM,      // ~10M params - good quality
    PRESET_LARGE,       // ~50M params - very good quality
    PRESET_XL,          // ~100M+ params - best quality, needs lots of RAM
    PRESET_COUNT
};

struct PresetInfo {
    const char* name;
    const char* description;
    ModelConfig config;
    size_t approx_params;
    const char* ram_estimate;
};

const PresetInfo& GetPreset(ModelPreset preset);

class Sidebar {
public:
    ModelConfig model_config;
    SamplerConfig sampler_config;
    TrainConfig train_config;
    ModelPreset current_preset = PRESET_TINY;

    bool model_loaded = false;
    bool is_training = false;
    bool data_loaded = false;
    std::string model_status = "NO MODEL LOADED";
    std::string train_status = "";
    float last_loss = 0.0f;
    size_t param_count = 0;

    void paint(HDC hdc, RECT area);
    int handle_click(int x, int y, RECT area);

    static const int BTN_NONE = 0;
    static const int BTN_TRAIN = 1;
    static const int BTN_STOP = 2;
    static const int BTN_NEW_CHAT = 3;
    static const int BTN_LOAD_MODEL = 4;
    static const int BTN_LOAD_DATA = 5;
    static const int BTN_FINETUNE = 6;
    static const int BTN_PRESET_NEXT = 7;
    static const int BTN_PRESET_PREV = 8;

private:
    struct ButtonRect {
        RECT rect;
        int id;
    };
    std::vector<ButtonRect> buttons_;

    void draw_neon_button(HDC hdc, const char* text, RECT r, COLORREF neon_color, COLORREF bg, int id);
    void draw_stat_line(HDC hdc, const char* label, const char* value, int y, int x, COLORREF val_color);
};

} // namespace nf
