#include "app/sidebar.h"
#include "app/theme.h"
#include <cstdio>

namespace nf {

static PresetInfo s_presets[PRESET_COUNT] = {
    {
        "TINY", "Fast training, basic patterns",
        {1000, 64, 2, 2, 256, 128, 0.0f},
        100000, "~50 MB"
    },
    {
        "SMALL", "Decent text generation",
        {2000, 128, 4, 4, 512, 256, 0.0f},
        1000000, "~200 MB"
    },
    {
        "MEDIUM", "Good quality output",
        {4000, 256, 8, 8, 1024, 512, 0.0f},
        10000000, "~1 GB"
    },
    {
        "LARGE", "High quality, slower training",
        {8000, 512, 8, 12, 2048, 1024, 0.0f},
        50000000, "~4 GB"
    },
    {
        "XL", "Maximum quality, needs 16GB+ RAM",
        {16000, 768, 12, 16, 3072, 2048, 0.0f},
        120000000, "~12 GB"
    }
};

const PresetInfo& GetPreset(ModelPreset preset) {
    return s_presets[preset];
}

void Sidebar::draw_neon_button(HDC hdc, const char* text, RECT r, COLORREF neon_color, COLORREF bg, int id) {
    DrawNeonRect(hdc, r, BlendColor(neon_color, RGB(0, 0, 0), 0.6f), neon_color, 4);

    HBRUSH brush = CreateSolidBrush(bg);
    HPEN pen = CreatePen(PS_SOLID, 1, neon_color);
    SelectObject(hdc, brush);
    SelectObject(hdc, pen);
    RoundRect(hdc, r.left, r.top, r.right, r.bottom, 4, 4);
    DeleteObject(brush);
    DeleteObject(pen);

    // Button text — bold and readable
    HFONT btn_text = CreateFontA(14, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                  DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
    HFONT old = (HFONT)SelectObject(hdc, btn_text);
    SetTextColor(hdc, neon_color);
    SetBkMode(hdc, TRANSPARENT);
    DrawTextA(hdc, text, -1, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    SelectObject(hdc, old);
    DeleteObject(btn_text);

    buttons_.push_back({r, id});
}

void Sidebar::draw_stat_line(HDC hdc, const char* label, const char* value, int y, int x, COLORREF val_color) {
    // Label in muted color
    HFONT stat_font = CreateFontA(g_theme.font_size_small, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                   DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
    HFONT old = (HFONT)SelectObject(hdc, stat_font);

    SetTextColor(hdc, g_theme.text_secondary);
    TextOutA(hdc, x, y, label, (int)strlen(label));

    // Measure label width properly
    SIZE sz;
    GetTextExtentPoint32A(hdc, label, (int)strlen(label), &sz);

    SetTextColor(hdc, val_color);
    TextOutA(hdc, x + sz.cx + 2, y, value, (int)strlen(value));

    SelectObject(hdc, old);
    DeleteObject(stat_font);
}

void Sidebar::paint(HDC hdc, RECT area) {
    buttons_.clear();

    // Background
    HBRUSH bg = CreateSolidBrush(g_theme.bg_secondary);
    FillRect(hdc, &area, bg);
    DeleteObject(bg);

    // Right border glow
    DrawNeonLine(hdc, area.right - 1, area.top, area.right - 1, area.bottom, g_theme.neon_purple);

    SetBkMode(hdc, TRANSPARENT);

    int y = area.top + g_theme.padding;
    int pad = g_theme.padding;
    int btn_h = 34;
    int left = area.left + pad;
    int right = area.right - pad;

    // ══════════ TITLE ══════════
    HFONT title_font = CreateFontA(g_theme.font_size_title, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                    DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
    HFONT old = (HFONT)SelectObject(hdc, title_font);

    // Glow effect on title
    SetTextColor(hdc, g_theme.glow_cyan);
    TextOutA(hdc, left - 1, y - 1, "NEURALFORGE", 11);
    TextOutA(hdc, left + 1, y + 1, "NEURALFORGE", 11);
    SetTextColor(hdc, g_theme.neon_cyan);
    TextOutA(hdc, left, y, "NEURALFORGE", 11);
    y += g_theme.font_size_title + 4;

    SelectObject(hdc, old);
    DeleteObject(title_font);

    // Version / subtitle
    HFONT ver_font = CreateFontA(12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                  DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
    SelectObject(hdc, ver_font);
    SetTextColor(hdc, g_theme.neon_purple);
    TextOutA(hdc, left, y, "v1.0 \xC4\xC4 LOCAL AI ENGINE", 23);
    y += 18;

    // Status readout
    HFONT status_font = CreateFontA(g_theme.font_size_small, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                     DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
    SelectObject(hdc, status_font);

    COLORREF status_color = model_loaded ? g_theme.neon_green : (data_loaded ? g_theme.neon_orange : g_theme.neon_magenta);
    const char* status_icon = model_loaded ? "\xFE" : (data_loaded ? "\xF0" : "\xF9");  // ■ ≡ ∙

    SetTextColor(hdc, status_color);
    char status_buf[128];
    snprintf(status_buf, sizeof(status_buf), "%s %s", status_icon, model_status.c_str());
    RECT status_r = {left, y, right, y + 36};
    DrawTextA(hdc, status_buf, -1, &status_r, DT_WORDBREAK | DT_LEFT);
    y += 22;

    // ══════════ DIVIDER ══════════
    y += 4;
    DrawNeonLine(hdc, left, y, right, y, g_theme.neon_cyan);
    y += 12;

    // ══════════ ACTION BUTTONS ══════════
    RECT btn_r = {left, y, right, y + btn_h};
    draw_neon_button(hdc, "\xAF NEW CHAT", btn_r, g_theme.neon_cyan, g_theme.bg_tertiary, BTN_NEW_CHAT);
    y += btn_h + 7;

    btn_r = {left, y, right, y + btn_h};
    draw_neon_button(hdc, "\xAF LOAD DATA (.txt)", btn_r, g_theme.neon_blue, g_theme.bg_tertiary, BTN_LOAD_DATA);
    y += btn_h + 7;

    btn_r = {left, y, right, y + btn_h};
    draw_neon_button(hdc, "\xAF LOAD MODEL (.bin)", btn_r, g_theme.neon_purple, g_theme.bg_tertiary, BTN_LOAD_MODEL);
    y += btn_h + 7;

    if (is_training) {
        btn_r = {left, y, right, y + btn_h};
        draw_neon_button(hdc, "!! ABORT TRAINING", btn_r, g_theme.neon_magenta, RGB(40, 5, 20), BTN_STOP);
    } else {
        btn_r = {left, y, right, y + btn_h};
        draw_neon_button(hdc, "\xAF\xAF TRAIN MODEL", btn_r, g_theme.neon_green, RGB(5, 30, 15), BTN_TRAIN);
    }
    y += btn_h + 7;

    btn_r = {left, y, right, y + btn_h};
    draw_neon_button(hdc, "\xAF FINE-TUNE", btn_r, g_theme.neon_orange, g_theme.bg_tertiary, BTN_FINETUNE);
    y += btn_h + 7;

    // Training status readout
    if (!train_status.empty()) {
        HFONT train_font = CreateFontA(13, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                        DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
        SelectObject(hdc, train_font);
        SetTextColor(hdc, is_training ? g_theme.neon_green : g_theme.text_secondary);
        RECT ts_r = {left, y, right, y + 48};
        DrawTextA(hdc, train_status.c_str(), -1, &ts_r, DT_WORDBREAK | DT_LEFT);
        DeleteObject(train_font);
        y += 52;
    }

    // ══════════ DIVIDER ══════════
    y += 2;
    DrawNeonLine(hdc, left, y, right, y, g_theme.neon_magenta);
    y += 12;

    // ══════════ MODEL PRESET ══════════
    HFONT section_font = CreateFontA(g_theme.font_size_label, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                      DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
    SelectObject(hdc, section_font);
    SetTextColor(hdc, g_theme.neon_cyan);
    TextOutA(hdc, left, y, "\xC4 MODEL PRESET", 14);
    y += g_theme.font_size_label + 6;

    const PresetInfo& preset = GetPreset(current_preset);

    // Preset selector: [<] NAME [>]
    int arrow_w = 34;
    RECT prev_r = {left, y, left + arrow_w, y + btn_h};
    draw_neon_button(hdc, "\x11", prev_r, g_theme.neon_cyan, g_theme.bg_tertiary, BTN_PRESET_PREV);

    RECT next_r = {right - arrow_w, y, right, y + btn_h};
    draw_neon_button(hdc, "\x10", next_r, g_theme.neon_cyan, g_theme.bg_tertiary, BTN_PRESET_NEXT);

    // Preset name centered — big and bright
    HFONT preset_font = CreateFontA(20, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                     DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
    SelectObject(hdc, preset_font);
    RECT name_r = {left + arrow_w + 4, y, right - arrow_w - 4, y + btn_h};
    SetTextColor(hdc, g_theme.neon_magenta);
    DrawTextA(hdc, preset.name, -1, &name_r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    DeleteObject(preset_font);
    y += btn_h + 6;

    // Preset description
    HFONT desc_font = CreateFontA(13, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                   DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
    SelectObject(hdc, desc_font);
    SetTextColor(hdc, g_theme.text_secondary);
    TextOutA(hdc, left, y, preset.description, (int)strlen(preset.description));
    y += 16;

    char param_buf[64];
    if (preset.approx_params >= 1000000) {
        snprintf(param_buf, sizeof(param_buf), "~%dM params", (int)(preset.approx_params / 1000000));
    } else {
        snprintf(param_buf, sizeof(param_buf), "~%dK params", (int)(preset.approx_params / 1000));
    }
    draw_stat_line(hdc, "Size: ", param_buf, y, left, g_theme.neon_orange);
    y += 18;
    draw_stat_line(hdc, "RAM:  ", preset.ram_estimate, y, left, g_theme.neon_orange);
    y += 22;

    // ══════════ DIVIDER ══════════
    DrawNeonLine(hdc, left, y, right, y, g_theme.neon_purple);
    y += 12;

    // ══════════ GENERATION SETTINGS ══════════
    SelectObject(hdc, section_font);
    SetTextColor(hdc, g_theme.neon_cyan);
    TextOutA(hdc, left, y, "\xC4 GENERATION", 12);
    y += g_theme.font_size_label + 6;

    char val_buf[32];

    snprintf(val_buf, sizeof(val_buf), "%.2f", sampler_config.temperature);
    draw_stat_line(hdc, "Temp:     ", val_buf, y, left, g_theme.neon_green); y += 18;

    snprintf(val_buf, sizeof(val_buf), "%d", sampler_config.top_k);
    draw_stat_line(hdc, "Top-K:    ", val_buf, y, left, g_theme.neon_green); y += 18;

    snprintf(val_buf, sizeof(val_buf), "%.2f", sampler_config.top_p);
    draw_stat_line(hdc, "Top-P:    ", val_buf, y, left, g_theme.neon_green); y += 18;

    snprintf(val_buf, sizeof(val_buf), "%.2f", sampler_config.repetition_penalty);
    draw_stat_line(hdc, "Rep Pen:  ", val_buf, y, left, g_theme.neon_green); y += 22;

    // ══════════ DIVIDER ══════════
    DrawNeonLine(hdc, left, y, right, y, g_theme.neon_purple);
    y += 12;

    // ══════════ ARCHITECTURE ══════════
    SelectObject(hdc, section_font);
    SetTextColor(hdc, g_theme.neon_cyan);
    TextOutA(hdc, left, y, "\xC4 ARCHITECTURE", 14);
    y += g_theme.font_size_label + 6;

    snprintf(val_buf, sizeof(val_buf), "%d", model_config.n_layers);
    draw_stat_line(hdc, "Layers:   ", val_buf, y, left, g_theme.neon_purple); y += 18;

    snprintf(val_buf, sizeof(val_buf), "%d", model_config.d_model);
    draw_stat_line(hdc, "d_model:  ", val_buf, y, left, g_theme.neon_purple); y += 18;

    snprintf(val_buf, sizeof(val_buf), "%d", model_config.n_heads);
    draw_stat_line(hdc, "Heads:    ", val_buf, y, left, g_theme.neon_purple); y += 18;

    snprintf(val_buf, sizeof(val_buf), "%d", model_config.d_ff);
    draw_stat_line(hdc, "d_ff:     ", val_buf, y, left, g_theme.neon_purple); y += 18;

    snprintf(val_buf, sizeof(val_buf), "%d", model_config.vocab_size);
    draw_stat_line(hdc, "Vocab:    ", val_buf, y, left, g_theme.neon_purple); y += 18;

    snprintf(val_buf, sizeof(val_buf), "%d", model_config.max_seq_len);
    draw_stat_line(hdc, "Max Seq:  ", val_buf, y, left, g_theme.neon_purple); y += 18;

    if (param_count > 0) {
        if (param_count >= 1000000) {
            snprintf(val_buf, sizeof(val_buf), "%dM", (int)(param_count / 1000000));
        } else {
            snprintf(val_buf, sizeof(val_buf), "%dK", (int)(param_count / 1000));
        }
        draw_stat_line(hdc, "Params:   ", val_buf, y, left, g_theme.neon_magenta);
    }

    SelectObject(hdc, old);
    DeleteObject(title_font);
    DeleteObject(ver_font);
    DeleteObject(status_font);
    DeleteObject(section_font);
    DeleteObject(desc_font);
}

int Sidebar::handle_click(int x, int y, RECT area) {
    POINT pt = {x, y};
    for (auto& btn : buttons_) {
        if (PtInRect(&btn.rect, pt)) {
            return btn.id;
        }
    }
    return BTN_NONE;
}

} // namespace nf
