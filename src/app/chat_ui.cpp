#include "app/chat_ui.h"
#include "app/theme.h"
#include <algorithm>
#include <cmath>

namespace nf {

void ChatUI::add_message(const std::string& text, bool is_user) {
    messages.push_back({text, is_user, false});
}

void ChatUI::append_to_last(const std::string& text) {
    if (!messages.empty()) {
        messages.back().text += text;
    }
}

void ChatUI::start_ai_message() {
    messages.push_back({"", false, true});
}

void ChatUI::clear() {
    messages.clear();
    scroll_offset = 0;
}

int ChatUI::measure_text_height(HDC hdc, const std::string& text, int width) {
    if (text.empty()) return g_theme.font_size + 10;
    RECT r = {0, 0, width, 0};
    DrawTextA(hdc, text.c_str(), -1, &r, DT_CALCRECT | DT_WORDBREAK | DT_LEFT);
    return r.bottom + 10;
}

void ChatUI::draw_message(HDC hdc, const ChatMessage& msg, RECT& area, int& y_pos) {
    int bubble_margin = 50;
    int max_bubble_width = (area.right - area.left) - bubble_margin - g_theme.msg_padding * 2;
    int text_width = max_bubble_width - g_theme.msg_padding * 2;

    int text_height = measure_text_height(hdc, msg.text, text_width);
    int bubble_height = text_height + g_theme.msg_padding * 2;

    // Thematic labels — styled like terminal readouts
    const char* label = msg.is_user
        ? "\xC4\xC4 OPERATOR"           // ── OPERATOR
        : "\xC4\xC4 NEURALFORGE ::";    // ── NEURALFORGE ::
    COLORREF label_color = msg.is_user ? g_theme.neon_cyan : g_theme.neon_magenta;
    COLORREF bubble_glow = msg.is_user ? g_theme.glow_cyan : g_theme.glow_magenta;
    COLORREF bubble_border = msg.is_user ? g_theme.neon_cyan : g_theme.neon_magenta;

    int label_x = msg.is_user ? (area.right - max_bubble_width - g_theme.padding) : (area.left + g_theme.padding);

    // Label — bigger and bolder
    SetTextColor(hdc, label_color);
    HFONT label_font = CreateFontA(g_theme.font_size_label, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                    DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
    HFONT old_font = (HFONT)SelectObject(hdc, label_font);
    TextOutA(hdc, label_x, y_pos, label, (int)strlen(label));
    SelectObject(hdc, old_font);
    DeleteObject(label_font);
    y_pos += g_theme.font_size_label + 6;

    // Bubble position
    RECT bubble;
    if (msg.is_user) {
        bubble.right = area.right - g_theme.padding;
        bubble.left = bubble.right - max_bubble_width;
    } else {
        bubble.left = area.left + g_theme.padding;
        bubble.right = bubble.left + max_bubble_width;
    }
    bubble.top = y_pos;
    bubble.bottom = y_pos + bubble_height;

    // Neon glow border
    DrawNeonRect(hdc, bubble, bubble_glow, bubble_border, g_theme.border_radius);

    // Fill bubble
    COLORREF bg = msg.is_user ? g_theme.bg_user : g_theme.bg_tertiary;
    HBRUSH brush = CreateSolidBrush(bg);
    HPEN pen = CreatePen(PS_SOLID, 1, bubble_border);
    SelectObject(hdc, brush);
    SelectObject(hdc, pen);
    RoundRect(hdc, bubble.left, bubble.top, bubble.right, bubble.bottom,
              g_theme.border_radius, g_theme.border_radius);
    DeleteObject(brush);
    DeleteObject(pen);

    // Neon accent bar on edge
    if (msg.is_user) {
        HPEN accent = CreatePen(PS_SOLID, 3, g_theme.neon_cyan);
        SelectObject(hdc, accent);
        MoveToEx(hdc, bubble.right - 3, bubble.top + 6, NULL);
        LineTo(hdc, bubble.right - 3, bubble.bottom - 6);
        DeleteObject(accent);
    } else {
        HPEN accent = CreatePen(PS_SOLID, 3, g_theme.neon_magenta);
        SelectObject(hdc, accent);
        MoveToEx(hdc, bubble.left + 3, bubble.top + 6, NULL);
        LineTo(hdc, bubble.left + 3, bubble.bottom - 6);
        DeleteObject(accent);
    }

    // Message text — main readable font
    RECT text_rect = {
        bubble.left + g_theme.msg_padding + 4,
        bubble.top + g_theme.msg_padding,
        bubble.right - g_theme.msg_padding,
        bubble.bottom - g_theme.msg_padding
    };

    HFONT text_font = CreateFontA(g_theme.font_size, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                   DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
    old_font = (HFONT)SelectObject(hdc, text_font);

    SetTextColor(hdc, g_theme.text_primary);
    SetBkMode(hdc, TRANSPARENT);

    std::string display_text = msg.text;
    if (msg.is_streaming && msg.text.empty()) {
        display_text = "\xB0\xB1\xB2 GENERATING \xB2\xB1\xB0";  // ░▒▓ GENERATING ▓▒░
        SetTextColor(hdc, g_theme.neon_purple);
    }

    DrawTextA(hdc, display_text.c_str(), -1, &text_rect, DT_WORDBREAK | DT_LEFT);

    SelectObject(hdc, old_font);
    DeleteObject(text_font);

    y_pos = bubble.bottom + g_theme.padding + 6;
}

void ChatUI::paint(HDC hdc, RECT area) {
    // Background — abyss void
    HBRUSH bg = CreateSolidBrush(g_theme.bg_primary);
    FillRect(hdc, &area, bg);
    DeleteObject(bg);

    // Quantum grid overlay
    HPEN grid_pen = CreatePen(PS_SOLID, 1, RGB(14, 12, 26));
    HPEN old_pen = (HPEN)SelectObject(hdc, grid_pen);
    for (int y = area.top; y < area.bottom; y += 48) {
        MoveToEx(hdc, area.left, y, NULL);
        LineTo(hdc, area.right, y);
    }
    for (int x = area.left; x < area.right; x += 48) {
        MoveToEx(hdc, x, area.top, NULL);
        LineTo(hdc, x, area.bottom);
    }
    SelectObject(hdc, old_pen);
    DeleteObject(grid_pen);

    if (messages.empty()) {
        SetBkMode(hdc, TRANSPARENT);

        // === Huge glowing title ===
        HFONT title_font = CreateFontA(48, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                        DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
        HFONT old = (HFONT)SelectObject(hdc, title_font);

        const char* title = "NEURALFORGE";
        RECT tr = area;
        tr.top = area.top + (area.bottom - area.top) / 3 - 30;

        // Glow layers
        for (int i = 4; i >= 1; i--) {
            COLORREF glow = BlendColor(g_theme.neon_cyan, RGB(0, 0, 0), (float)i / 5.0f);
            SetTextColor(hdc, glow);
            RECT gtr = tr;
            gtr.top -= i; gtr.left -= i;
            DrawTextA(hdc, title, -1, &gtr, DT_CENTER | DT_SINGLELINE);
            gtr = tr;
            gtr.top += i; gtr.left += i;
            DrawTextA(hdc, title, -1, &gtr, DT_CENTER | DT_SINGLELINE);
        }
        SetTextColor(hdc, g_theme.neon_cyan);
        DrawTextA(hdc, title, -1, &tr, DT_CENTER | DT_SINGLELINE);

        SelectObject(hdc, old);
        DeleteObject(title_font);

        // Subtitle line 1 — quantum tagline
        HFONT sub_font = CreateFontA(17, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                      DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
        old = (HFONT)SelectObject(hdc, sub_font);
        SetTextColor(hdc, g_theme.neon_purple);
        tr.top += 60;
        const char* sub = "\xC4\xC4\xC4 QUANTUM NEURAL ARCHITECTURE \xC4\xC4\xC4";
        DrawTextA(hdc, sub, -1, &tr, DT_CENTER | DT_SINGLELINE);

        // Subtitle line 2 — workflow hint
        tr.top += 28;
        SetTextColor(hdc, g_theme.text_secondary);
        HFONT hint_font = CreateFontA(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                       DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
        SelectObject(hdc, hint_font);
        const char* sub2 = "Load data  \xAF  Train model  \xAF  Chat with your AI";
        DrawTextA(hdc, sub2, -1, &tr, DT_CENTER | DT_SINGLELINE);

        // Decorative neon line
        int cx = (area.left + area.right) / 2;
        tr.top += 36;
        DrawNeonLine(hdc, cx - 160, tr.top, cx + 160, tr.top, g_theme.neon_magenta);

        // Version badge
        tr.top += 18;
        SetTextColor(hdc, RGB(60, 55, 90));
        HFONT tiny_font = CreateFontA(13, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                       DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
        SelectObject(hdc, tiny_font);
        DrawTextA(hdc, "v1.0  //  100% LOCAL  //  ZERO DEPENDENCIES  //  PURE C++", -1, &tr, DT_CENTER | DT_SINGLELINE);

        SelectObject(hdc, old);
        DeleteObject(sub_font);
        DeleteObject(hint_font);
        DeleteObject(tiny_font);
        return;
    }

    // Clipping region for message scroll
    HRGN clip = CreateRectRgn(area.left, area.top, area.right, area.bottom);
    SelectClipRgn(hdc, clip);

    int y = area.top + g_theme.padding - scroll_offset;

    for (auto& msg : messages) {
        if (y > area.bottom + 300) break;
        if (y + 300 >= area.top) {
            draw_message(hdc, msg, area, y);
        } else {
            // Estimate height for offscreen skip
            HFONT measure_font = CreateFontA(g_theme.font_size, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                              DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, "Consolas");
            HFONT oldf = (HFONT)SelectObject(hdc, measure_font);
            int text_height = measure_text_height(hdc, msg.text,
                (area.right - area.left) - 50 - g_theme.msg_padding * 4);
            y += text_height + g_theme.msg_padding * 2 + g_theme.padding + g_theme.font_size_label + 16;
            SelectObject(hdc, oldf);
            DeleteObject(measure_font);
        }
    }

    max_scroll = std::max(0, (int)(y + scroll_offset - (area.bottom - area.top)));

    SelectClipRgn(hdc, NULL);
    DeleteObject(clip);
}

void ChatUI::scroll(int delta) {
    scroll_offset -= delta * 60;
    scroll_offset = std::max(0, std::min(scroll_offset, max_scroll));
}

} // namespace nf
