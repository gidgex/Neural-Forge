#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace nf {

struct Theme {
    // === HYPERLIGHT QUANTUM COLOR SCHEME ===
    // Deep void backgrounds with neon energy accents

    // Backgrounds - deep space void with subtle blue/purple undertones
    COLORREF bg_primary    = RGB(8, 8, 16);         // Abyss black-blue
    COLORREF bg_secondary  = RGB(12, 10, 24);       // Sidebar - dark indigo void
    COLORREF bg_tertiary   = RGB(16, 14, 32);       // AI message bubble - deep purple-black
    COLORREF bg_user       = RGB(20, 12, 40);       // User message - dark magenta-tinted
    COLORREF bg_input      = RGB(14, 12, 28);       // Input field - dark
    COLORREF bg_button     = RGB(0, 180, 220);      // Electric cyan button
    COLORREF bg_button_hover = RGB(0, 220, 255);    // Bright cyan hover

    // Neon text colors
    COLORREF text_primary  = RGB(200, 210, 230);    // Cool white with blue tint
    COLORREF text_secondary = RGB(100, 110, 140);   // Muted blue-gray
    COLORREF text_accent   = RGB(0, 255, 255);      // Electric cyan

    // Neon accent palette
    COLORREF neon_cyan     = RGB(0, 255, 255);      // Primary neon
    COLORREF neon_magenta  = RGB(255, 0, 180);      // Hot magenta
    COLORREF neon_purple   = RGB(160, 60, 255);     // Electric purple
    COLORREF neon_pink     = RGB(255, 80, 200);     // Soft neon pink
    COLORREF neon_blue     = RGB(40, 120, 255);     // Deep neon blue
    COLORREF neon_green    = RGB(0, 255, 160);      // Quantum green
    COLORREF neon_orange   = RGB(255, 140, 0);      // Warm energy

    // Glow colors (dimmer versions for outer glow effects)
    COLORREF glow_cyan     = RGB(0, 80, 100);
    COLORREF glow_magenta  = RGB(80, 0, 60);
    COLORREF glow_purple   = RGB(50, 20, 80);

    // Structure
    COLORREF border        = RGB(30, 25, 55);       // Subtle purple border
    COLORREF border_glow   = RGB(0, 180, 220);      // Neon border highlight
    COLORREF scrollbar     = RGB(0, 120, 150);      // Cyan scrollbar

    int padding = 14;
    int msg_padding = 18;
    int border_radius = 8;
    int font_size = 18;
    int font_size_small = 14;
    int font_size_label = 16;
    int font_size_header = 17;
    int font_size_title = 26;
    int sidebar_width = 310;
    int input_height = 100;
    int header_height = 56;
};

extern Theme g_theme;

// GDI helper: draw a "neon glow" rectangle (multiple offset rects with decreasing alpha)
void DrawNeonRect(HDC hdc, RECT r, COLORREF glow_color, COLORREF border_color, int radius = 8);
void DrawNeonLine(HDC hdc, int x1, int y1, int x2, int y2, COLORREF color);
COLORREF BlendColor(COLORREF c1, COLORREF c2, float t);

} // namespace nf
