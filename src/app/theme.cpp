#include "app/theme.h"
#include <algorithm>

namespace nf {

Theme g_theme;

COLORREF BlendColor(COLORREF c1, COLORREF c2, float t) {
    int r = (int)(GetRValue(c1) * (1 - t) + GetRValue(c2) * t);
    int g = (int)(GetGValue(c1) * (1 - t) + GetGValue(c2) * t);
    int b = (int)(GetBValue(c1) * (1 - t) + GetBValue(c2) * t);
    return RGB(std::min(255, std::max(0, r)), std::min(255, std::max(0, g)), std::min(255, std::max(0, b)));
}

void DrawNeonRect(HDC hdc, RECT r, COLORREF glow_color, COLORREF border_color, int radius) {
    // Outer glow layers (3 passes, expanding outward)
    for (int i = 3; i >= 1; i--) {
        RECT glow_r = {r.left - i * 2, r.top - i * 2, r.right + i * 2, r.bottom + i * 2};
        float fade = (float)i / 4.0f;
        COLORREF dimmed = BlendColor(glow_color, RGB(0, 0, 0), fade);
        HPEN pen = CreatePen(PS_SOLID, 1, dimmed);
        HBRUSH old_brush = (HBRUSH)SelectObject(hdc, GetStockObject(NULL_BRUSH));
        HPEN old_pen = (HPEN)SelectObject(hdc, pen);
        RoundRect(hdc, glow_r.left, glow_r.top, glow_r.right, glow_r.bottom, radius + i, radius + i);
        SelectObject(hdc, old_pen);
        SelectObject(hdc, old_brush);
        DeleteObject(pen);
    }

    // Main border
    HPEN pen = CreatePen(PS_SOLID, 1, border_color);
    HPEN old_pen = (HPEN)SelectObject(hdc, pen);
    HBRUSH old_brush = (HBRUSH)SelectObject(hdc, GetStockObject(NULL_BRUSH));
    RoundRect(hdc, r.left, r.top, r.right, r.bottom, radius, radius);
    SelectObject(hdc, old_pen);
    SelectObject(hdc, old_brush);
    DeleteObject(pen);
}

void DrawNeonLine(HDC hdc, int x1, int y1, int x2, int y2, COLORREF color) {
    // Glow layers
    for (int i = 3; i >= 1; i--) {
        float fade = (float)i / 4.0f;
        COLORREF dimmed = BlendColor(color, RGB(0, 0, 0), fade);
        HPEN pen = CreatePen(PS_SOLID, 1, dimmed);
        HPEN old = (HPEN)SelectObject(hdc, pen);
        MoveToEx(hdc, x1, y1 - i, NULL); LineTo(hdc, x2, y2 - i);
        MoveToEx(hdc, x1, y1 + i, NULL); LineTo(hdc, x2, y2 + i);
        SelectObject(hdc, old);
        DeleteObject(pen);
    }
    // Core line
    HPEN pen = CreatePen(PS_SOLID, 1, color);
    HPEN old = (HPEN)SelectObject(hdc, pen);
    MoveToEx(hdc, x1, y1, NULL);
    LineTo(hdc, x2, y2);
    SelectObject(hdc, old);
    DeleteObject(pen);
}

} // namespace nf
