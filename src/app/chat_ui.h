#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <string>
#include <vector>

namespace nf {

struct ChatMessage {
    std::string text;
    bool is_user;
    bool is_streaming = false;
};

class ChatUI {
public:
    std::vector<ChatMessage> messages;
    int scroll_offset = 0;
    int max_scroll = 0;

    void add_message(const std::string& text, bool is_user);
    void append_to_last(const std::string& text);
    void start_ai_message();
    void clear();

    void paint(HDC hdc, RECT area);
    void scroll(int delta);

private:
    void draw_message(HDC hdc, const ChatMessage& msg, RECT& area, int& y_pos);
    int measure_text_height(HDC hdc, const std::string& text, int width);
};

} // namespace nf
