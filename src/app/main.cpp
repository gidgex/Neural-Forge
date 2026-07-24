#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "app/window.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    nf::MainWindow window;
    if (!window.create(hInstance, nCmdShow)) {
        MessageBoxA(NULL, "Failed to create window", "NeuralForge AI", MB_ICONERROR);
        return 1;
    }
    return window.run();
}
