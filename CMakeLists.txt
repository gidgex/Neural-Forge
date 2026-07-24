cmake_minimum_required(VERSION 3.16)
project(NeuralForge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core library (shared between trainer and GUI app)
add_library(neuralforge_core STATIC
    src/core/tensor.cpp
    src/core/random.cpp
    src/tokenizer/tokenizer.cpp
    src/model/embedding.cpp
    src/model/attention.cpp
    src/model/feedforward.cpp
    src/model/transformer.cpp
    src/model/lm_head.cpp
    src/training/optimizer.cpp
    src/training/dataset.cpp
    src/training/trainer.cpp
    src/training/checkpoint.cpp
    src/inference/sampler.cpp
    src/inference/generator.cpp
)
target_include_directories(neuralforge_core PUBLIC src)

# CLI Trainer
add_executable(ai_trainer src/trainer_main.cpp)
target_link_libraries(ai_trainer PRIVATE neuralforge_core)

# Desktop GUI App
if(WIN32)
    add_executable(ai_assistant WIN32
        src/app/main.cpp
        src/app/window.cpp
        src/app/chat_ui.cpp
        src/app/sidebar.cpp
        src/app/theme.cpp
    )
    target_link_libraries(ai_assistant PRIVATE neuralforge_core user32 gdi32 gdiplus comctl32)
endif()
