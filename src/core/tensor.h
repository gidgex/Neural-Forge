#pragma once
#include <vector>
#include <cstddef>
#include <cassert>
#include <string>
#include <functional>
#include <cmath>
#include <iostream>

namespace nf {

class Tensor {
public:
    std::vector<float> data;
    std::vector<float> grad;
    size_t rows = 0;
    size_t cols = 0;
    bool requires_grad = false;

    Tensor() = default;
    Tensor(size_t rows, size_t cols, bool requires_grad = false);
    Tensor(size_t rows, size_t cols, float fill_val);

    static Tensor zeros(size_t rows, size_t cols, bool requires_grad = false);
    static Tensor ones(size_t rows, size_t cols);
    static Tensor randn(size_t rows, size_t cols, float mean = 0.0f, float std = 1.0f, bool requires_grad = false);

    float& at(size_t r, size_t c);
    float at(size_t r, size_t c) const;
    float& grad_at(size_t r, size_t c);
    float grad_at(size_t r, size_t c) const;

    size_t size() const { return data.size(); }
    void zero_grad();
    void fill(float val);

    // Element-wise operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const; // element-wise
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);

    // Matrix operations
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;

    // Activation / math
    Tensor softmax(int axis = -1) const; // axis=-1 means last axis (cols)
    Tensor gelu() const;
    Tensor layer_norm(float eps = 1e-5f) const;
    Tensor relu() const;

    // Reduction
    float sum() const;
    float mean() const;
    float max_val() const;

    // Row/col operations
    Tensor row(size_t r) const;
    void set_row(size_t r, const Tensor& row_data);
    Tensor slice_rows(size_t start, size_t end) const;

    // Utility
    void print(const std::string& name = "") const;
    Tensor clone() const;
    void resize(size_t new_rows, size_t new_cols);
};

// Non-member operators
Tensor operator*(float scalar, const Tensor& t);

} // namespace nf
