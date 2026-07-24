#include "core/tensor.h"
#include "core/random.h"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <stdexcept>

namespace nf {

Tensor::Tensor(size_t rows, size_t cols, bool requires_grad)
    : rows(rows), cols(cols), requires_grad(requires_grad) {
    data.resize(rows * cols, 0.0f);
    if (requires_grad) grad.resize(rows * cols, 0.0f);
}

Tensor::Tensor(size_t rows, size_t cols, float fill_val)
    : rows(rows), cols(cols), requires_grad(false) {
    data.resize(rows * cols, fill_val);
}

Tensor Tensor::zeros(size_t rows, size_t cols, bool requires_grad) {
    return Tensor(rows, cols, requires_grad);
}

Tensor Tensor::ones(size_t rows, size_t cols) {
    return Tensor(rows, cols, 1.0f);
}

Tensor Tensor::randn(size_t rows, size_t cols, float mean, float std, bool requires_grad) {
    Tensor t(rows, cols, requires_grad);
    auto& rng = Random::instance();
    for (auto& v : t.data) {
        v = rng.normal(mean, std);
    }
    return t;
}

float& Tensor::at(size_t r, size_t c) {
    return data[r * cols + c];
}

float Tensor::at(size_t r, size_t c) const {
    return data[r * cols + c];
}

float& Tensor::grad_at(size_t r, size_t c) {
    return grad[r * cols + c];
}

float Tensor::grad_at(size_t r, size_t c) const {
    return grad[r * cols + c];
}

void Tensor::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

void Tensor::fill(float val) {
    std::fill(data.begin(), data.end(), val);
}

Tensor Tensor::operator+(const Tensor& other) const {
    assert(rows == other.rows && cols == other.cols);
    Tensor result(rows, cols);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    assert(rows == other.rows && cols == other.cols);
    Tensor result(rows, cols);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    assert(rows == other.rows && cols == other.cols);
    Tensor result(rows, cols);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(rows, cols);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator/(float scalar) const {
    Tensor result(rows, cols);
    float inv = 1.0f / scalar;
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] * inv;
    }
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    assert(rows == other.rows && cols == other.cols);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] += other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    assert(rows == other.rows && cols == other.cols);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] -= other.data[i];
    }
    return *this;
}

Tensor Tensor::matmul(const Tensor& other) const {
    assert(cols == other.rows);
    Tensor result(rows, other.cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t k = 0; k < cols; k++) {
            float a_ik = data[i * cols + k];
            if (a_ik == 0.0f) continue;
            for (size_t j = 0; j < other.cols; j++) {
                result.data[i * other.cols + j] += a_ik * other.data[k * other.cols + j];
            }
        }
    }
    return result;
}

Tensor Tensor::transpose() const {
    Tensor result(cols, rows);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[j * rows + i] = data[i * cols + j];
        }
    }
    return result;
}

Tensor Tensor::softmax(int axis) const {
    Tensor result(rows, cols);
    if (axis == -1 || axis == 1) {
        // Softmax along columns (each row independently)
        for (size_t i = 0; i < rows; i++) {
            float max_val = data[i * cols];
            for (size_t j = 1; j < cols; j++) {
                max_val = std::max(max_val, data[i * cols + j]);
            }
            float sum = 0.0f;
            for (size_t j = 0; j < cols; j++) {
                result.data[i * cols + j] = std::exp(data[i * cols + j] - max_val);
                sum += result.data[i * cols + j];
            }
            float inv_sum = 1.0f / (sum + 1e-10f);
            for (size_t j = 0; j < cols; j++) {
                result.data[i * cols + j] *= inv_sum;
            }
        }
    }
    return result;
}

Tensor Tensor::gelu() const {
    Tensor result(rows, cols);
    const float sqrt_2_over_pi = 0.7978845608f;
    for (size_t i = 0; i < data.size(); i++) {
        float x = data[i];
        float cdf = 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
        result.data[i] = x * cdf;
    }
    return result;
}

Tensor Tensor::layer_norm(float eps) const {
    Tensor result(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        // Compute mean
        float mean = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            mean += data[i * cols + j];
        }
        mean /= static_cast<float>(cols);

        // Compute variance
        float var = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float diff = data[i * cols + j] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(cols);

        // Normalize
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (size_t j = 0; j < cols; j++) {
            result.data[i * cols + j] = (data[i * cols + j] - mean) * inv_std;
        }
    }
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(rows, cols);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = std::max(0.0f, data[i]);
    }
    return result;
}

float Tensor::sum() const {
    return std::accumulate(data.begin(), data.end(), 0.0f);
}

float Tensor::mean() const {
    return sum() / static_cast<float>(data.size());
}

float Tensor::max_val() const {
    return *std::max_element(data.begin(), data.end());
}

Tensor Tensor::row(size_t r) const {
    Tensor result(1, cols);
    std::memcpy(result.data.data(), &data[r * cols], cols * sizeof(float));
    return result;
}

void Tensor::set_row(size_t r, const Tensor& row_data) {
    assert(row_data.cols == cols && row_data.rows == 1);
    std::memcpy(&data[r * cols], row_data.data.data(), cols * sizeof(float));
}

Tensor Tensor::slice_rows(size_t start, size_t end) const {
    assert(end > start && end <= rows);
    Tensor result(end - start, cols);
    std::memcpy(result.data.data(), &data[start * cols], (end - start) * cols * sizeof(float));
    return result;
}

void Tensor::print(const std::string& name) const {
    if (!name.empty()) std::cout << name << " ";
    std::cout << "[" << rows << "x" << cols << "]:\n";
    for (size_t i = 0; i < std::min(rows, (size_t)8); i++) {
        std::cout << "  [";
        for (size_t j = 0; j < std::min(cols, (size_t)8); j++) {
            printf("%8.4f", data[i * cols + j]);
            if (j < cols - 1) std::cout << ", ";
        }
        if (cols > 8) std::cout << " ...";
        std::cout << "]\n";
    }
    if (rows > 8) std::cout << "  ...\n";
}

Tensor Tensor::clone() const {
    Tensor result(rows, cols, requires_grad);
    result.data = data;
    result.grad = grad;
    return result;
}

void Tensor::resize(size_t new_rows, size_t new_cols) {
    rows = new_rows;
    cols = new_cols;
    data.resize(rows * cols, 0.0f);
    if (requires_grad) grad.resize(rows * cols, 0.0f);
}

Tensor operator*(float scalar, const Tensor& t) {
    return t * scalar;
}

} // namespace nf
