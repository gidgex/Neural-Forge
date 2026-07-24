#pragma once
#include <random>
#include <cstdint>

namespace nf {

class Random {
public:
    static Random& instance() {
        static Random r;
        return r;
    }

    void seed(uint64_t s) { gen_.seed(s); }

    float uniform(float lo = 0.0f, float hi = 1.0f) {
        std::uniform_real_distribution<float> dist(lo, hi);
        return dist(gen_);
    }

    float normal(float mean = 0.0f, float std = 1.0f) {
        std::normal_distribution<float> dist(mean, std);
        return dist(gen_);
    }

    int randint(int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(gen_);
    }

    std::mt19937& engine() { return gen_; }

private:
    Random() : gen_(42) {}
    std::mt19937 gen_;
};

} // namespace nf
