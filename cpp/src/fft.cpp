#include "n16/fft.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace n16 {

void fft(std::vector<std::complex<double>>& values, bool inverse) {
    const std::size_t n = values.size();
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("FFT size must be a power of two");
    }

    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }

    constexpr double pi = 3.14159265358979323846;
    for (std::size_t length = 2; length <= n; length <<= 1) {
        const double angle = 2.0 * pi / static_cast<double>(length) * (inverse ? 1.0 : -1.0);
        const std::complex<double> root(std::cos(angle), std::sin(angle));
        for (std::size_t start = 0; start < n; start += length) {
            std::complex<double> factor(1.0, 0.0);
            for (std::size_t offset = 0; offset < length / 2; ++offset) {
                const auto even = values[start + offset];
                const auto odd = values[start + offset + length / 2] * factor;
                values[start + offset] = even + odd;
                values[start + offset + length / 2] = even - odd;
                factor *= root;
            }
        }
    }

    if (inverse) {
        for (auto& value : values) {
            value /= static_cast<double>(n);
        }
    }
}

}  // namespace n16
