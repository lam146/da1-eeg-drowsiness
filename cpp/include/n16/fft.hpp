#pragma once

#include <complex>
#include <vector>

namespace n16 {

void fft(std::vector<std::complex<double>>& values, bool inverse = false);

}  // namespace n16
