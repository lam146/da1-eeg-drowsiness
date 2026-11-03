#pragma once

#include <vector>

#include "n16/types.hpp"

namespace n16 {

std::vector<double> preprocess_window(
    const std::vector<double>& samples,
    double fs = 512.0,
    double low_hz = 0.5,
    double high_hz = 45.0,
    double notch_hz = 50.0,
    double notch_width_hz = 1.0);

FeatureMap spectral_features(const std::vector<double>& samples, double fs = 512.0);

}  // namespace n16
