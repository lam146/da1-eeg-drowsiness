#pragma once

#include <vector>

#include "n16/types.hpp"

namespace n16 {

struct BlinkEvent {
    std::size_t sample_index{};
    double time_seconds{};
    double amplitude{};
    double width_seconds{};
};

std::vector<BlinkEvent> detect_blinks(
    const std::vector<double>& samples,
    double fs = 512.0,
    double threshold_mad = 6.0,
    double refractory_s = 0.50,
    double min_width_s = 0.08,
    double max_width_s = 0.60);

FeatureMap blink_features(const std::vector<double>& samples, double fs = 512.0);

}  // namespace n16
