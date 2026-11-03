#pragma once

#include <cstddef>
#include <deque>
#include <optional>

#include "n16/model.hpp"
#include "n16/types.hpp"

namespace n16 {

class LiveDetector {
public:
    LiveDetector(
        LogisticModel model,
        int fs = 512,
        double window_s = 4.0,
        double step_s = 1.0,
        int consecutive = 2);

    std::optional<Decision> add_sample(double sample, double poor_signal, long long timestamp_ms);

private:
    LogisticModel model_;
    int fs_;
    std::size_t window_samples_;
    std::size_t step_samples_;
    int consecutive_;
    std::deque<double> samples_;
    std::deque<double> poor_signal_;
    std::size_t samples_since_decision_{};
    int alert_streak_{};
};

}  // namespace n16
