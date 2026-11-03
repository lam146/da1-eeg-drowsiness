#include "n16/blink.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace n16 {
namespace {

std::vector<double> moving_average(const std::vector<double>& values, std::size_t width) {
    width = std::max<std::size_t>(1, width);
    if (width == 1) {
        return values;
    }
    std::vector<double> result(values.size(), 0.0);
    // Match NumPy's `convolve(..., mode="same")` alignment for even kernels.
    const std::size_t left_padding = width / 2;
    for (std::size_t i = 0; i < values.size(); ++i) {
        double sum = 0.0;
        for (std::size_t k = 0; k < width; ++k) {
            const long long source = static_cast<long long>(i) + static_cast<long long>(k)
                - static_cast<long long>(left_padding);
            if (source >= 0 && source < static_cast<long long>(values.size())) {
                sum += values[static_cast<std::size_t>(source)];
            }
        }
        result[i] = sum / static_cast<double>(width);
    }
    return result;
}

double median(std::vector<double> values) {
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    const double upper = *middle;
    if (values.size() % 2 != 0) {
        return upper;
    }
    return (*std::max_element(values.begin(), middle) + upper) / 2.0;
}

}  // namespace

std::vector<BlinkEvent> detect_blinks(
    const std::vector<double>& samples,
    double fs,
    double threshold_mad,
    double refractory_s,
    double min_width_s,
    double max_width_s) {
    if (samples.size() < static_cast<std::size_t>(fs)) {
        return {};
    }
    const auto baseline = moving_average(samples, static_cast<std::size_t>(0.8 * fs));
    std::vector<double> centered(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        centered[i] = samples[i] - baseline[i];
    }
    const auto smooth = moving_average(centered, static_cast<std::size_t>(0.04 * fs));
    const double center = median(smooth);
    std::vector<double> deviations(smooth.size());
    std::transform(smooth.begin(), smooth.end(), deviations.begin(), [center](double value) {
        return std::abs(value - center);
    });
    const double sigma = std::max(1.4826 * median(deviations), std::numeric_limits<double>::epsilon());
    std::vector<double> score(smooth.size());
    std::transform(smooth.begin(), smooth.end(), score.begin(), [center, sigma](double value) {
        return std::abs(value - center) / sigma;
    });

    std::vector<std::size_t> candidates;
    for (std::size_t i = 1; i + 1 < score.size(); ++i) {
        if (score[i] >= score[i - 1] && score[i] > score[i + 1] && score[i] >= threshold_mad) {
            candidates.push_back(i);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [&score](std::size_t left, std::size_t right) {
        return score[left] > score[right];
    });
    const auto refractory = std::max<std::size_t>(1, static_cast<std::size_t>(refractory_s * fs));
    const double low_threshold = threshold_mad * 0.35;
    std::vector<std::size_t> selected;
    std::vector<BlinkEvent> events;
    for (const auto index : candidates) {
        bool too_close = false;
        for (const auto previous : selected) {
            const auto distance = index > previous ? index - previous : previous - index;
            if (distance < refractory) {
                too_close = true;
                break;
            }
        }
        if (too_close) {
            continue;
        }
        std::size_t left = index;
        while (left > 0 && score[left] > low_threshold) {
            --left;
        }
        std::size_t right = index;
        while (right + 1 < score.size() && score[right] > low_threshold) {
            ++right;
        }
        const double width = static_cast<double>(right - left) / fs;
        if (width >= min_width_s && width <= max_width_s) {
            events.push_back({index, static_cast<double>(index) / fs, smooth[index], width});
            selected.push_back(index);
        }
    }
    std::sort(events.begin(), events.end(), [](const BlinkEvent& left, const BlinkEvent& right) {
        return left.sample_index < right.sample_index;
    });
    return events;
}

FeatureMap blink_features(const std::vector<double>& samples, double fs) {
    const auto events = detect_blinks(samples, fs);
    const double duration_minutes = static_cast<double>(samples.size()) / fs / 60.0;
    double amplitude_sum = 0.0;
    double width_sum = 0.0;
    int long_count = 0;
    for (const auto& event : events) {
        amplitude_sum += std::abs(event.amplitude);
        width_sum += event.width_seconds;
        if (event.width_seconds >= 0.35) {
            ++long_count;
        }
    }
    const double count = static_cast<double>(events.size());
    return {
        {"blink_count", count},
        {"blink_rate_per_min", count / std::max(duration_minutes, 1e-9)},
        {"blink_amp_mean", events.empty() ? 0.0 : amplitude_sum / count},
        {"blink_width_mean", events.empty() ? 0.0 : width_sum / count},
        {"long_blink_count", static_cast<double>(long_count)},
    };
}

}  // namespace n16
