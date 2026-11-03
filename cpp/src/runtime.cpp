#include "n16/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "n16/blink.hpp"
#include "n16/signal.hpp"

namespace n16 {

LiveDetector::LiveDetector(
    LogisticModel model,
    int fs,
    double window_s,
    double step_s,
    int consecutive)
    : model_(std::move(model)),
      fs_(fs),
      window_samples_(static_cast<std::size_t>(window_s * fs)),
      step_samples_(static_cast<std::size_t>(step_s * fs)),
      consecutive_(consecutive) {
    if (fs <= 90 || window_samples_ == 0 || step_samples_ == 0 || consecutive < 1
        || (window_samples_ & (window_samples_ - 1)) != 0) {
        throw std::invalid_argument("Invalid detector parameters");
    }
}

std::optional<Decision> LiveDetector::add_sample(
    double sample,
    double poor_signal,
    long long timestamp_ms) {
    if (samples_.size() == window_samples_) {
        samples_.pop_front();
        poor_signal_.pop_front();
    }
    samples_.push_back(sample);
    poor_signal_.push_back(poor_signal);
    ++samples_since_decision_;
    if (samples_.size() < window_samples_ || samples_since_decision_ < step_samples_) {
        return std::nullopt;
    }
    samples_since_decision_ = 0;
    const auto started = std::chrono::steady_clock::now();
    const auto bad_count = std::count_if(poor_signal_.begin(), poor_signal_.end(), [](double value) {
        return value > 50.0;
    });
    const double poor_fraction = static_cast<double>(bad_count) / static_cast<double>(poor_signal_.size());
    Decision decision;
    decision.timestamp_ms = timestamp_ms;
    decision.poor_signal_fraction = poor_fraction;
    if (poor_fraction > 0.20) {
        alert_streak_ = 0;
        const auto stopped = std::chrono::steady_clock::now();
        decision.latency_ms = std::chrono::duration<double, std::milli>(stopped - started).count();
        return decision;
    }

    const std::vector<double> values(samples_.begin(), samples_.end());
    auto features = spectral_features(values, static_cast<double>(fs_));
    const auto blink = blink_features(values, static_cast<double>(fs_));
    features.insert(blink.begin(), blink.end());
    decision.risk = model_.predict(features);
    decision.theta_alpha = features.at("theta_alpha");
    decision.blink_count = static_cast<int>(features.at("blink_count"));
    decision.long_blink_count = static_cast<int>(features.at("long_blink_count"));
    alert_streak_ = decision.risk >= model_.threshold() ? alert_streak_ + 1 : 0;
    decision.alarm = alert_streak_ >= consecutive_;
    decision.state = decision.alarm ? "CANH_BAO" : (decision.risk >= model_.threshold() ? "NGUY_CO" : "TINH");
    const auto stopped = std::chrono::steady_clock::now();
    decision.latency_ms = std::chrono::duration<double, std::milli>(stopped - started).count();
    return decision;
}

}  // namespace n16
