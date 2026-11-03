#pragma once

#include <map>
#include <string>

namespace n16 {

using FeatureMap = std::map<std::string, double>;

struct Decision {
    long long timestamp_ms{};
    std::string state{"TIN_HIEU_KEM"};
    double risk{-1.0};
    bool alarm{false};
    double poor_signal_fraction{1.0};
    double theta_alpha{};
    int blink_count{};
    int long_blink_count{};
    double latency_ms{};
};

}  // namespace n16
