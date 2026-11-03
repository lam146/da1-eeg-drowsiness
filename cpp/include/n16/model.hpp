#pragma once

#include <string>
#include <vector>

#include "n16/types.hpp"

namespace n16 {

class LogisticModel {
public:
    static LogisticModel load(const std::string& path);
    double predict(const FeatureMap& features) const;

    double threshold() const noexcept { return threshold_; }
    const std::vector<std::string>& feature_names() const noexcept { return feature_names_; }

private:
    std::vector<std::string> feature_names_;
    std::vector<double> mean_;
    std::vector<double> scale_;
    std::vector<double> weights_;
    double bias_{};
    double threshold_{0.5};
};

}  // namespace n16
