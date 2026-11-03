#include "n16/model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace n16 {
namespace {

std::string read_text(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open model: " + path);
    }
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

std::string array_body(const std::string& json, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    const auto key_position = json.find(marker);
    if (key_position == std::string::npos) {
        throw std::runtime_error("Missing JSON key: " + key);
    }
    const auto begin = json.find('[', key_position + marker.size());
    const auto end = json.find(']', begin == std::string::npos ? key_position : begin);
    if (begin == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("Invalid JSON array: " + key);
    }
    return json.substr(begin + 1, end - begin - 1);
}

std::vector<double> number_array(const std::string& json, const std::string& key) {
    const auto body = array_body(json, key);
    const std::regex number_pattern(R"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)");
    std::vector<double> result;
    for (auto iterator = std::sregex_iterator(body.begin(), body.end(), number_pattern);
         iterator != std::sregex_iterator(); ++iterator) {
        result.push_back(std::stod(iterator->str()));
    }
    return result;
}

std::vector<std::string> string_array(const std::string& json, const std::string& key) {
    const auto body = array_body(json, key);
    const std::regex string_pattern("\"([^\"]+)\"");
    std::vector<std::string> result;
    for (auto iterator = std::sregex_iterator(body.begin(), body.end(), string_pattern);
         iterator != std::sregex_iterator(); ++iterator) {
        result.push_back((*iterator)[1].str());
    }
    return result;
}

double number_value(const std::string& json, const std::string& key) {
    const std::regex pattern(
        "\"" + key + "\"\\s*:\\s*([-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?)");
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        throw std::runtime_error("Missing numeric JSON key: " + key);
    }
    return std::stod(match[1].str());
}

}  // namespace

LogisticModel LogisticModel::load(const std::string& path) {
    const auto json = read_text(path);
    if (json.find("numpy_logistic_regression") == std::string::npos) {
        throw std::runtime_error("Unsupported model type");
    }
    LogisticModel model;
    model.feature_names_ = string_array(json, "feature_names");
    model.mean_ = number_array(json, "mean");
    model.scale_ = number_array(json, "scale");
    model.weights_ = number_array(json, "weights");
    model.bias_ = number_value(json, "bias");
    model.threshold_ = number_value(json, "threshold");
    const auto count = model.feature_names_.size();
    if (count == 0 || model.mean_.size() != count || model.scale_.size() != count
        || model.weights_.size() != count) {
        throw std::runtime_error("Model arrays have inconsistent sizes");
    }
    return model;
}

double LogisticModel::predict(const FeatureMap& features) const {
    double z = bias_;
    for (std::size_t i = 0; i < feature_names_.size(); ++i) {
        const auto found = features.find(feature_names_[i]);
        if (found == features.end()) {
            throw std::runtime_error("Missing feature: " + feature_names_[i]);
        }
        const double scale = std::abs(scale_[i]) < 1e-12 ? 1.0 : scale_[i];
        z += ((found->second - mean_[i]) / scale) * weights_[i];
    }
    z = std::clamp(z, -30.0, 30.0);
    return 1.0 / (1.0 + std::exp(-z));
}

}  // namespace n16
