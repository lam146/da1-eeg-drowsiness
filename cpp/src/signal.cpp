#include "n16/signal.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "n16/fft.hpp"

namespace n16 {
namespace {

double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot calculate median of empty data");
    }
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    const double upper = *middle;
    if (values.size() % 2 != 0) {
        return upper;
    }
    const double lower = *std::max_element(values.begin(), middle);
    return (lower + upper) / 2.0;
}

double integrate_band(
    const std::vector<double>& frequencies,
    const std::vector<double>& psd,
    double low,
    double high,
    bool include_high) {
    std::vector<std::size_t> selected;
    for (std::size_t i = 0; i < frequencies.size(); ++i) {
        if (frequencies[i] >= low && (include_high ? frequencies[i] <= high : frequencies[i] < high)) {
            selected.push_back(i);
        }
    }
    if (selected.size() < 2) {
        return 0.0;
    }
    double area = 0.0;
    for (std::size_t i = 1; i < selected.size(); ++i) {
        const auto left = selected[i - 1];
        const auto right = selected[i];
        area += 0.5 * (psd[left] + psd[right]) * (frequencies[right] - frequencies[left]);
    }
    return area;
}

}  // namespace

std::vector<double> preprocess_window(
    const std::vector<double>& samples,
    double fs,
    double low_hz,
    double high_hz,
    double notch_hz,
    double notch_width_hz) {
    if (samples.size() < 8 || (samples.size() & (samples.size() - 1)) != 0) {
        throw std::invalid_argument("EEG window must have a power-of-two sample count");
    }
    if (!(0.0 <= low_hz && low_hz < high_hz && high_hz < fs / 2.0)) {
        throw std::invalid_argument("Invalid frequency limits");
    }
    const double center = median(samples);
    std::vector<std::complex<double>> spectrum;
    spectrum.reserve(samples.size());
    for (const double sample : samples) {
        if (!std::isfinite(sample)) {
            throw std::invalid_argument("EEG contains non-finite samples");
        }
        spectrum.emplace_back(sample - center, 0.0);
    }
    fft(spectrum);

    const std::size_t n = samples.size();
    for (std::size_t index = 0; index < n; ++index) {
        const double signed_frequency = index <= n / 2
            ? static_cast<double>(index) * fs / static_cast<double>(n)
            : -static_cast<double>(n - index) * fs / static_cast<double>(n);
        const double frequency = std::abs(signed_frequency);
        bool keep = frequency >= low_hz && frequency <= high_hz;
        if (notch_hz > 0.0 && notch_hz < fs / 2.0) {
            keep = keep && std::abs(frequency - notch_hz) > notch_width_hz / 2.0;
        }
        if (!keep) {
            spectrum[index] = {0.0, 0.0};
        }
    }
    fft(spectrum, true);
    std::vector<double> filtered;
    filtered.reserve(n);
    for (const auto& value : spectrum) {
        filtered.push_back(value.real());
    }
    return filtered;
}

FeatureMap spectral_features(const std::vector<double>& samples, double fs) {
    auto filtered = preprocess_window(samples, fs);
    const double center = median(filtered);
    const std::size_t n = filtered.size();
    constexpr double pi = 3.14159265358979323846;
    std::vector<std::complex<double>> transformed(n);
    double window_energy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double hann = 0.5 - 0.5 * std::cos(2.0 * pi * static_cast<double>(i) / static_cast<double>(n - 1));
        transformed[i] = {(filtered[i] - center) * hann, 0.0};
        window_energy += hann * hann;
    }
    fft(transformed);

    std::vector<double> frequencies(n / 2 + 1);
    std::vector<double> psd(n / 2 + 1);
    const double scale = std::max(fs * window_energy, std::numeric_limits<double>::epsilon());
    for (std::size_t i = 0; i <= n / 2; ++i) {
        frequencies[i] = static_cast<double>(i) * fs / static_cast<double>(n);
        psd[i] = std::norm(transformed[i]) / scale;
        if (i != 0 && i != n / 2) {
            psd[i] *= 2.0;
        }
    }

    const double epsilon = std::numeric_limits<double>::epsilon();
    const double total = std::max(integrate_band(frequencies, psd, 0.5, 45.0, true), epsilon);
    const std::vector<std::pair<std::string, std::pair<double, double>>> bands = {
        {"delta", {0.5, 4.0}}, {"theta", {4.0, 8.0}}, {"alpha", {8.0, 13.0}},
        {"beta", {13.0, 30.0}}, {"gamma", {30.0, 45.0}},
    };
    FeatureMap features;
    FeatureMap powers;
    for (const auto& band : bands) {
        const double power = integrate_band(frequencies, psd, band.second.first, band.second.second, false);
        powers[band.first] = power;
        features["abs_" + band.first] = power;
        features["rel_" + band.first] = power / total;
    }
    features["theta_alpha"] = powers["theta"] / std::max(powers["alpha"], epsilon);
    features["theta_beta"] = powers["theta"] / std::max(powers["beta"], epsilon);
    features["alpha_beta"] = powers["alpha"] / std::max(powers["beta"], epsilon);

    double psd_sum = 0.0;
    std::size_t entropy_count = 0;
    for (std::size_t i = 0; i < frequencies.size(); ++i) {
        if (frequencies[i] >= 0.5 && frequencies[i] <= 45.0) {
            psd_sum += psd[i];
            ++entropy_count;
        }
    }
    double entropy = 0.0;
    for (std::size_t i = 0; i < frequencies.size(); ++i) {
        if (frequencies[i] >= 0.5 && frequencies[i] <= 45.0) {
            const double probability = psd[i] / std::max(psd_sum, epsilon);
            entropy -= probability * std::log2(probability + epsilon);
        }
    }
    features["spectral_entropy"] = entropy / std::max(std::log2(static_cast<double>(entropy_count)), epsilon);
    double square_sum = 0.0;
    for (const double value : filtered) {
        square_sum += value * value;
    }
    const auto minmax = std::minmax_element(filtered.begin(), filtered.end());
    features["rms"] = std::sqrt(square_sum / static_cast<double>(filtered.size()));
    features["peak_to_peak"] = *minmax.second - *minmax.first;
    return features;
}

}  // namespace n16
