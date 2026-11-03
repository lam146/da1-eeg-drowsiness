#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "n16/model.hpp"
#include "n16/runtime.hpp"
#include "n16/serial_port.hpp"
#include "n16/signal.hpp"
#include "n16/thinkgear.hpp"

namespace {

using Options = std::map<std::string, std::string>;

void print_help() {
    std::cout
        << "N16 EEG C++17 real-time runtime\n\n"
        << "Commands:\n"
        << "  n16_runtime self-test\n"
        << "  n16_runtime replay --input FILE [--model FILE] [--log FILE] [--sound]\n"
        << "  n16_runtime live --port COM5 [--model FILE] [--log FILE] [--duration-min 30]\n\n"
        << "Python is used for training/LOSO. This program loads the exported JSON model\n"
        << "and performs acquisition, DSP, inference and warning in C++.\n";
}

Options parse_options(int argc, char** argv, int start) {
    Options options;
    for (int i = start; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("Unexpected argument: " + key);
        }
        if (key == "--sound" || key == "--no-sound") {
            options[key] = "true";
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        options[key] = argv[++i];
    }
    return options;
}

std::string option(const Options& options, const std::string& key, const std::string& fallback = "") {
    const auto found = options.find(key);
    return found == options.end() ? fallback : found->second;
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char character = line[i];
        if (character == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (character == ',' && !quoted) {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(character);
        }
    }
    fields.push_back(current);
    return fields;
}

void prepare_output(const std::string& path) {
    const std::filesystem::path output(path);
    if (output.has_parent_path()) {
        std::filesystem::create_directories(output.parent_path());
    }
}

void write_header(std::ofstream& output) {
    output << "timestamp_ms,state,risk,alarm,poor_signal_fraction,theta_alpha,"
              "blink_count,long_blink_count,latency_ms\n";
}

void write_decision(std::ofstream& output, const n16::Decision& decision) {
    output << decision.timestamp_ms << ',' << decision.state << ',';
    if (decision.risk >= 0.0) {
        output << std::fixed << std::setprecision(6) << decision.risk;
    }
    output << ',' << (decision.alarm ? 1 : 0) << ',' << decision.poor_signal_fraction << ',';
    if (decision.risk >= 0.0) {
        output << decision.theta_alpha;
    }
    output << ',' << decision.blink_count << ',' << decision.long_blink_count << ','
           << decision.latency_ms << '\n';
}

void show_decision(const n16::Decision& decision) {
    std::cout << "state=" << decision.state;
    if (decision.risk >= 0.0) {
        std::cout << "  risk=" << std::fixed << std::setprecision(3) << decision.risk
                  << "  theta/alpha=" << decision.theta_alpha
                  << "  blink=" << decision.blink_count;
    }
    std::cout << "  latency=" << std::setprecision(3) << decision.latency_ms << " ms\n";
}

void beep_warning() {
#ifdef _WIN32
    Beep(1250, 250);
#else
    std::cout << '\a' << std::flush;
#endif
}

int replay(const Options& options) {
    const auto input_path = option(options, "--input");
    if (input_path.empty()) {
        throw std::runtime_error("replay requires --input FILE");
    }
    const auto model_path = option(options, "--model", "output/model/n16_logistic_model.json");
    const auto log_path = option(options, "--log", "output/demo/cpp_replay_log.csv");
    std::ifstream input(input_path);
    if (!input) {
        throw std::runtime_error("Cannot open CSV: " + input_path);
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("CSV is empty");
    }
    const auto header = split_csv(line);
    auto column_index = [&header](const std::string& name) {
        const auto found = std::find(header.begin(), header.end(), name);
        if (found == header.end()) {
            throw std::runtime_error("Missing CSV column: " + name);
        }
        return static_cast<std::size_t>(std::distance(header.begin(), found));
    };
    const auto timestamp_column = column_index("timestamp_ms");
    const auto raw_column = column_index("raw_eeg");
    const auto poor_column = column_index("poor_signal");

    n16::LiveDetector detector(n16::LogisticModel::load(model_path));
    prepare_output(log_path);
    std::ofstream output(log_path);
    write_header(output);
    int decisions = 0;
    int alarms = 0;
    while (std::getline(input, line)) {
        const auto fields = split_csv(line);
        const auto needed = std::max({timestamp_column, raw_column, poor_column});
        if (fields.size() <= needed) {
            continue;
        }
        const auto decision = detector.add_sample(
            std::stod(fields[raw_column]), std::stod(fields[poor_column]), std::stoll(fields[timestamp_column]));
        if (!decision) {
            continue;
        }
        write_decision(output, *decision);
        show_decision(*decision);
        ++decisions;
        if (decision->alarm) {
            ++alarms;
            if (options.count("--sound") != 0) {
                beep_warning();
            }
        }
    }
    std::cout << "Wrote " << decisions << " decisions to " << log_path
              << "; alarms=" << alarms << '\n';
    return decisions > 0 ? 0 : 2;
}

long long now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

int live(const Options& options) {
    const auto port = option(options, "--port");
    if (port.empty()) {
        throw std::runtime_error("live requires --port COMx");
    }
    const auto model_path = option(options, "--model", "output/model/n16_logistic_model.json");
    const auto log_path = option(options, "--log", "output/live/cpp_live_log.csv");
    const auto duration_minutes = std::stod(option(options, "--duration-min", "30"));
    const auto baud = static_cast<unsigned long>(std::stoul(option(options, "--baud", "57600")));
    n16::LiveDetector detector(n16::LogisticModel::load(model_path));
    n16::ThinkGearParser parser;
    n16::SerialPort serial(port, baud);
    prepare_output(log_path);
    std::ofstream output(log_path);
    write_header(output);
    int poor_signal = 255;
    std::vector<std::uint8_t> bytes(4096);
    const auto started = std::chrono::steady_clock::now();
    std::cout << "Connected to " << port << " at " << baud << " baud\n";
    while (std::chrono::duration<double, std::ratio<60>>(
               std::chrono::steady_clock::now() - started).count() < duration_minutes) {
        const auto count = serial.read(bytes.data(), bytes.size());
        for (const auto& packet : parser.feed(bytes.data(), count)) {
            if (packet.poor_signal) {
                poor_signal = *packet.poor_signal;
            }
            for (const auto sample : packet.raw_eeg) {
                const auto decision = detector.add_sample(sample, poor_signal, now_ms());
                if (!decision) {
                    continue;
                }
                write_decision(output, *decision);
                output.flush();
                show_decision(*decision);
                if (decision->alarm && options.count("--no-sound") == 0) {
                    beep_warning();
                }
            }
        }
    }
    return 0;
}

int self_test() {
    constexpr int fs = 512;
    constexpr int seconds = 4;
    constexpr double pi = 3.14159265358979323846;
    std::vector<double> signal(fs * seconds);
    for (std::size_t i = 0; i < signal.size(); ++i) {
        signal[i] = std::sin(2.0 * pi * 10.0 * static_cast<double>(i) / fs);
    }
    const auto features = n16::spectral_features(signal, fs);
    if (features.at("rel_alpha") < 0.90 || features.at("rel_alpha") <= features.at("rel_theta")) {
        throw std::runtime_error("Spectral feature self-test failed");
    }

    const std::vector<std::uint8_t> payload = {0x02, 0x00, 0x04, 55, 0x80, 0x02, 0xFF, 0x9C};
    unsigned int sum = 0;
    for (const auto value : payload) sum += value;
    std::vector<std::uint8_t> packet = {0xAA, 0xAA, static_cast<std::uint8_t>(payload.size())};
    packet.insert(packet.end(), payload.begin(), payload.end());
    packet.push_back(static_cast<std::uint8_t>(~(sum & 0xFFu)));
    n16::ThinkGearParser parser;
    const auto parsed = parser.feed(packet);
    if (parsed.size() != 1 || parsed.front().raw_eeg.size() != 1
        || parsed.front().raw_eeg.front() != -100 || parsed.front().attention != 55) {
        throw std::runtime_error("ThinkGear parser self-test failed");
    }
    std::cout << "C++ self-test passed: FFT features and ThinkGear parser\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_help();
            return 1;
        }
        const std::string command = argv[1];
        if (command == "self-test") {
            return self_test();
        }
        if (command == "replay") {
            return replay(parse_options(argc, argv, 2));
        }
        if (command == "live") {
            return live(parse_options(argc, argv, 2));
        }
        if (command == "--help" || command == "help") {
            print_help();
            return 0;
        }
        throw std::runtime_error("Unknown command: " + command);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
