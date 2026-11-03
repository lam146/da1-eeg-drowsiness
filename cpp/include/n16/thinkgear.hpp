#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace n16 {

struct ThinkGearPacket {
    std::vector<std::int16_t> raw_eeg;
    std::optional<int> poor_signal;
    std::optional<int> attention;
    std::optional<int> meditation;
    std::optional<int> blink_strength;
};

class ThinkGearParser {
public:
    std::vector<ThinkGearPacket> feed(const std::uint8_t* data, std::size_t size);
    std::vector<ThinkGearPacket> feed(const std::vector<std::uint8_t>& data) {
        return feed(data.data(), data.size());
    }

private:
    std::vector<std::uint8_t> buffer_;
    static ThinkGearPacket parse_payload(const std::vector<std::uint8_t>& payload);
};

}  // namespace n16
