#include "n16/thinkgear.hpp"

#include <algorithm>
#include <numeric>

namespace n16 {
namespace {
constexpr std::uint8_t sync_byte = 0xAA;
constexpr std::size_t max_payload = 169;
}

std::vector<ThinkGearPacket> ThinkGearParser::feed(const std::uint8_t* data, std::size_t size) {
    buffer_.insert(buffer_.end(), data, data + size);
    std::vector<ThinkGearPacket> packets;
    while (true) {
        auto start = buffer_.end();
        for (auto iterator = buffer_.begin(); iterator != buffer_.end(); ++iterator) {
            if (*iterator == sync_byte && iterator + 1 != buffer_.end() && *(iterator + 1) == sync_byte) {
                start = iterator;
                break;
            }
        }
        if (start == buffer_.end()) {
            const bool keep_sync = !buffer_.empty() && buffer_.back() == sync_byte;
            buffer_.clear();
            if (keep_sync) {
                buffer_.push_back(sync_byte);
            }
            break;
        }
        buffer_.erase(buffer_.begin(), start);
        if (buffer_.size() < 3) {
            break;
        }
        const std::size_t payload_length = buffer_[2];
        if (payload_length > max_payload) {
            buffer_.erase(buffer_.begin());
            continue;
        }
        const std::size_t total_length = 3 + payload_length + 1;
        if (buffer_.size() < total_length) {
            break;
        }
        std::vector<std::uint8_t> payload(buffer_.begin() + 3, buffer_.begin() + 3 + payload_length);
        const auto checksum = buffer_[3 + payload_length];
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(total_length));
        const auto sum = std::accumulate(payload.begin(), payload.end(), 0u);
        const auto expected = static_cast<std::uint8_t>(~(sum & 0xFFu));
        if (checksum == expected) {
            packets.push_back(parse_payload(payload));
        }
    }
    return packets;
}

ThinkGearPacket ThinkGearParser::parse_payload(const std::vector<std::uint8_t>& payload) {
    ThinkGearPacket packet;
    std::size_t index = 0;
    while (index < payload.size()) {
        while (index < payload.size() && payload[index] == 0x55) {
            ++index;
        }
        if (index >= payload.size()) {
            break;
        }
        const auto code = payload[index++];
        if (code < 0x80) {
            if (index >= payload.size()) {
                break;
            }
            const int value = payload[index++];
            if (code == 0x02) packet.poor_signal = value;
            else if (code == 0x04) packet.attention = value;
            else if (code == 0x05) packet.meditation = value;
            else if (code == 0x16) packet.blink_strength = value;
            continue;
        }
        if (index >= payload.size()) {
            break;
        }
        const std::size_t length = payload[index++];
        if (index + length > payload.size()) {
            break;
        }
        if (code == 0x80 && length == 2) {
            const auto unsigned_value = static_cast<std::uint16_t>(payload[index] << 8u)
                | static_cast<std::uint16_t>(payload[index + 1]);
            packet.raw_eeg.push_back(static_cast<std::int16_t>(unsigned_value));
        }
        index += length;
    }
    return packet;
}

}  // namespace n16
