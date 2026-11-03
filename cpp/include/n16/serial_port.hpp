#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace n16 {

class SerialPort {
public:
    SerialPort(const std::string& port, unsigned long baud);
    ~SerialPort();
    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;
    std::size_t read(std::uint8_t* buffer, std::size_t capacity);

private:
#ifdef _WIN32
    void* handle_{nullptr};
#endif
};

}  // namespace n16
