#include "n16/serial_port.hpp"

#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace n16 {

SerialPort::SerialPort(const std::string& port, unsigned long baud) {
#ifdef _WIN32
    const std::string device = port.rfind("\\\\.\\", 0) == 0 ? port : "\\\\.\\" + port;
    HANDLE handle = CreateFileA(
        device.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Cannot open serial port " + port);
    }
    DCB settings{};
    settings.DCBlength = sizeof(settings);
    if (!GetCommState(handle, &settings)) {
        CloseHandle(handle);
        throw std::runtime_error("Cannot read serial settings");
    }
    settings.BaudRate = baud;
    settings.ByteSize = 8;
    settings.StopBits = ONESTOPBIT;
    settings.Parity = NOPARITY;
    settings.fDtrControl = DTR_CONTROL_ENABLE;
    if (!SetCommState(handle, &settings)) {
        CloseHandle(handle);
        throw std::runtime_error("Cannot configure serial port");
    }
    COMMTIMEOUTS timeouts{};
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 250;
    timeouts.ReadTotalTimeoutMultiplier = 0;
    SetCommTimeouts(handle, &timeouts);
    PurgeComm(handle, PURGE_RXCLEAR);
    handle_ = handle;
#else
    (void)port;
    (void)baud;
    throw std::runtime_error("Live serial mode is currently implemented for Windows only");
#endif
}

SerialPort::~SerialPort() {
#ifdef _WIN32
    if (handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(handle_));
    }
#endif
}

std::size_t SerialPort::read(std::uint8_t* buffer, std::size_t capacity) {
#ifdef _WIN32
    DWORD received = 0;
    if (!ReadFile(static_cast<HANDLE>(handle_), buffer, static_cast<DWORD>(capacity), &received, nullptr)) {
        throw std::runtime_error("Serial read failed");
    }
    return static_cast<std::size_t>(received);
#else
    (void)buffer;
    (void)capacity;
    return 0;
#endif
}

}  // namespace n16
