#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "serial/serial.h"
#include "tools/exiter.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono_literals;

namespace
{
constexpr uint8_t kUsbRxHeader = 0x5A;
constexpr size_t kUsbRxSize = 28;
constexpr double kPi = 3.14159265358979323846;
constexpr uint16_t kCrc16Init = 0xFFFF;
constexpr uint16_t kCrc16Poly = 0xA001;

struct UsbRxFrame
{
  uint8_t detect_color = 0;
  uint8_t reset_tracker = 0;
  float roll = 0.0f;
  float pitch = 0.0f;
  float yaw = 0.0f;
  float aim_x = 0.0f;
  float aim_y = 0.0f;
  float aim_z = 0.0f;
};

uint16_t crc16_ibm(const uint8_t * data, size_t length)
{
  uint16_t crc = kCrc16Init;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      if (crc & 1) {
        crc = static_cast<uint16_t>((crc >> 1) ^ kCrc16Poly);
      } else {
        crc >>= 1;
      }
    }
  }
  return crc;
}

float unpack_float(const uint8_t * data)
{
  float value = 0.0f;
  std::memcpy(&value, data, sizeof(float));
  return value;
}

bool parse_usb_rx_frame(
  std::vector<uint8_t> & buffer, UsbRxFrame & out, bool & checksum_ok)
{
  for (size_t i = 0; i + kUsbRxSize <= buffer.size(); ++i) {
    if (buffer[i] != kUsbRxHeader) continue;

    uint16_t crc_expected = crc16_ibm(buffer.data() + i, kUsbRxSize - 2);
    uint16_t crc_got = static_cast<uint16_t>(buffer[i + kUsbRxSize - 2]) |
                       static_cast<uint16_t>(buffer[i + kUsbRxSize - 1] << 8);
    checksum_ok = (crc_expected == crc_got);

    uint8_t flags = buffer[i + 1];
    out.detect_color = flags & 0x01;
    out.reset_tracker = (flags >> 1) & 0x01;
    out.roll = unpack_float(buffer.data() + i + 2);
    out.pitch = unpack_float(buffer.data() + i + 6);
    out.yaw = unpack_float(buffer.data() + i + 10);
    out.aim_x = unpack_float(buffer.data() + i + 14);
    out.aim_y = unpack_float(buffer.data() + i + 18);
    out.aim_z = unpack_float(buffer.data() + i + 22);

    buffer.erase(buffer.begin(), buffer.begin() + i + kUsbRxSize);
    return true;
  }

  if (buffer.size() > 64) {
    buffer.erase(buffer.begin(), buffer.end() - (kUsbRxSize - 1));
  }
  return false;
}

}  // namespace

const std::string keys =
  "{help h usage ? |      | show command line options}"
  "{@port          |      | serial port path }"
  "{baud           | 115200 | serial baud rate }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto port = cli.get<std::string>(0);
  int baud = cli.get<int>("baud");
  if (cli.has("help") || port.empty()) {
    cli.printMessage();
    return 0;
  }

  serial::Serial serial;
  serial.setPort(port);
  serial.setBaudrate(static_cast<uint32_t>(baud));
  auto timeout = serial::Timeout::simpleTimeout(5);
  serial.setTimeout(timeout);
  serial.open();

  tools::Exiter exiter;

  std::vector<uint8_t> rx_buffer;
  UsbRxFrame rx{};
  bool has_rx = false;
  bool checksum_ok = false;
  auto last_rx = std::chrono::steady_clock::now();
  auto last_print = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    size_t avail = serial.available();
    if (avail > 0) {
      std::vector<uint8_t> chunk(avail);
      size_t got = serial.read(chunk, avail);
      rx_buffer.insert(rx_buffer.end(), chunk.begin(), chunk.begin() + got);
    }

    while (parse_usb_rx_frame(rx_buffer, rx, checksum_ok)) {
      has_rx = true;
      last_rx = std::chrono::steady_clock::now();
    }

    auto now = std::chrono::steady_clock::now();
    if (tools::delta_time(now, last_print) > 0.2) {
      double rx_age_ms = has_rx ? tools::delta_time(now, last_rx) * 1e3 : -1.0;
      std::printf(
        "rx_ok:%d crc:%d age:%.0fms yaw:%.2f pitch:%.2f roll:%.2f aim:(%.2f %.2f %.2f) color:%u reset:%u\n",
        has_rx ? 1 : 0, checksum_ok ? 1 : 0, rx_age_ms, rx.yaw * 180.0 / kPi,
        rx.pitch * 180.0 / kPi, rx.roll * 180.0 / kPi, rx.aim_x, rx.aim_y, rx.aim_z,
        rx.detect_color, rx.reset_tracker);
      std::fflush(stdout);
      last_print = now;
    }

    std::this_thread::sleep_for(5ms);
  }

  return 0;
}
