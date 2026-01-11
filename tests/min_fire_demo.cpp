#include <chrono>
#include <cstdint>
#include <thread>

#include "serial/serial.h"
#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

struct __attribute__((packed)) VisionToGimbal
{
  uint8_t header = 0xA5;
  uint8_t tracking = 0;
  float pitch = 0.0f;
  float yaw = 0.0f;
  uint8_t fire = 0;
  uint8_t fric_on = 1;
  uint16_t checksum = 0;
};

int main(int argc, char * argv[])
{
  if (argc < 2) {
    tools::logger()->error("Usage: {} <config.yaml>", argv[0]);
    return 1;
  }

  auto yaml = tools::load(argv[1]);
  auto com_port = tools::read<std::string>(yaml, "com_port");

  serial::Serial serial;
  try {
    serial.setPort(com_port);
    serial.setBaudrate(115200);
    auto timeout = serial::Timeout::simpleTimeout(100);
    serial.setTimeout(timeout);
    serial.open();
  } catch (const std::exception & e) {
    tools::logger()->error("Failed to open serial {}: {}", com_port, e.what());
    return 1;
  }

  tools::logger()->info("Serial opened: {}", com_port);

  using clock = std::chrono::steady_clock;
  auto last_fire = clock::now();
  auto fire_until = clock::time_point{};

  const auto fire_interval = std::chrono::milliseconds(1600);
  const auto fire_pulse = std::chrono::milliseconds(120);

  while (true) {
    auto now = clock::now();
    if (now - last_fire >= fire_interval) {
      fire_until = now + fire_pulse;
      last_fire = now;
      tools::logger()->info("fire pulse");
    }

    VisionToGimbal packet;
    packet.fire = (now < fire_until) ? 1 : 0;
    packet.fric_on = 1;
    packet.checksum = tools::get_crc16(
      reinterpret_cast<const uint8_t *>(&packet), sizeof(packet) - sizeof(packet.checksum));

    try {
      serial.write(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
    } catch (const std::exception & e) {
      tools::logger()->warn("Failed to write serial: {}", e.what());
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  return 0;
}
