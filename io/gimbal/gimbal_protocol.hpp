#ifndef IO__GIMBAL_PROTOCOL_HPP
#define IO__GIMBAL_PROTOCOL_HPP

#include <array>
#include <cstdint>
#include <cstring>

#include "tools/crc.hpp"

namespace io
{
struct GimbalToVision;
struct VisionToGimbal;

inline constexpr uint8_t kGimbalToVisionHeader = 0x5A;
inline constexpr uint8_t kVisionToGimbalHeader = 0xA5;

template <typename T>
inline std::array<uint8_t, sizeof(T)> to_bytes(const T & packet)
{
  std::array<uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &packet, sizeof(T));
  return bytes;
}

template <typename T>
inline T from_bytes(const uint8_t * data)
{
  T packet{};
  std::memcpy(&packet, data, sizeof(T));
  return packet;
}

template <typename T>
inline uint16_t compute_crc16(const T & packet)
{
  return tools::get_crc16(reinterpret_cast<const uint8_t *>(&packet), sizeof(T) - sizeof(uint16_t));
}

template <typename T>
inline void refresh_crc16(T & packet)
{
  packet.checksum = compute_crc16(packet);
}

template <typename T>
inline bool validate_crc16(const T & packet)
{
  return packet.checksum == compute_crc16(packet);
}

}  // namespace io

#endif  // IO__GIMBAL_PROTOCOL_HPP
