#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>

#include "tools/protocol/crc.hpp"

namespace py = pybind11;

namespace
{
std::vector<uint8_t> buffer_to_bytes(const py::buffer & buffer)
{
  py::buffer_info info = buffer.request();
  if (info.itemsize != 1) {
    throw py::type_error("expected a byte-oriented buffer");
  }

  const auto * begin = static_cast<const uint8_t *>(info.ptr);
  return {begin, begin + info.size};
}
}  // namespace

void bind_crc(py::module_ & m)
{
  m.def(
    "crc8", [](py::buffer data) {
      auto bytes = buffer_to_bytes(data);
      return tools::get_crc8(bytes.data(), static_cast<uint16_t>(bytes.size()));
    },
    "Compute CRC8 for a byte buffer.");
  m.def(
    "check_crc8", [](py::buffer data) {
      auto bytes = buffer_to_bytes(data);
      if (bytes.empty()) return false;
      return tools::check_crc8(bytes.data(), static_cast<uint16_t>(bytes.size()));
    },
    "Check CRC8 for a byte buffer that includes the CRC byte.");
  m.def(
    "crc16", [](py::buffer data) {
      auto bytes = buffer_to_bytes(data);
      return tools::get_crc16(bytes.data(), static_cast<uint32_t>(bytes.size()));
    },
    "Compute CRC16 for a byte buffer.");
  m.def(
    "check_crc16", [](py::buffer data) {
      auto bytes = buffer_to_bytes(data);
      if (bytes.size() < 2) return false;
      return tools::check_crc16(bytes.data(), static_cast<uint32_t>(bytes.size()));
    },
    "Check CRC16 for a byte buffer that includes the CRC16 bytes.");
}
