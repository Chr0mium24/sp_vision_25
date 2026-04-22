#include <chrono>
#include <cstdint>
#include <tuple>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "io/cboard.hpp"

namespace py = pybind11;

namespace
{
std::chrono::steady_clock::time_point ns_to_time_point(int64_t ns)
{
  return std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(ns));
}
}  // namespace

void bind_cboard(py::module_ & m)
{
  py::class_<io::CBoard>(m, "CBoard")
    .def(py::init<const std::string &>(), py::arg("config_path"))
    .def_property_readonly("bullet_speed", [](const io::CBoard & self) { return self.bullet_speed; })
    .def_property_readonly("mode", [](const io::CBoard & self) { return self.mode; })
    .def_property_readonly("shoot_mode", [](const io::CBoard & self) { return self.shoot_mode; })
    .def_property_readonly("ft_angle", [](const io::CBoard & self) { return self.ft_angle; })
    .def(
      "imu_at",
      [](io::CBoard & self, int64_t timestamp_ns) {
        auto q = self.imu_at(ns_to_time_point(timestamp_ns));
        return py::make_tuple(q.w(), q.x(), q.y(), q.z());
      },
      py::arg("timestamp_ns"))
    .def(
      "send",
      [](const io::CBoard & self, const io::Command & command) { self.send(command); },
      py::arg("command"))
    .def("__repr__", [](const io::CBoard &) { return py::str("CBoard()"); });
}
