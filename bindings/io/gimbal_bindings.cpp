#include <chrono>
#include <tuple>

#include <fmt/core.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "io/gimbal/gimbal.hpp"
#include "tools/math_tools.hpp"

namespace py = pybind11;

namespace
{
py::tuple ypr_now(io::Gimbal & self)
{
  const auto q = self.q(std::chrono::steady_clock::now());
  const auto ypr = tools::eulers(q, 2, 1, 0);
  return py::make_tuple(ypr[0], ypr[1], ypr[2]);
}
}  // namespace

void bind_gimbal(py::module_ & m)
{
  py::enum_<io::GimbalMode>(m, "GimbalMode")
    .value("IDLE", io::GimbalMode::IDLE)
    .value("AUTO_AIM", io::GimbalMode::AUTO_AIM)
    .value("SMALL_BUFF", io::GimbalMode::SMALL_BUFF)
    .value("BIG_BUFF", io::GimbalMode::BIG_BUFF)
    .export_values();

  py::class_<io::VisionToGimbal>(m, "VisionToGimbal")
    .def(py::init<>())
    .def_readwrite("header", &io::VisionToGimbal::header)
    .def_readwrite("tracking", &io::VisionToGimbal::tracking)
    .def_readwrite("pitch", &io::VisionToGimbal::pitch)
    .def_readwrite("yaw", &io::VisionToGimbal::yaw)
    .def_readwrite("fire", &io::VisionToGimbal::fire)
    .def_readwrite("fric_on", &io::VisionToGimbal::fric_on)
    .def_readwrite("checksum", &io::VisionToGimbal::checksum);

  py::class_<io::GimbalState>(m, "GimbalState")
    .def(py::init<>())
    .def_readwrite("yaw", &io::GimbalState::yaw)
    .def_readwrite("yaw_vel", &io::GimbalState::yaw_vel)
    .def_readwrite("pitch", &io::GimbalState::pitch)
    .def_readwrite("pitch_vel", &io::GimbalState::pitch_vel)
    .def_readwrite("roll", &io::GimbalState::roll)
    .def_readwrite("yaw_odom", &io::GimbalState::yaw_odom)
    .def_readwrite("pitch_odom", &io::GimbalState::pitch_odom)
    .def_readwrite("bullet_speed", &io::GimbalState::bullet_speed)
    .def_readwrite("bullet_count", &io::GimbalState::bullet_count)
    .def_readwrite("robot_id", &io::GimbalState::robot_id)
    .def("__repr__", [](const io::GimbalState & self) {
      return fmt::format(
        "GimbalState(yaw={:.3f}, pitch={:.3f}, roll={:.3f}, yaw_vel={:.3f}, pitch_vel={:.3f}, "
        "yaw_odom={:.3f}, pitch_odom={:.3f}, bullet_speed={:.3f}, bullet_count={}, robot_id={})",
        self.yaw, self.pitch, self.roll, self.yaw_vel, self.pitch_vel, self.yaw_odom,
        self.pitch_odom, self.bullet_speed, self.bullet_count, static_cast<int>(self.robot_id));
    });

  py::class_<io::GimbalRxStats>(m, "GimbalRxStats")
    .def(py::init<>())
    .def_readwrite("good_frames", &io::GimbalRxStats::good_frames)
    .def_readwrite("crc_fail", &io::GimbalRxStats::crc_fail)
    .def_readwrite("short_read", &io::GimbalRxStats::short_read)
    .def_readwrite("header_mismatch", &io::GimbalRxStats::header_mismatch)
    .def_readwrite("reconnect_count", &io::GimbalRxStats::reconnect_count)
    .def_readwrite("consecutive_crc_fail", &io::GimbalRxStats::consecutive_crc_fail)
    .def_readwrite("last_header", &io::GimbalRxStats::last_header)
    .def_readwrite("last_rx_crc", &io::GimbalRxStats::last_rx_crc)
    .def_readwrite("last_calc_crc", &io::GimbalRxStats::last_calc_crc)
    .def_property_readonly("last_good_frame_age_ms", [](const io::GimbalRxStats & self) -> py::object {
      if (self.last_good_frame_time.time_since_epoch().count() == 0) return py::none();
      return py::float_(std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - self.last_good_frame_time)
                          .count());
    })
    .def("__repr__", [](const io::GimbalRxStats & self) {
      return fmt::format(
        "GimbalRxStats(good_frames={}, crc_fail={}, short_read={}, header_mismatch={}, "
        "reconnect_count={}, consecutive_crc_fail={}, last_header=0x{:02X}, last_rx_crc=0x{:04X}, "
        "last_calc_crc=0x{:04X})",
        self.good_frames, self.crc_fail, self.short_read, self.header_mismatch,
        self.reconnect_count, self.consecutive_crc_fail, self.last_header, self.last_rx_crc,
        self.last_calc_crc);
    });

  py::class_<io::Gimbal>(m, "Gimbal")
    .def(py::init<const std::string &, bool>(), py::arg("config_path"),
         py::arg("wait_for_first_q") = true)
    .def("mode", &io::Gimbal::mode)
    .def("state", &io::Gimbal::state)
    .def("rx_stats", &io::Gimbal::rx_stats)
    .def("has_valid_q", &io::Gimbal::has_valid_q)
    .def("mode_name", &io::Gimbal::str, py::arg("mode"))
    .def(
      "q_at_ns",
      [](io::Gimbal & self, int64_t timestamp_ns) {
        auto timestamp = std::chrono::steady_clock::time_point(
          std::chrono::steady_clock::duration(timestamp_ns));
        const auto q = self.q(timestamp);
        return py::make_tuple(q.w(), q.x(), q.y(), q.z());
      },
      py::arg("timestamp_ns"))
    .def("ypr_now", &ypr_now)
    .def(
      "send",
      py::overload_cast<
        bool, bool, float, float, float, float, float, float>(&io::Gimbal::send),
      py::arg("control"), py::arg("fire"), py::arg("yaw"), py::arg("yaw_vel"),
      py::arg("yaw_acc"), py::arg("pitch"), py::arg("pitch_vel"), py::arg("pitch_acc"))
    .def(
      "send_vision",
      [](io::Gimbal & self, const io::VisionToGimbal & vision) { self.send(vision); },
      py::arg("vision"))
    .def("__repr__", [](const io::Gimbal & self) {
      return fmt::format("Gimbal(mode={}, has_valid_q={})", static_cast<int>(self.mode()),
                         self.has_valid_q() ? "true" : "false");
    });
}
