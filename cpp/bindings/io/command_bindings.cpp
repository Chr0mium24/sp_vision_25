#include <fmt/format.h>

#include <pybind11/pybind11.h>

#include "io/cboard.hpp"

namespace py = pybind11;

void bind_command(py::module_ & m)
{
  py::class_<io::Command>(m, "Command")
    .def(py::init<>())
    .def_readwrite("control", &io::Command::control)
    .def_readwrite("shoot", &io::Command::shoot)
    .def_readwrite("yaw", &io::Command::yaw)
    .def_readwrite("pitch", &io::Command::pitch)
    .def_readwrite("horizon_distance", &io::Command::horizon_distance)
    .def("__repr__", [](const io::Command & self) {
      return fmt::format(
        "Command(control={}, shoot={}, yaw={}, pitch={}, horizon_distance={})", self.control,
        self.shoot, self.yaw, self.pitch, self.horizon_distance);
    });

  py::enum_<io::Mode>(m, "Mode")
    .value("idle", io::idle)
    .value("auto_aim", io::auto_aim)
    .value("small_buff", io::small_buff)
    .value("big_buff", io::big_buff)
    .value("outpost", io::outpost)
    .export_values();

  py::enum_<io::ShootMode>(m, "ShootMode")
    .value("left_shoot", io::left_shoot)
    .value("right_shoot", io::right_shoot)
    .value("both_shoot", io::both_shoot)
    .export_values();
}
