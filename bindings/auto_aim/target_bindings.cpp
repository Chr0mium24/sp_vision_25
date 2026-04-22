#include <chrono>
#include <tuple>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tasks/auto_aim/target.hpp"

namespace py = pybind11;

namespace
{
py::dict target_predict_debug_to_dict(const auto_aim::TargetPredictDebug & debug)
{
  py::dict out;
  out["valid"] = debug.valid;
  out["dt"] = debug.dt;
  out["outpost_speed_clamped"] = debug.outpost_speed_clamped;
  out["x_before"] = debug.x_before;
  out["x_after"] = debug.x_after;
  out["F"] = debug.F;
  out["Q"] = debug.Q;
  return out;
}

py::dict target_update_debug_to_dict(const auto_aim::TargetUpdateDebug & debug)
{
  py::dict out;
  out["valid"] = debug.valid;
  out["matched_id"] = debug.matched_id;
  out["last_id"] = debug.last_id;
  out["switch_count"] = debug.switch_count;
  out["update_count"] = debug.update_count;
  out["jumped"] = debug.jumped;
  out["is_switch"] = debug.is_switch;
  out["candidate_count"] = debug.candidate_count;
  out["center_yaw"] = debug.center_yaw;
  out["delta_angle"] = debug.delta_angle;
  out["x_before"] = debug.x_before;
  out["x_after"] = debug.x_after;
  out["z"] = debug.z;
  out["H"] = debug.H;
  out["R"] = debug.R;
  out["candidate_xyza_list"] = debug.candidate_xyza_list;
  return out;
}

py::dict target_debug_to_dict(const auto_aim::TargetDebug & debug)
{
  py::dict out;
  out["last_predict"] = target_predict_debug_to_dict(debug.last_predict);
  out["last_update"] = target_update_debug_to_dict(debug.last_update);
  return out;
}

std::vector<double> default_p0_dig()
{
  return {1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1};
}
}  // namespace

void bind_target(py::module_ & m)
{
  py::class_<auto_aim::Target>(m, "Target")
    .def(py::init<double, double, double, double>(), py::arg("x"), py::arg("vyaw"),
         py::arg("radius"), py::arg("h"))
    .def_static(
      "from_armor",
      [](const auto_aim::Armor & armor, double radius, int armor_num,
         const std::vector<double> & p0_dig) {
        if (p0_dig.size() != 11) {
          throw py::value_error("p0_dig must contain exactly 11 values");
        }
        Eigen::VectorXd p0(11);
        for (std::size_t i = 0; i < p0_dig.size(); ++i) {
          p0[static_cast<Eigen::Index>(i)] = p0_dig[i];
        }
        return auto_aim::Target(
          armor, std::chrono::steady_clock::now(), radius, armor_num, p0);
      },
      py::arg("armor"), py::arg("radius") = 0.2, py::arg("armor_num") = 4,
      py::arg("p0_dig") = default_p0_dig())
    .def_property_readonly("ekf_x", &auto_aim::Target::ekf_x)
    .def_property_readonly("armor_xyza_list", &auto_aim::Target::armor_xyza_list)
    .def_property_readonly("debug", [](const auto_aim::Target & self) {
      return target_debug_to_dict(self.debug());
    })
    .def_readwrite("name", &auto_aim::Target::name)
    .def_readwrite("armor_type", &auto_aim::Target::armor_type)
    .def_readwrite("priority", &auto_aim::Target::priority)
    .def_readwrite("jumped", &auto_aim::Target::jumped)
    .def_readwrite("last_id", &auto_aim::Target::last_id)
    .def_readwrite("isinit", &auto_aim::Target::isinit)
    .def("predict", py::overload_cast<double>(&auto_aim::Target::predict), py::arg("dt"))
    .def("update", &auto_aim::Target::update, py::arg("armor"))
    .def("diverged", &auto_aim::Target::diverged)
    .def("convergened", &auto_aim::Target::convergened)
    .def("checkinit", &auto_aim::Target::checkinit);
}
