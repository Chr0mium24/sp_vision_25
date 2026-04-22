#include <tuple>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tasks/auto_aim/solver.hpp"

namespace py = pybind11;

namespace
{
py::tuple point_to_tuple(const cv::Point2f & point)
{
  return py::make_tuple(point.x, point.y);
}

py::list points_to_list(const std::vector<cv::Point2f> & points)
{
  py::list result;
  for (const auto & point : points) {
    result.append(point_to_tuple(point));
  }
  return result;
}

std::vector<cv::Point3f> world_points_from_sequence(const py::handle & handle)
{
  py::sequence seq = py::reinterpret_borrow<py::sequence>(handle);
  std::vector<cv::Point3f> points;
  points.reserve(seq.size());
  for (const auto & item : seq) {
    py::sequence point = py::reinterpret_borrow<py::sequence>(item);
    if (point.size() != 3) {
      throw py::value_error("each world point must be a 3-item sequence");
    }
    points.emplace_back(
      point[0].cast<float>(), point[1].cast<float>(), point[2].cast<float>());
  }
  return points;
}

py::dict solver_debug_to_dict(const auto_aim::SolverDebug & debug)
{
  py::dict out;
  out["valid"] = debug.valid;
  out["color"] = debug.color;
  out["type"] = debug.type;
  out["name"] = debug.name;
  out["image_points"] = points_to_list(debug.image_points);
  out["xyz_in_camera"] = debug.xyz_in_camera;
  out["xyz_in_gimbal"] = debug.xyz_in_gimbal;
  out["xyz_in_world"] = debug.xyz_in_world;
  out["ypr_in_gimbal"] = debug.ypr_in_gimbal;
  out["ypr_in_world_before_opt"] = debug.ypr_in_world_before_opt;
  out["ypr_in_world_after_opt"] = debug.ypr_in_world_after_opt;
  out["ypd_in_world"] = debug.ypd_in_world;
  out["is_balance"] = debug.is_balance;
  out["yaw_optimized"] = debug.yaw_optimized;
  out["yaw_raw"] = debug.yaw_raw;
  out["best_yaw"] = debug.best_yaw;
  out["min_error"] = debug.min_error;
  out["search_start_yaw"] = debug.search_start_yaw;

  py::list yaw_search;
  for (const auto & sample : debug.yaw_search) {
    py::dict item;
    item["yaw"] = sample.yaw;
    item["error"] = sample.error;
    item["inclined"] = sample.inclined;
    yaw_search.append(item);
  }
  out["yaw_search"] = yaw_search;
  return out;
}
}  // namespace

void bind_solver(py::module_ & m)
{
  py::class_<auto_aim::SolverYawSearchSample>(m, "SolverYawSearchSample")
    .def_readwrite("yaw", &auto_aim::SolverYawSearchSample::yaw)
    .def_readwrite("error", &auto_aim::SolverYawSearchSample::error)
    .def_readwrite("inclined", &auto_aim::SolverYawSearchSample::inclined);

  py::class_<auto_aim::Solver>(m, "Solver")
    .def(py::init<const std::string &>(), py::arg("config_path"))
    .def_property_readonly("R_gimbal2world", &auto_aim::Solver::R_gimbal2world)
    .def(
      "set_R_gimbal2world_quat",
      [](auto_aim::Solver & self, const py::sequence & quat) {
        if (quat.size() != 4) {
          throw py::value_error("quat must be a 4-item sequence in w, x, y, z order");
        }
        Eigen::Quaterniond q(
          quat[0].cast<double>(), quat[1].cast<double>(), quat[2].cast<double>(),
          quat[3].cast<double>());
        self.set_R_gimbal2world(q);
      },
      py::arg("quat"))
    .def("solve", &auto_aim::Solver::solve, py::arg("armor"))
    .def(
      "reproject_armor",
      [](const auto_aim::Solver & self, const Eigen::Vector3d & xyz_in_world, double yaw,
         auto_aim::ArmorType type, auto_aim::ArmorName name) {
        return points_to_list(self.reproject_armor(xyz_in_world, yaw, type, name));
      },
      py::arg("xyz_in_world"), py::arg("yaw"), py::arg("type"), py::arg("name"))
    .def(
      "world2pixel",
      [](auto_aim::Solver & self, const py::sequence & world_points) {
        return points_to_list(self.world2pixel(world_points_from_sequence(world_points)));
      },
      py::arg("world_points"))
    .def_property_readonly("debug", [](const auto_aim::Solver & self) {
      return solver_debug_to_dict(self.debug());
    });
}
