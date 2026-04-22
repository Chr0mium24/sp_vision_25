#include <chrono>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tasks/auto_aim/auto_aim_runtime.hpp"

namespace py = pybind11;

namespace
{
cv::Mat numpy_to_bgr_mat(const py::array & image)
{
  py::array_t<uint8_t, py::array::c_style | py::array::forcecast> array(image);
  py::buffer_info info = array.request();

  if (info.ndim != 2 && info.ndim != 3) {
    throw py::value_error("image must be a 2D grayscale or 3D BGR uint8 array");
  }

  if (info.ndim == 2) {
    cv::Mat mat(info.shape[0], info.shape[1], CV_8UC1, info.ptr);
    return mat.clone();
  }

  if (info.shape[2] != 3) {
    throw py::value_error("3D image must have shape (H, W, 3)");
  }

  cv::Mat mat(info.shape[0], info.shape[1], CV_8UC3, info.ptr);
  return mat.clone();
}

Eigen::Quaterniond quaternion_from_sequence(const py::sequence & quat)
{
  if (quat.size() != 4) {
    throw py::value_error("quat must be a 4-item sequence in w, x, y, z order");
  }
  return Eigen::Quaterniond(
    quat[0].cast<double>(), quat[1].cast<double>(), quat[2].cast<double>(),
    quat[3].cast<double>());
}

py::dict runtime_debug_to_dict(const auto_aim::RuntimeDebug & debug)
{
  py::dict out;
  out["valid"] = debug.valid;
  out["frame_index"] = debug.frame_index;
  out["bullet_speed"] = debug.bullet_speed;
  out["use_enemy_color"] = debug.use_enemy_color;
  out["to_now"] = debug.to_now;
  out["q_gimbal2world"] = debug.q_gimbal2world.coeffs();
  out["R_gimbal2world"] = debug.R_gimbal2world;
  py::list armors;
  for (const auto & armor : debug.detected_armors) {
    armors.append(armor);
  }
  out["detected_armors"] = armors;
  out["command"] = py::make_tuple(
    debug.command.control, debug.command.shoot, debug.command.yaw, debug.command.pitch,
    debug.command.horizon_distance);
  out["tracker_state"] = debug.tracker_state;
  return out;
}

py::list targets_to_list(const std::list<auto_aim::Target> & targets)
{
  py::list out;
  for (const auto & target : targets) {
    out.append(target);
  }
  return out;
}

py::list armors_to_list(const std::list<auto_aim::Armor> & armors)
{
  py::list out;
  for (const auto & armor : armors) {
    out.append(armor);
  }
  return out;
}
}  // namespace

void bind_runtime(py::module_ & m)
{
  py::class_<auto_aim::RuntimeInput>(m, "RuntimeInput");
  py::class_<auto_aim::RuntimeOutput>(m, "RuntimeOutput");

  py::class_<auto_aim::RuntimeDebug>(m, "RuntimeDebug");

  py::class_<auto_aim::Runtime>(m, "Runtime")
    .def(py::init<const std::string &, bool>(), py::arg("config_path"),
         py::arg("yolo_debug") = false)
    .def(
      "step",
      [](auto_aim::Runtime & self, const py::array & image, const py::sequence & quat,
         double bullet_speed, int frame_index, bool use_enemy_color, bool to_now) {
        cv::Mat bgr = numpy_to_bgr_mat(image);
        auto runtime_input = auto_aim::RuntimeInput{
          bgr, std::chrono::steady_clock::now(), quaternion_from_sequence(quat), bullet_speed,
          frame_index, use_enemy_color, to_now};
        auto output = self.step(runtime_input);

        py::dict out;
        out["armors"] = armors_to_list(output.armors);
        out["targets"] = targets_to_list(output.targets);
        out["command"] = py::make_tuple(
          output.command.control, output.command.shoot, output.command.yaw, output.command.pitch,
          output.command.horizon_distance);
        out["tracker_state"] = output.tracker_state;
        out["debug"] = runtime_debug_to_dict(self.debug());
        return out;
      },
      py::arg("image"), py::arg("quat_wxyz"), py::arg("bullet_speed") = 25.0,
      py::arg("frame_index") = -1, py::arg("use_enemy_color") = true, py::arg("to_now") = true)
    .def_property_readonly("debug", [](const auto_aim::Runtime & self) {
      return runtime_debug_to_dict(self.debug());
    });
}
