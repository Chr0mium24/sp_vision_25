#include <tuple>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tasks/auto_aim/armor.hpp"

namespace py = pybind11;

namespace
{
cv::Rect rect_from_tuple(const py::handle & handle)
{
  py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
  if (tuple.size() != 4) {
    throw py::value_error("box must be a 4-item tuple: (x, y, w, h)");
  }
  return cv::Rect(
    tuple[0].cast<int>(), tuple[1].cast<int>(), tuple[2].cast<int>(), tuple[3].cast<int>());
}

std::vector<cv::Point2f> points_from_sequence(const py::handle & handle)
{
  py::sequence seq = py::reinterpret_borrow<py::sequence>(handle);
  if (seq.size() != 4) {
    throw py::value_error("keypoints must contain exactly 4 points");
  }

  std::vector<cv::Point2f> points;
  points.reserve(4);
  for (const auto & item : seq) {
    py::sequence point = py::reinterpret_borrow<py::sequence>(item);
    if (point.size() != 2) {
      throw py::value_error("each keypoint must be a 2-item sequence");
    }
    points.emplace_back(point[0].cast<float>(), point[1].cast<float>());
  }
  return points;
}

py::tuple point_to_tuple(const cv::Point2f & point)
{
  return py::make_tuple(point.x, point.y);
}

py::tuple rect_to_tuple(const cv::Rect & rect)
{
  return py::make_tuple(rect.x, rect.y, rect.width, rect.height);
}

py::list points_to_list(const std::vector<cv::Point2f> & points)
{
  py::list result;
  for (const auto & point : points) {
    result.append(point_to_tuple(point));
  }
  return result;
}
}  // namespace

void bind_armor(py::module_ & m)
{
  py::enum_<auto_aim::Color>(m, "Color")
    .value("red", auto_aim::red)
    .value("blue", auto_aim::blue)
    .value("extinguish", auto_aim::extinguish)
    .value("purple", auto_aim::purple)
    .export_values();

  py::enum_<auto_aim::ArmorType>(m, "ArmorType")
    .value("big", auto_aim::big)
    .value("small", auto_aim::small)
    .export_values();

  py::enum_<auto_aim::ArmorName>(m, "ArmorName")
    .value("one", auto_aim::one)
    .value("two", auto_aim::two)
    .value("three", auto_aim::three)
    .value("four", auto_aim::four)
    .value("five", auto_aim::five)
    .value("sentry", auto_aim::sentry)
    .value("outpost", auto_aim::outpost)
    .value("base", auto_aim::base)
    .value("not_armor", auto_aim::not_armor)
    .export_values();

  py::enum_<auto_aim::ArmorPriority>(m, "ArmorPriority")
    .value("first", auto_aim::first)
    .value("second", auto_aim::second)
    .value("third", auto_aim::third)
    .value("forth", auto_aim::forth)
    .value("fifth", auto_aim::fifth)
    .export_values();

  py::class_<auto_aim::Lightbar>(m, "Lightbar");

  py::class_<auto_aim::Armor>(m, "Armor")
    .def(
      py::init([](int class_id, float confidence, const py::tuple & box, const py::sequence & keypoints) {
        return auto_aim::Armor(
          class_id, confidence, rect_from_tuple(box), points_from_sequence(keypoints));
      }),
      py::arg("class_id"), py::arg("confidence"), py::arg("box"), py::arg("keypoints"))
    .def_readwrite("color", &auto_aim::Armor::color)
    .def_readwrite("left", &auto_aim::Armor::left)
    .def_readwrite("right", &auto_aim::Armor::right)
    .def_property_readonly("center", [](const auto_aim::Armor & self) { return point_to_tuple(self.center); })
    .def_property_readonly(
      "center_norm", [](const auto_aim::Armor & self) { return point_to_tuple(self.center_norm); })
    .def_property_readonly("points", [](const auto_aim::Armor & self) { return points_to_list(self.points); })
    .def_property_readonly("box", [](const auto_aim::Armor & self) { return rect_to_tuple(self.box); })
    .def_readwrite("ratio", &auto_aim::Armor::ratio)
    .def_readwrite("side_ratio", &auto_aim::Armor::side_ratio)
    .def_readwrite("rectangular_error", &auto_aim::Armor::rectangular_error)
    .def_readwrite("type", &auto_aim::Armor::type)
    .def_readwrite("name", &auto_aim::Armor::name)
    .def_readwrite("priority", &auto_aim::Armor::priority)
    .def_readwrite("class_id", &auto_aim::Armor::class_id)
    .def_readwrite("confidence", &auto_aim::Armor::confidence)
    .def_readwrite("duplicated", &auto_aim::Armor::duplicated)
    .def_readwrite("xyz_in_gimbal", &auto_aim::Armor::xyz_in_gimbal)
    .def_readwrite("xyz_in_world", &auto_aim::Armor::xyz_in_world)
    .def_readwrite("ypr_in_gimbal", &auto_aim::Armor::ypr_in_gimbal)
    .def_readwrite("ypr_in_world", &auto_aim::Armor::ypr_in_world)
    .def_readwrite("ypd_in_world", &auto_aim::Armor::ypd_in_world)
    .def_readwrite("yaw_raw", &auto_aim::Armor::yaw_raw);
}
