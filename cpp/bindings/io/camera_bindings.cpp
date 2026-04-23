#include <chrono>
#include <cstring>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "io/camera.hpp"

namespace py = pybind11;

namespace
{
py::array mat_to_numpy(const cv::Mat & input)
{
  if (input.empty()) {
    return py::array();
  }

  cv::Mat mat = input.isContinuous() ? input : input.clone();
  if (mat.type() != CV_8UC1 && mat.type() != CV_8UC3) {
    throw std::runtime_error("Camera image must be CV_8UC1 or CV_8UC3");
  }

  if (mat.channels() == 1) {
    std::vector<ssize_t> shape = {mat.rows, mat.cols};
    py::array_t<uint8_t> out(shape);
    std::memcpy(out.mutable_data(), mat.data, static_cast<size_t>(mat.total() * mat.elemSize()));
    return out;
  }

  std::vector<ssize_t> shape = {mat.rows, mat.cols, mat.channels()};
  py::array_t<uint8_t> out(shape);
  std::memcpy(out.mutable_data(), mat.data, static_cast<size_t>(mat.total() * mat.elemSize()));
  return out;
}

std::chrono::steady_clock::time_point ns_to_time_point(int64_t ns)
{
  return std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(ns));
}

int64_t time_point_to_ns(std::chrono::steady_clock::time_point timestamp)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count();
}
}  // namespace

void bind_camera(py::module_ & m)
{
  py::class_<io::Camera>(m, "Camera")
    .def(py::init<const std::string &>(), py::arg("config_path"))
    .def(
      "read",
      [](io::Camera & self) {
        cv::Mat img;
        std::chrono::steady_clock::time_point timestamp;
        self.read(img, timestamp);
        return py::make_tuple(mat_to_numpy(img), time_point_to_ns(timestamp));
      })
    .def("__repr__", [](const io::Camera &) { return py::str("Camera()"); });

  m.def(
    "steady_clock_time_point_from_ns",
    [](int64_t ns) { return ns_to_time_point(ns); },
    py::arg("ns"));
}
