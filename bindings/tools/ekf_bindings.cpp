#include <functional>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tools/extended_kalman_filter.hpp"

namespace py = pybind11;

void bind_ekf(py::module_ & m)
{
  using EKF = tools::ExtendedKalmanFilter;
  using XAdd = std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)>;
  using FFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;

  py::class_<EKF>(m, "ExtendedKalmanFilter")
    .def(
      py::init<
        const Eigen::VectorXd &, const Eigen::MatrixXd &, XAdd>(),
      py::arg("x0"), py::arg("P0"),
      py::arg("x_add") = XAdd([](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
        return a + b;
      }))
    .def_readwrite("x", &EKF::x)
    .def_readwrite("P", &EKF::P)
    .def_readwrite("data", &EKF::data)
    .def_readwrite("window_size", &EKF::window_size)
    .def_readwrite("last_nis", &EKF::last_nis)
    .def(
      "predict",
      [](EKF & self, const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q) {
        return self.predict(F, Q);
      },
      py::arg("F"), py::arg("Q"))
    .def(
      "predict_custom",
      [](EKF & self, const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q, FFunc f) {
        return self.predict(F, Q, std::move(f));
      },
      py::arg("F"), py::arg("Q"), py::arg("f"))
    .def(
      "update",
      [](EKF & self, const Eigen::VectorXd & z, const Eigen::MatrixXd & H,
         const Eigen::MatrixXd & R) { return self.update(z, H, R); },
      py::arg("z"), py::arg("H"), py::arg("R"))
    .def(
      "update_custom",
      [](EKF & self, const Eigen::VectorXd & z, const Eigen::MatrixXd & H,
         const Eigen::MatrixXd & R, FFunc h, XAdd z_subtract) {
        return self.update(z, H, R, std::move(h), std::move(z_subtract));
      },
      py::arg("z"), py::arg("H"), py::arg("R"), py::arg("h"),
      py::arg("z_subtract") = XAdd([](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
        return a - b;
      }));
}
