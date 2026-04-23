#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_crc(py::module_ & m);
void bind_command(py::module_ & m);
void bind_armor(py::module_ & m);
void bind_ekf(py::module_ & m);
void bind_solver(py::module_ & m);
void bind_target(py::module_ & m);
void bind_tracker(py::module_ & m);
void bind_aimer(py::module_ & m);
void bind_runtime(py::module_ & m);
void bind_camera(py::module_ & m);
void bind_cboard(py::module_ & m);
void bind_gimbal(py::module_ & m);

PYBIND11_MODULE(sp_vision_bindings, m)
{
  m.doc() = "Python bindings for sp_vision_25 core utilities";
  bind_crc(m);
  bind_command(m);
  bind_armor(m);
  bind_ekf(m);
  bind_solver(m);
  bind_target(m);
  bind_tracker(m);
  bind_aimer(m);
  bind_runtime(m);
  bind_camera(m);
  bind_cboard(m);
  bind_gimbal(m);
}
