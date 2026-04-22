#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_crc(py::module_ & m);
void bind_armor(py::module_ & m);
void bind_ekf(py::module_ & m);
void bind_solver(py::module_ & m);

PYBIND11_MODULE(sp_vision_bindings, m)
{
  m.doc() = "Python bindings for sp_vision_25 core utilities";
  bind_crc(m);
  bind_armor(m);
  bind_ekf(m);
  bind_solver(m);
}
