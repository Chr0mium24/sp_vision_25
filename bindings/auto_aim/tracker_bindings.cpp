#include <chrono>
#include <tuple>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tasks/auto_aim/tracker.hpp"

namespace py = pybind11;

namespace
{
py::dict tracker_armor_debug_to_dict(const auto_aim::TrackerArmorDebug & debug)
{
  py::dict out;
  out["index"] = debug.index;
  out["color"] = debug.color;
  out["type"] = debug.type;
  out["name"] = debug.name;
  out["priority"] = debug.priority;
  out["center"] = py::make_tuple(debug.center.x, debug.center.y);
  out["confidence"] = debug.confidence;
  return out;
}

py::dict tracker_debug_to_dict(const auto_aim::TrackerDebug & debug)
{
  py::dict out;
  out["valid"] = debug.valid;
  out["dt"] = debug.dt;
  out["reset_due_to_large_dt"] = debug.reset_due_to_large_dt;
  out["found"] = debug.found;
  out["diverged"] = debug.diverged;
  out["bad_converge"] = debug.bad_converge;
  out["armors_before_filter"] = debug.armors_before_filter;
  out["armors_after_filter"] = debug.armors_after_filter;
  out["filtered_by_color"] = debug.filtered_by_color;
  out["matched_count"] = debug.matched_count;
  out["detect_count"] = debug.detect_count;
  out["temp_lost_count"] = debug.temp_lost_count;
  out["max_temp_lost_count"] = debug.max_temp_lost_count;
  out["prev_state"] = debug.prev_state;
  out["next_state"] = debug.next_state;
  out["operation"] = debug.operation;

  py::list candidates;
  for (const auto & candidate : debug.candidates) {
    candidates.append(tracker_armor_debug_to_dict(candidate));
  }
  out["candidates"] = candidates;
  out["target_ekf_x"] = debug.target_ekf_x;
  return out;
}

std::list<auto_aim::Armor> armor_list_from_sequence(const py::sequence & sequence)
{
  std::list<auto_aim::Armor> armors;
  for (const auto & item : sequence) {
    armors.push_back(item.cast<auto_aim::Armor>());
  }
  return armors;
}

py::list target_list_to_python(const std::list<auto_aim::Target> & targets)
{
  py::list result;
  for (const auto & target : targets) {
    result.append(target);
  }
  return result;
}

py::dict track_result_to_dict(const std::list<auto_aim::Target> & targets, const auto_aim::Tracker & tracker)
{
  py::dict out;
  out["targets"] = target_list_to_python(targets);
  out["state"] = tracker.state();
  out["debug"] = tracker_debug_to_dict(tracker.debug());
  return out;
}
}  // namespace

void bind_tracker(py::module_ & m)
{
  py::class_<auto_aim::TrackerArmorDebug>(m, "TrackerArmorDebug")
    .def_readwrite("index", &auto_aim::TrackerArmorDebug::index)
    .def_readwrite("color", &auto_aim::TrackerArmorDebug::color)
    .def_readwrite("type", &auto_aim::TrackerArmorDebug::type)
    .def_readwrite("name", &auto_aim::TrackerArmorDebug::name)
    .def_readwrite("priority", &auto_aim::TrackerArmorDebug::priority)
    .def_readwrite("center", &auto_aim::TrackerArmorDebug::center)
    .def_readwrite("confidence", &auto_aim::TrackerArmorDebug::confidence);

  py::class_<auto_aim::TrackerDebug>(m, "TrackerDebug")
    .def_readwrite("valid", &auto_aim::TrackerDebug::valid)
    .def_readwrite("dt", &auto_aim::TrackerDebug::dt)
    .def_readwrite("reset_due_to_large_dt", &auto_aim::TrackerDebug::reset_due_to_large_dt)
    .def_readwrite("found", &auto_aim::TrackerDebug::found)
    .def_readwrite("diverged", &auto_aim::TrackerDebug::diverged)
    .def_readwrite("bad_converge", &auto_aim::TrackerDebug::bad_converge)
    .def_readwrite("armors_before_filter", &auto_aim::TrackerDebug::armors_before_filter)
    .def_readwrite("armors_after_filter", &auto_aim::TrackerDebug::armors_after_filter)
    .def_readwrite("filtered_by_color", &auto_aim::TrackerDebug::filtered_by_color)
    .def_readwrite("matched_count", &auto_aim::TrackerDebug::matched_count)
    .def_readwrite("detect_count", &auto_aim::TrackerDebug::detect_count)
    .def_readwrite("temp_lost_count", &auto_aim::TrackerDebug::temp_lost_count)
    .def_readwrite("max_temp_lost_count", &auto_aim::TrackerDebug::max_temp_lost_count)
    .def_readwrite("prev_state", &auto_aim::TrackerDebug::prev_state)
    .def_readwrite("next_state", &auto_aim::TrackerDebug::next_state)
    .def_readwrite("operation", &auto_aim::TrackerDebug::operation)
    .def_readwrite("candidates", &auto_aim::TrackerDebug::candidates)
    .def_readwrite("target_ekf_x", &auto_aim::TrackerDebug::target_ekf_x);

  py::class_<auto_aim::Tracker>(m, "Tracker")
    .def(
      py::init<const std::string &, auto_aim::Solver &>(),
      py::arg("config_path"), py::arg("solver"), py::keep_alive<1, 2>())
    .def_property_readonly("state", &auto_aim::Tracker::state)
    .def(
      "track",
      [](auto_aim::Tracker & self, const py::sequence & armors, bool use_enemy_color) {
        auto armor_list = armor_list_from_sequence(armors);
        auto targets = self.track(armor_list, std::chrono::steady_clock::now(), use_enemy_color);
        return track_result_to_dict(targets, self);
      },
      py::arg("armors"), py::arg("use_enemy_color") = true)
    .def_property_readonly("debug", [](const auto_aim::Tracker & self) {
      return tracker_debug_to_dict(self.debug());
    });
}
