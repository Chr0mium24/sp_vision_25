#include <chrono>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "io/cboard.hpp"
#include "io/command.hpp"
#include "tasks/auto_aim/aimer.hpp"

namespace py = pybind11;

namespace
{
py::dict aim_choice_debug_to_dict(const auto_aim::AimerAimChoiceDebug & debug)
{
  py::dict out;
  out["valid"] = debug.valid;
  out["chosen_id"] = debug.chosen_id;
  out["center_yaw"] = debug.center_yaw;
  out["abs_vyaw"] = debug.abs_vyaw;
  out["coming_angle"] = debug.coming_angle;
  out["leaving_angle"] = debug.leaving_angle;
  out["lock_id_before"] = debug.lock_id_before;
  out["lock_id_after"] = debug.lock_id_after;
  out["jumped"] = debug.jumped;
  out["low_spin"] = debug.low_spin;
  out["reason"] = debug.reason;
  out["delta_angle_list"] = debug.delta_angle_list;
  out["candidate_ids"] = debug.candidate_ids;
  out["armor_xyza_list"] = debug.armor_xyza_list;
  return out;
}

py::dict iteration_debug_to_dict(const auto_aim::AimerIterationDebug & debug)
{
  py::dict out;
  out["iter"] = debug.iter;
  out["previous_fly_time"] = debug.previous_fly_time;
  out["predict_dt"] = debug.predict_dt;
  out["ekf_x_after_predict"] = debug.ekf_x_after_predict;
  out["choice"] = aim_choice_debug_to_dict(debug.choice);
  out["xyz"] = debug.xyz;
  out["horizontal_distance"] = debug.horizontal_distance;
  out["trajectory_unsolvable"] = debug.trajectory_unsolvable;
  out["trajectory_pitch"] = debug.trajectory_pitch;
  out["trajectory_fly_time"] = debug.trajectory_fly_time;
  out["converged"] = debug.converged;
  return out;
}

py::dict aimer_debug_to_dict(const auto_aim::AimerDebug & debug)
{
  py::dict out;
  out["valid"] = debug.valid;
  out["has_target"] = debug.has_target;
  out["to_now"] = debug.to_now;
  out["converged"] = debug.converged;
  out["final_valid"] = debug.final_valid;
  out["bullet_speed_input"] = debug.bullet_speed_input;
  out["bullet_speed_used"] = debug.bullet_speed_used;
  out["delay_time"] = debug.delay_time;
  out["now_delay"] = debug.now_delay;
  out["future_dt"] = debug.future_dt;
  out["target_vyaw"] = debug.target_vyaw;
  out["yaw_offset"] = debug.yaw_offset;
  out["pitch_offset"] = debug.pitch_offset;
  out["fail_reason"] = debug.fail_reason;
  out["ekf_x_before"] = debug.ekf_x_before;
  out["initial_choice"] = aim_choice_debug_to_dict(debug.initial_choice);
  out["initial_trajectory_unsolvable"] = debug.initial_trajectory_unsolvable;
  out["initial_horizontal_distance"] = debug.initial_horizontal_distance;
  out["initial_pitch"] = debug.initial_pitch;
  out["initial_fly_time"] = debug.initial_fly_time;
  py::list iterations;
  for (const auto & iteration : debug.iterations) {
    iterations.append(iteration_debug_to_dict(iteration));
  }
  out["iterations"] = iterations;
  out["final_xyz"] = debug.final_xyz;
  out["final_yaw_no_offset"] = debug.final_yaw_no_offset;
  out["final_pitch_no_offset"] = debug.final_pitch_no_offset;
  out["final_command"] = py::make_tuple(
    debug.final_command.control, debug.final_command.shoot, debug.final_command.yaw,
    debug.final_command.pitch, debug.final_command.horizon_distance);
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

std::list<auto_aim::Target> targets_from_sequence(const py::sequence & sequence)
{
  std::list<auto_aim::Target> targets;
  for (const auto & item : sequence) {
    targets.push_back(item.cast<auto_aim::Target>());
  }
  return targets;
}
}  // namespace

void bind_aimer(py::module_ & m)
{
  py::class_<auto_aim::AimPoint>(m, "AimPoint")
    .def_readwrite("valid", &auto_aim::AimPoint::valid)
    .def_readwrite("xyza", &auto_aim::AimPoint::xyza);

  py::class_<auto_aim::Aimer>(m, "Aimer")
    .def(py::init<const std::string &>(), py::arg("config_path"))
    .def(
      "aim",
      [](auto_aim::Aimer & self, const py::sequence & targets, double bullet_speed,
         bool to_now) {
        auto command =
          self.aim(targets_from_sequence(targets), std::chrono::steady_clock::now(), bullet_speed,
                   to_now);
        return py::make_tuple(command.control, command.shoot, command.yaw, command.pitch,
                              command.horizon_distance);
      },
      py::arg("targets"), py::arg("bullet_speed"), py::arg("to_now") = true)
    .def(
      "aim_with_mode",
      [](auto_aim::Aimer & self, const py::sequence & targets, double bullet_speed,
         io::ShootMode shoot_mode, bool to_now) {
        auto command = self.aim(
          targets_from_sequence(targets), std::chrono::steady_clock::now(), bullet_speed,
          shoot_mode, to_now);
        return py::make_tuple(command.control, command.shoot, command.yaw, command.pitch,
                              command.horizon_distance);
      },
      py::arg("targets"), py::arg("bullet_speed"), py::arg("shoot_mode"),
      py::arg("to_now") = true)
    .def_property_readonly("debug", [](const auto_aim::Aimer & self) {
      return aimer_debug_to_dict(self.debug());
    })
    .def_readwrite("debug_aim_point", &auto_aim::Aimer::debug_aim_point);
}
