#include "auto_aim_runtime.hpp"

#include <utility>

namespace auto_aim
{
Runtime::Runtime(const std::string & config_path, bool yolo_debug)
: yolo_(config_path, yolo_debug), solver_(config_path), tracker_(config_path, solver_),
  aimer_(config_path)
{
}

RuntimeOutput Runtime::step(const RuntimeInput & input)
{
  solver_.set_R_gimbal2world(input.q_gimbal2world);

  auto armors = yolo_.detect(input.image, input.frame_index);
  auto targets = tracker_.track(armors, input.timestamp, input.use_enemy_color);
  auto command = aimer_.aim(targets, input.timestamp, input.bullet_speed, input.to_now);

  RuntimeOutput output{};
  output.armors = std::move(armors);
  output.targets = std::move(targets);
  output.command = command;
  output.tracker_state = tracker_.state();
  return output;
}

Solver & Runtime::solver() { return solver_; }

Tracker & Runtime::tracker() { return tracker_; }

Aimer & Runtime::aimer() { return aimer_; }

YOLO & Runtime::yolo() { return yolo_; }

}  // namespace auto_aim
