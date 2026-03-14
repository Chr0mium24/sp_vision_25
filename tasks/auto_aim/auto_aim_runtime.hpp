#ifndef AUTO_AIM__AUTO_AIM_RUNTIME_HPP
#define AUTO_AIM__AUTO_AIM_RUNTIME_HPP

#include <chrono>
#include <list>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "aimer.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tracker.hpp"
#include "yolo.hpp"

namespace auto_aim
{
struct RuntimeInput
{
  const cv::Mat & image;
  std::chrono::steady_clock::time_point timestamp;
  Eigen::Quaterniond q_gimbal2world;
  double bullet_speed = 25.0;
  int frame_index = -1;
  bool use_enemy_color = true;
  bool to_now = true;
};

struct RuntimeOutput
{
  std::list<Armor> armors;
  std::list<Target> targets;
  io::Command command;
  std::string tracker_state;
};

class Runtime
{
public:
  explicit Runtime(const std::string & config_path, bool yolo_debug = false);

  RuntimeOutput step(const RuntimeInput & input);

  Solver & solver();
  Tracker & tracker();
  Aimer & aimer();
  YOLO & yolo();

private:
  YOLO yolo_;
  Solver solver_;
  Tracker tracker_;
  Aimer aimer_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AUTO_AIM_RUNTIME_HPP
