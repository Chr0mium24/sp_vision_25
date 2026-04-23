#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>

#include "io/cboard.hpp"
#include "io/command.hpp"
#include "target.hpp"

namespace auto_aim
{
struct AimPoint
{
  bool valid;
  Eigen::Vector4d xyza;
};

struct AimerAimChoiceDebug
{
  bool valid = false;
  int chosen_id = -1;
  double center_yaw = 0.0;
  double abs_vyaw = 0.0;
  double coming_angle = 0.0;
  double leaving_angle = 0.0;
  double lock_id_before = -1.0;
  double lock_id_after = -1.0;
  bool jumped = false;
  bool low_spin = false;
  std::string reason;
  std::vector<double> delta_angle_list;
  std::vector<int> candidate_ids;
  std::vector<Eigen::Vector4d> armor_xyza_list;
};

struct AimerIterationDebug
{
  int iter = 0;
  double previous_fly_time = 0.0;
  double predict_dt = 0.0;
  Eigen::VectorXd ekf_x_after_predict;
  AimerAimChoiceDebug choice;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  double horizontal_distance = 0.0;
  bool trajectory_unsolvable = false;
  double trajectory_pitch = 0.0;
  double trajectory_fly_time = 0.0;
  bool converged = false;
};

struct AimerDebug
{
  bool valid = false;
  bool has_target = false;
  bool to_now = true;
  bool converged = false;
  bool final_valid = false;
  double bullet_speed_input = 0.0;
  double bullet_speed_used = 0.0;
  double delay_time = 0.0;
  double now_delay = 0.0;
  double future_dt = 0.0;
  double target_vyaw = 0.0;
  double yaw_offset = 0.0;
  double pitch_offset = 0.0;
  std::string fail_reason;
  Eigen::VectorXd ekf_x_before;
  AimerAimChoiceDebug initial_choice;
  bool initial_trajectory_unsolvable = false;
  double initial_horizontal_distance = 0.0;
  double initial_pitch = 0.0;
  double initial_fly_time = 0.0;
  std::vector<AimerIterationDebug> iterations;
  Eigen::Vector3d final_xyz = Eigen::Vector3d::Zero();
  double final_yaw_no_offset = 0.0;
  double final_pitch_no_offset = 0.0;
  io::Command final_command{false, false, 0, 0};
};

class Aimer
{
public:
  AimPoint debug_aim_point;
  explicit Aimer(const std::string & config_path);
  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true);

  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    io::ShootMode shoot_mode, bool to_now = true);

  const AimerDebug & debug() const;

private:
  double yaw_offset_;
  std::optional<double> left_yaw_offset_, right_yaw_offset_;
  double pitch_offset_;
  double comming_angle_;
  double leaving_angle_;
  double lock_id_ = -1;
  double high_speed_delay_time_;
  double low_speed_delay_time_;
  double decision_speed_;
  AimerDebug debug_;

  AimPoint choose_aim_point(const Target & target, AimerAimChoiceDebug * debug = nullptr);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AIMER_HPP
