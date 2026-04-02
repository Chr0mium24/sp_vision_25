#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{
Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
{
  auto yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;        // degree to rad
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;    // degree to rad
  comming_angle_ = yaml["comming_angle"].as<double>() / 57.3;  // degree to rad
  leaving_angle_ = yaml["leaving_angle"].as<double>() / 57.3;  // degree to rad
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
  decision_speed_ = yaml["decision_speed"].as<double>();
  if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) {
    left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() / 57.3;    // degree to rad
    right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() / 57.3;  // degree to rad
    tools::logger()->info("[Aimer] successfully loading shootmode");
  }
}

const AimerDebug & Aimer::debug() const { return debug_; }

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  debug_ = AimerDebug{};
  debug_.valid = true;
  debug_.to_now = to_now;
  debug_.bullet_speed_input = bullet_speed;
  debug_.yaw_offset = yaw_offset_;
  debug_.pitch_offset = pitch_offset_;

  if (targets.empty()) {
    debug_.fail_reason = "no_target";
    return {false, false, 0, 0};
  }
  auto target = targets.front();
  debug_.has_target = true;
  debug_.ekf_x_before = target.ekf_x();

  double delay_time =
    target.ekf_x()[7] > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;
  debug_.delay_time = delay_time;
  debug_.target_vyaw = target.ekf_x()[7];

  if (bullet_speed < 14) bullet_speed = 23;
  debug_.bullet_speed_used = bullet_speed;

  // 考虑detecor和tracker所消耗的时间，此外假设aimer的用时可忽略不计
  auto future = timestamp;
  if (to_now) {
    double dt;
    dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
    debug_.now_delay = dt;
    debug_.future_dt = dt;
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  else {
    auto dt = 0.005 + delay_time;  //detector-aimer耗时0.005+发弹延时0.1
    // tools::logger()->info("dt is {:.4f} second", dt);
    debug_.future_dt = dt;
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  auto aim_point0 = choose_aim_point(target, &debug_.initial_choice);
  debug_aim_point = aim_point0;
  if (!aim_point0.valid) {
    // tools::logger()->debug("Invalid aim_point0.");
    debug_.fail_reason = "invalid_initial_aim_point";
    return {false, false, 0, 0};
  }

  Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);
  auto d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  tools::Trajectory trajectory0(bullet_speed, d0, xyz0[2]);
  debug_.initial_horizontal_distance = d0;
  debug_.initial_trajectory_unsolvable = trajectory0.unsolvable;
  if (trajectory0.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
    debug_aim_point.valid = false;
    debug_.fail_reason = "initial_trajectory_unsolvable";
    return {false, false, 0, 0};
  }
  debug_.initial_pitch = trajectory0.pitch;
  debug_.initial_fly_time = trajectory0.fly_time;

  // 迭代求解飞行时间 (最多10次，收敛条件：相邻两次fly_time差 <0.001)
  bool converged = false;
  double prev_fly_time = trajectory0.fly_time;
  tools::Trajectory current_traj = trajectory0;
  std::vector<Target> iteration_target(10, target);  // 创建10个目标副本用于迭代预测

  for (int iter = 0; iter < 10; ++iter) {
    AimerIterationDebug iteration_debug;
    iteration_debug.iter = iter + 1;
    iteration_debug.previous_fly_time = prev_fly_time;
    iteration_debug.predict_dt = debug_.future_dt + prev_fly_time;

    // 预测目标在 future + prev_fly_time 时刻的位置
    auto predict_time = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
    iteration_target[iter].predict(predict_time);
    iteration_debug.ekf_x_after_predict = iteration_target[iter].ekf_x();

    // 计算瞄准点
    auto aim_point = choose_aim_point(iteration_target[iter], &iteration_debug.choice);
    debug_aim_point = aim_point;
    if (!aim_point.valid) {
      debug_.iterations.push_back(std::move(iteration_debug));
      debug_.fail_reason = fmt::format("invalid_iter_aim_point_{}", iter + 1);
      return {false, false, 0, 0};
    }

    // 计算新弹道
    Eigen::Vector3d xyz = aim_point.xyza.head(3);
    double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    current_traj = tools::Trajectory(bullet_speed, d, xyz.z());
    iteration_debug.xyz = xyz;
    iteration_debug.horizontal_distance = d;
    iteration_debug.trajectory_unsolvable = current_traj.unsolvable;
    iteration_debug.trajectory_pitch = current_traj.pitch;
    iteration_debug.trajectory_fly_time = current_traj.fly_time;

    // 检查弹道是否可解
    if (current_traj.unsolvable) {
      tools::logger()->debug(
        "[Aimer] Unsolvable trajectory in iter {}: speed={:.2f}, d={:.2f}, z={:.2f}", iter + 1,
        bullet_speed, d, xyz.z());
      debug_aim_point.valid = false;
      debug_.iterations.push_back(std::move(iteration_debug));
      debug_.fail_reason = fmt::format("trajectory_unsolvable_iter_{}", iter + 1);
      return {false, false, 0, 0};
    }

    // 检查收敛条件
    if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) {
      converged = true;
      iteration_debug.converged = true;
      debug_.iterations.push_back(std::move(iteration_debug));
      break;
    }
    debug_.iterations.push_back(std::move(iteration_debug));
    prev_fly_time = current_traj.fly_time;
  }
  debug_.converged = converged;

  // 计算最终角度
  Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  double pitch = current_traj.pitch + pitch_offset_;  // world frame: pitch up positive
  debug_.final_valid = true;
  debug_.final_xyz = final_xyz;
  debug_.final_yaw_no_offset = std::atan2(final_xyz.y(), final_xyz.x());
  debug_.final_pitch_no_offset = current_traj.pitch;
  debug_.final_command = {true, false, yaw, pitch};
  return debug_.final_command;
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  io::ShootMode shoot_mode, bool to_now)
{
  double yaw_offset;
  if (shoot_mode == io::left_shoot && left_yaw_offset_.has_value()) {
    yaw_offset = left_yaw_offset_.value();
  } else if (shoot_mode == io::right_shoot && right_yaw_offset_.has_value()) {
    yaw_offset = right_yaw_offset_.value();
  } else {
    yaw_offset = yaw_offset_;
  }

  auto command = aim(targets, timestamp, bullet_speed, to_now);
  command.yaw = command.yaw - yaw_offset_ + yaw_offset;

  return command;
}

AimPoint Aimer::choose_aim_point(const Target & target, AimerAimChoiceDebug * debug)
{
  Eigen::VectorXd ekf_x = target.ekf_x();
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  auto armor_num = armor_xyza_list.size();
  if (debug != nullptr) {
    *debug = AimerAimChoiceDebug{};
    debug->valid = true;
    debug->armor_xyza_list = armor_xyza_list;
    debug->jumped = target.jumped;
    debug->lock_id_before = lock_id_;
    debug->abs_vyaw = std::abs(target.ekf_x()[8]);
  }
  // 如果装甲板未发生过跳变，则只有当前装甲板的位置已知
  if (!target.jumped) {
    if (debug != nullptr) {
      debug->chosen_id = 0;
      debug->lock_id_after = lock_id_;
      debug->reason = "not_jumped";
    }
    return {true, armor_xyza_list[0]};
  }

  // 整车旋转中心的球坐标yaw
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
  if (debug != nullptr) debug->center_yaw = center_yaw;

  // 如果delta_angle为0，则该装甲板中心和整车中心的连线在世界坐标系的xy平面过原点
  std::vector<double> delta_angle_list;
  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }
  if (debug != nullptr) debug->delta_angle_list = delta_angle_list;

  // 不考虑小陀螺
  if (std::abs(target.ekf_x()[8]) <= 2 && target.name != ArmorName::outpost) {
    if (debug != nullptr) debug->low_spin = true;
    // 选择在可射击范围内的装甲板
    std::vector<int> id_list;
    for (int i = 0; i < armor_num; i++) {
      if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
      id_list.push_back(i);
    }
    if (debug != nullptr) debug->candidate_ids = id_list;
    // 绝无可能
    if (id_list.empty()) {
      tools::logger()->warn("Empty id list!");
      if (debug != nullptr) {
        debug->reason = "empty_low_spin_candidate";
        debug->valid = false;
      }
      return {false, armor_xyza_list[0]};
    }

    // 锁定模式：防止在两个都呈45度的装甲板之间来回切换
    if (id_list.size() > 1) {
      int id0 = id_list[0], id1 = id_list[1];

      // 未处于锁定模式时，选择delta_angle绝对值较小的装甲板，进入锁定模式
      if (lock_id_ != id0 && lock_id_ != id1)
        lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;

      if (debug != nullptr) {
        debug->chosen_id = static_cast<int>(lock_id_);
        debug->lock_id_after = lock_id_;
        debug->reason = "low_spin_lock";
      }
      return {true, armor_xyza_list[lock_id_]};
    }

    // 只有一个装甲板在可射击范围内时，退出锁定模式
    lock_id_ = -1;
    if (debug != nullptr) {
      debug->chosen_id = id_list[0];
      debug->lock_id_after = lock_id_;
      debug->reason = "low_spin_single_candidate";
    }
    return {true, armor_xyza_list[id_list[0]]};
  }

  double coming_angle, leaving_angle;
  if (target.name == ArmorName::outpost) {
    coming_angle = 70 / 57.3;
    leaving_angle = 30 / 57.3;
  } else {
    coming_angle = comming_angle_;
    leaving_angle = leaving_angle_;
  }
  if (debug != nullptr) {
    debug->coming_angle = coming_angle;
    debug->leaving_angle = leaving_angle;
  }

  // 在小陀螺时，一侧的装甲板不断出现，另一侧的装甲板不断消失，显然前者被打中的概率更高
  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;
    if (debug != nullptr) debug->candidate_ids.push_back(i);
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) {
      if (debug != nullptr) {
        debug->chosen_id = i;
        debug->reason = "high_spin_positive_vyaw";
        debug->lock_id_after = lock_id_;
      }
      return {true, armor_xyza_list[i]};
    }
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) {
      if (debug != nullptr) {
        debug->chosen_id = i;
        debug->reason = "high_spin_negative_vyaw";
        debug->lock_id_after = lock_id_;
      }
      return {true, armor_xyza_list[i]};
    }
  }

  if (debug != nullptr) {
    debug->reason = "no_candidate";
    debug->valid = false;
    debug->lock_id_after = lock_id_;
  }
  return {false, armor_xyza_list[0]};
}

}  // namespace auto_aim
