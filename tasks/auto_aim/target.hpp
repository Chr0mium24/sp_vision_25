#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{
struct TargetPredictDebug
{
  bool valid = false;
  double dt = 0.0;
  bool outpost_speed_clamped = false;
  Eigen::VectorXd x_before;
  Eigen::VectorXd x_after;
  Eigen::MatrixXd F;
  Eigen::MatrixXd Q;
};

struct TargetUpdateDebug
{
  bool valid = false;
  int matched_id = -1;
  int last_id = -1;
  int switch_count = 0;
  int update_count = 0;
  bool jumped = false;
  bool is_switch = false;
  int candidate_count = 0;
  double center_yaw = 0.0;
  double delta_angle = 0.0;
  Eigen::VectorXd x_before;
  Eigen::VectorXd x_after;
  Eigen::VectorXd z;
  Eigen::MatrixXd H;
  Eigen::MatrixXd R;
  std::vector<Eigen::Vector4d> candidate_xyza_list;
};

struct TargetDebug
{
  TargetPredictDebug last_predict;
  TargetUpdateDebug last_update;
};

class Target
{
public:
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  const TargetDebug & debug() const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;
  TargetDebug debug_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
