#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/concurrency/thread_safe_queue.hpp"

namespace auto_aim
{
struct TrackerArmorDebug
{
  int index = -1;
  Color color = blue;
  ArmorType type = small;
  ArmorName name = not_armor;
  ArmorPriority priority = fifth;
  cv::Point2f center;
  float confidence = 0.0f;
};

struct TrackerDebug
{
  bool valid = false;
  double dt = 0.0;
  bool reset_due_to_large_dt = false;
  bool found = false;
  bool diverged = false;
  bool bad_converge = false;
  int armors_before_filter = 0;
  int armors_after_filter = 0;
  int filtered_by_color = 0;
  int matched_count = 0;
  int detect_count = 0;
  int temp_lost_count = 0;
  int max_temp_lost_count = 0;
  std::string prev_state;
  std::string next_state;
  std::string operation;
  std::vector<TrackerArmorDebug> candidates;
  Eigen::VectorXd target_ekf_x;
  TargetDebug target_debug;
};

class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;

  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

private:
  Solver & solver_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;
  TrackerDebug debug_;

  void state_machine(bool found);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

public:
  const TrackerDebug & debug() const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP
