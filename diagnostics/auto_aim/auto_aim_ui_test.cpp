#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <termios.h>

#include <Eigen/Geometry>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/auto_aim_runtime.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono_literals;

namespace
{
struct UiState
{
  bool tracking = true;
  bool fric_on = true;
  bool fire_pulse = false;
  std::chrono::steady_clock::time_point fire_pulse_until{};
  uint8_t fire_mode = 0;

  double bullet_speed = 25.0;
  double speed_step = 0.2;
  double yaw_offset_deg = 0.0;
  double pitch_offset_deg = 0.0;
  double offset_step_deg = 0.2;
};

struct SnapshotContext
{
  int snapshot_index = 0;
  int frame_index = -1;
  bool no_send = false;
  double dt = 0.0;
  double send_yaw_deg = 0.0;
  double send_pitch_deg = 0.0;
  double delta_yaw_deg = 0.0;
  double delta_pitch_deg = 0.0;
  Eigen::Quaterniond q_gimbal2world = Eigen::Quaterniond::Identity();
  Eigen::Vector3d gimbal_ypr_deg = Eigen::Vector3d::Zero();
  io::GimbalState gimbal_state{};
  io::Command command{false, false, 0, 0};
  std::string tracker_state;
};

class TerminalRawMode
{
public:
  TerminalRawMode() = default;

  bool enable()
  {
    if (!isatty(STDIN_FILENO)) return false;
    if (tcgetattr(STDIN_FILENO, &orig_) != 0) return false;

    termios raw = orig_;
    raw.c_lflag &= static_cast<unsigned long>(~(ICANON | ECHO));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) return false;

    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (flags < 0) return false;
    if (fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK) != 0) return false;

    enabled_ = true;
    return true;
  }

  ~TerminalRawMode()
  {
    if (enabled_) {
      tcsetattr(STDIN_FILENO, TCSANOW, &orig_);
    }
  }

  TerminalRawMode(const TerminalRawMode &) = delete;
  TerminalRawMode & operator=(const TerminalRawMode &) = delete;

private:
  termios orig_{};
  bool enabled_ = false;
};

enum class Key
{
  None,
  Quit,
  Left,
  Right,
  Up,
  Down,
  Char
};

struct KeyEvent
{
  Key key = Key::None;
  int ch = 0;
};

enum class FireMode : uint8_t
{
  Off = 0,
  Ready = 1,
  Single = 2,
  Fire = 3
};

const char * fire_mode_name(uint8_t mode)
{
  switch (static_cast<FireMode>(mode)) {
    case FireMode::Off:
      return "off";
    case FireMode::Ready:
      return "ready";
    case FireMode::Single:
      return "single";
    case FireMode::Fire:
      return "fire";
    default:
      return "unknown";
  }
}

KeyEvent read_key()
{
  unsigned char c = 0;
  ssize_t n = ::read(STDIN_FILENO, &c, 1);
  if (n <= 0) return {};

  if (c == 'q') return {Key::Quit, 'q'};
  if (c == 27) {
    unsigned char seq[2] = {0, 0};
    if (::read(STDIN_FILENO, &seq[0], 1) <= 0) return {};
    if (::read(STDIN_FILENO, &seq[1], 1) <= 0) return {};
    if (seq[0] != '[') return {};
    switch (seq[1]) {
      case 'A':
        return {Key::Up, 0};
      case 'B':
        return {Key::Down, 0};
      case 'C':
        return {Key::Right, 0};
      case 'D':
        return {Key::Left, 0};
      default:
        return {};
    }
  }

  return {Key::Char, static_cast<int>(c)};
}

std::string timestamp_string()
{
  auto now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return oss.str();
}

const char * color_name(auto_aim::Color color)
{
  auto index = static_cast<size_t>(color);
  return index < auto_aim::COLORS.size() ? auto_aim::COLORS[index].c_str() : "unknown";
}

const char * armor_type_name(auto_aim::ArmorType type)
{
  auto index = static_cast<size_t>(type);
  return index < auto_aim::ARMOR_TYPES.size() ? auto_aim::ARMOR_TYPES[index].c_str() : "unknown";
}

const char * armor_name(auto_aim::ArmorName name)
{
  auto index = static_cast<size_t>(name);
  return index < auto_aim::ARMOR_NAMES.size() ? auto_aim::ARMOR_NAMES[index].c_str() : "unknown";
}

template <typename Derived>
nlohmann::json eigen_vector_json(const Eigen::MatrixBase<Derived> & v)
{
  nlohmann::json out = nlohmann::json::array();
  for (int i = 0; i < v.size(); ++i) out.push_back(v(i));
  return out;
}

template <typename Derived>
nlohmann::json eigen_matrix_json(const Eigen::MatrixBase<Derived> & m)
{
  nlohmann::json out = nlohmann::json::array();
  for (int r = 0; r < m.rows(); ++r) {
    nlohmann::json row = nlohmann::json::array();
    for (int c = 0; c < m.cols(); ++c) row.push_back(m(r, c));
    out.push_back(std::move(row));
  }
  return out;
}

nlohmann::json point_json(const cv::Point2f & point) { return {point.x, point.y}; }

nlohmann::json rect_json(const cv::Rect & rect)
{
  return {{"x", rect.x}, {"y", rect.y}, {"width", rect.width}, {"height", rect.height}};
}

nlohmann::json quaternion_json(const Eigen::Quaterniond & q)
{
  return {{"w", q.w()}, {"x", q.x()}, {"y", q.y()}, {"z", q.z()}};
}

nlohmann::json armor_json(const auto_aim::Armor & armor)
{
  nlohmann::json points = nlohmann::json::array();
  for (const auto & point : armor.points) points.push_back(point_json(point));

  return {
    {"color", color_name(armor.color)},
    {"type", armor_type_name(armor.type)},
    {"name", armor_name(armor.name)},
    {"priority", armor.priority},
    {"confidence", armor.confidence},
    {"center", point_json(armor.center)},
    {"center_norm", point_json(armor.center_norm)},
    {"box", rect_json(armor.box)},
    {"points", points},
    {"ratio", armor.ratio},
    {"side_ratio", armor.side_ratio},
    {"rectangular_error", armor.rectangular_error},
    {"xyz_in_gimbal", eigen_vector_json(armor.xyz_in_gimbal)},
    {"xyz_in_world", eigen_vector_json(armor.xyz_in_world)},
    {"ypr_in_gimbal", eigen_vector_json(armor.ypr_in_gimbal)},
    {"ypr_in_world", eigen_vector_json(armor.ypr_in_world)},
    {"ypd_in_world", eigen_vector_json(armor.ypd_in_world)},
    {"yaw_raw", armor.yaw_raw}};
}

nlohmann::json solver_debug_json(const auto_aim::SolverDebug & debug)
{
  nlohmann::json yaw_search = nlohmann::json::array();
  for (const auto & sample : debug.yaw_search) {
    yaw_search.push_back(
      {{"yaw", sample.yaw}, {"error", sample.error}, {"inclined", sample.inclined}});
  }

  nlohmann::json image_points = nlohmann::json::array();
  for (const auto & point : debug.image_points) image_points.push_back(point_json(point));

  return {
    {"valid", debug.valid},
    {"color", color_name(debug.color)},
    {"type", armor_type_name(debug.type)},
    {"name", armor_name(debug.name)},
    {"image_points", image_points},
    {"xyz_in_camera", eigen_vector_json(debug.xyz_in_camera)},
    {"xyz_in_gimbal", eigen_vector_json(debug.xyz_in_gimbal)},
    {"xyz_in_world", eigen_vector_json(debug.xyz_in_world)},
    {"ypr_in_gimbal", eigen_vector_json(debug.ypr_in_gimbal)},
    {"ypr_in_world_before_opt", eigen_vector_json(debug.ypr_in_world_before_opt)},
    {"ypr_in_world_after_opt", eigen_vector_json(debug.ypr_in_world_after_opt)},
    {"ypd_in_world", eigen_vector_json(debug.ypd_in_world)},
    {"is_balance", debug.is_balance},
    {"yaw_optimized", debug.yaw_optimized},
    {"yaw_raw", debug.yaw_raw},
    {"best_yaw", debug.best_yaw},
    {"min_error", debug.min_error},
    {"search_start_yaw", debug.search_start_yaw},
    {"yaw_search", yaw_search}};
}

nlohmann::json target_debug_json(const auto_aim::TargetDebug & debug)
{
  nlohmann::json candidate_xyza_list = nlohmann::json::array();
  for (const auto & xyza : debug.last_update.candidate_xyza_list) {
    candidate_xyza_list.push_back(eigen_vector_json(xyza));
  }

  return {
    {"predict",
     {{"valid", debug.last_predict.valid},
      {"dt", debug.last_predict.dt},
      {"outpost_speed_clamped", debug.last_predict.outpost_speed_clamped},
      {"x_before", eigen_vector_json(debug.last_predict.x_before)},
      {"x_after", eigen_vector_json(debug.last_predict.x_after)},
      {"F", eigen_matrix_json(debug.last_predict.F)},
      {"Q", eigen_matrix_json(debug.last_predict.Q)}}},
    {"update",
     {{"valid", debug.last_update.valid},
      {"matched_id", debug.last_update.matched_id},
      {"last_id", debug.last_update.last_id},
      {"switch_count", debug.last_update.switch_count},
      {"update_count", debug.last_update.update_count},
      {"jumped", debug.last_update.jumped},
      {"is_switch", debug.last_update.is_switch},
      {"candidate_count", debug.last_update.candidate_count},
      {"center_yaw", debug.last_update.center_yaw},
      {"delta_angle", debug.last_update.delta_angle},
      {"x_before", eigen_vector_json(debug.last_update.x_before)},
      {"x_after", eigen_vector_json(debug.last_update.x_after)},
      {"z", eigen_vector_json(debug.last_update.z)},
      {"H", eigen_matrix_json(debug.last_update.H)},
      {"R", eigen_matrix_json(debug.last_update.R)},
      {"candidate_xyza_list", candidate_xyza_list}}}};
}

nlohmann::json target_json(const auto_aim::Target & target)
{
  nlohmann::json armor_xyza_list = nlohmann::json::array();
  for (const auto & xyza : target.armor_xyza_list()) armor_xyza_list.push_back(eigen_vector_json(xyza));

  nlohmann::json ekf_data = nlohmann::json::object();
  for (const auto & [key, value] : target.ekf().data) ekf_data[key] = value;

  return {
    {"name", armor_name(target.name)},
    {"armor_type", armor_type_name(target.armor_type)},
    {"priority", target.priority},
    {"jumped", target.jumped},
    {"last_id", target.last_id},
    {"ekf_x", eigen_vector_json(target.ekf_x())},
    {"ekf_P", eigen_matrix_json(target.ekf().P)},
    {"ekf_data", ekf_data},
    {"armor_xyza_list", armor_xyza_list},
    {"debug", target_debug_json(target.debug())}};
}

nlohmann::json tracker_debug_json(const auto_aim::TrackerDebug & debug)
{
  nlohmann::json candidates = nlohmann::json::array();
  for (const auto & armor : debug.candidates) {
    candidates.push_back(
      {{"index", armor.index},
       {"color", color_name(armor.color)},
       {"type", armor_type_name(armor.type)},
       {"name", armor_name(armor.name)},
       {"priority", armor.priority},
       {"center", point_json(armor.center)},
       {"confidence", armor.confidence}});
  }

  return {
    {"valid", debug.valid},
    {"dt", debug.dt},
    {"reset_due_to_large_dt", debug.reset_due_to_large_dt},
    {"found", debug.found},
    {"diverged", debug.diverged},
    {"bad_converge", debug.bad_converge},
    {"armors_before_filter", debug.armors_before_filter},
    {"armors_after_filter", debug.armors_after_filter},
    {"filtered_by_color", debug.filtered_by_color},
    {"matched_count", debug.matched_count},
    {"detect_count", debug.detect_count},
    {"temp_lost_count", debug.temp_lost_count},
    {"max_temp_lost_count", debug.max_temp_lost_count},
    {"prev_state", debug.prev_state},
    {"next_state", debug.next_state},
    {"operation", debug.operation},
    {"candidates", candidates},
    {"target_ekf_x", eigen_vector_json(debug.target_ekf_x)},
    {"target_debug", target_debug_json(debug.target_debug)}};
}

nlohmann::json aimer_choice_json(const auto_aim::AimerAimChoiceDebug & debug)
{
  nlohmann::json delta_angle_list = nlohmann::json::array();
  for (double delta_angle : debug.delta_angle_list) delta_angle_list.push_back(delta_angle);

  nlohmann::json candidate_ids = nlohmann::json::array();
  for (int id : debug.candidate_ids) candidate_ids.push_back(id);

  nlohmann::json armor_xyza_list = nlohmann::json::array();
  for (const auto & xyza : debug.armor_xyza_list) armor_xyza_list.push_back(eigen_vector_json(xyza));

  return {
    {"valid", debug.valid},
    {"chosen_id", debug.chosen_id},
    {"center_yaw", debug.center_yaw},
    {"abs_vyaw", debug.abs_vyaw},
    {"coming_angle", debug.coming_angle},
    {"leaving_angle", debug.leaving_angle},
    {"lock_id_before", debug.lock_id_before},
    {"lock_id_after", debug.lock_id_after},
    {"jumped", debug.jumped},
    {"low_spin", debug.low_spin},
    {"reason", debug.reason},
    {"delta_angle_list", delta_angle_list},
    {"candidate_ids", candidate_ids},
    {"armor_xyza_list", armor_xyza_list}};
}

nlohmann::json aimer_debug_json(const auto_aim::AimerDebug & debug)
{
  nlohmann::json iterations = nlohmann::json::array();
  for (const auto & iteration : debug.iterations) {
    iterations.push_back(
      {{"iter", iteration.iter},
       {"previous_fly_time", iteration.previous_fly_time},
       {"predict_dt", iteration.predict_dt},
       {"ekf_x_after_predict", eigen_vector_json(iteration.ekf_x_after_predict)},
       {"choice", aimer_choice_json(iteration.choice)},
       {"xyz", eigen_vector_json(iteration.xyz)},
       {"horizontal_distance", iteration.horizontal_distance},
       {"trajectory_unsolvable", iteration.trajectory_unsolvable},
       {"trajectory_pitch", iteration.trajectory_pitch},
       {"trajectory_fly_time", iteration.trajectory_fly_time},
       {"converged", iteration.converged}});
  }

  return {
    {"valid", debug.valid},
    {"has_target", debug.has_target},
    {"to_now", debug.to_now},
    {"converged", debug.converged},
    {"final_valid", debug.final_valid},
    {"bullet_speed_input", debug.bullet_speed_input},
    {"bullet_speed_used", debug.bullet_speed_used},
    {"delay_time", debug.delay_time},
    {"now_delay", debug.now_delay},
    {"future_dt", debug.future_dt},
    {"target_vyaw", debug.target_vyaw},
    {"yaw_offset", debug.yaw_offset},
    {"pitch_offset", debug.pitch_offset},
    {"fail_reason", debug.fail_reason},
    {"ekf_x_before", eigen_vector_json(debug.ekf_x_before)},
    {"initial_choice", aimer_choice_json(debug.initial_choice)},
    {"initial_trajectory_unsolvable", debug.initial_trajectory_unsolvable},
    {"initial_horizontal_distance", debug.initial_horizontal_distance},
    {"initial_pitch", debug.initial_pitch},
    {"initial_fly_time", debug.initial_fly_time},
    {"iterations", iterations},
    {"final_xyz", eigen_vector_json(debug.final_xyz)},
    {"final_yaw_no_offset", debug.final_yaw_no_offset},
    {"final_pitch_no_offset", debug.final_pitch_no_offset},
    {"final_command",
     {{"control", debug.final_command.control},
      {"shoot", debug.final_command.shoot},
      {"yaw", debug.final_command.yaw},
      {"pitch", debug.final_command.pitch},
      {"horizon_distance", debug.final_command.horizon_distance}}}};
}

nlohmann::json runtime_debug_json(const auto_aim::RuntimeDebug & debug)
{
  nlohmann::json detected_armors = nlohmann::json::array();
  for (const auto & armor : debug.detected_armors) detected_armors.push_back(armor_json(armor));

  return {
    {"valid", debug.valid},
    {"frame_index", debug.frame_index},
    {"bullet_speed", debug.bullet_speed},
    {"use_enemy_color", debug.use_enemy_color},
    {"to_now", debug.to_now},
    {"q_gimbal2world", quaternion_json(debug.q_gimbal2world)},
    {"R_gimbal2world", eigen_matrix_json(debug.R_gimbal2world)},
    {"detected_armors", detected_armors},
    {"solver", solver_debug_json(debug.solver)},
    {"tracker", tracker_debug_json(debug.tracker)},
    {"aimer", aimer_debug_json(debug.aimer)},
    {"command",
     {{"control", debug.command.control},
      {"shoot", debug.command.shoot},
      {"yaw", debug.command.yaw},
      {"pitch", debug.command.pitch},
      {"horizon_distance", debug.command.horizon_distance}}},
    {"tracker_state", debug.tracker_state}};
}

void render_debug_overlay(
  cv::Mat & image, const std::list<auto_aim::Target> & targets, const std::string & tracker_state,
  auto_aim::Solver & solver, auto_aim::Aimer & aimer, const UiState & ui, bool no_send,
  double send_yaw_deg, double send_pitch_deg, double delta_yaw_deg, double delta_pitch_deg)
{
  if (!targets.empty()) {
    auto target = targets.front();
    tools::draw_text(image, fmt::format("[{}]", tracker_state), {10, 30}, {255, 255, 255});

    for (const auto & xyza : target.armor_xyza_list()) {
      auto image_points =
        solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
      tools::draw_points(image, image_points, {0, 255, 0});
    }

    auto aim_point = aimer.debug_aim_point;
    if (aim_point.valid) {
      auto image_points =
        solver.reproject_armor(
          aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
      tools::draw_points(image, image_points, {0, 0, 255});
    }
  }

  tools::draw_text(
    image,
    fmt::format(
      "spd:{:.2f} off_y:{:+.2f} off_p:{:+.2f} fire:{} no_send:{}",
      ui.bullet_speed, ui.yaw_offset_deg, ui.pitch_offset_deg, fire_mode_name(ui.fire_mode),
      no_send ? 1 : 0),
    {10, 60}, {255, 255, 0}, 0.8, 2);
  tools::draw_text(
    image,
    fmt::format(
      "send y:{:+.2f} p:{:+.2f} d_fb y:{:+.2f} p:{:+.2f}",
      send_yaw_deg, send_pitch_deg, delta_yaw_deg, delta_pitch_deg),
    {10, 90}, {0, 255, 255}, 0.8, 2);
}

bool save_snapshot(
  const cv::Mat & raw_image, const cv::Mat & annotated_image, const UiState & ui,
  const SnapshotContext & snapshot, const auto_aim::RuntimeOutput & output,
  const auto_aim::RuntimeDebug & runtime_debug, std::string & out_path, std::string & error)
{
  try {
    std::filesystem::path root = "logs/auto_aim_snapshots";
    std::filesystem::create_directories(root);
    std::filesystem::path dir =
      root / fmt::format("{}_{:04d}", timestamp_string(), snapshot.snapshot_index);
    std::filesystem::create_directories(dir);

    if (!cv::imwrite((dir / "raw.png").string(), raw_image)) {
      error = "failed to save raw image";
      return false;
    }
    if (!cv::imwrite((dir / "annotated.png").string(), annotated_image)) {
      error = "failed to save annotated image";
      return false;
    }

    nlohmann::json targets = nlohmann::json::array();
    for (const auto & target : output.targets) targets.push_back(target_json(target));

    nlohmann::json tracked_armors = nlohmann::json::array();
    for (const auto & armor : output.armors) tracked_armors.push_back(armor_json(armor));

    nlohmann::json data{
      {"snapshot_index", snapshot.snapshot_index},
      {"frame_index", snapshot.frame_index},
      {"dt", snapshot.dt},
      {"ui",
       {{"tracking", ui.tracking},
        {"fric_on", ui.fric_on},
        {"fire_pulse", ui.fire_pulse},
        {"fire_mode", ui.fire_mode},
        {"bullet_speed", ui.bullet_speed},
        {"speed_step", ui.speed_step},
        {"yaw_offset_deg", ui.yaw_offset_deg},
        {"pitch_offset_deg", ui.pitch_offset_deg},
        {"offset_step_deg", ui.offset_step_deg}}},
      {"gimbal_feedback",
       {{"yaw_rad", snapshot.gimbal_state.yaw},
        {"pitch_rad", snapshot.gimbal_state.pitch},
        {"roll_rad", snapshot.gimbal_state.roll},
        {"yaw_vel", snapshot.gimbal_state.yaw_vel},
        {"pitch_vel", snapshot.gimbal_state.pitch_vel},
        {"ypr_deg", eigen_vector_json(snapshot.gimbal_ypr_deg)},
        {"q_gimbal2world", quaternion_json(snapshot.q_gimbal2world)}}},
      {"command",
       {{"control", snapshot.command.control},
        {"shoot", snapshot.command.shoot},
        {"yaw", snapshot.command.yaw},
        {"pitch", snapshot.command.pitch},
        {"horizon_distance", snapshot.command.horizon_distance},
        {"send_yaw_deg", snapshot.send_yaw_deg},
        {"send_pitch_deg", snapshot.send_pitch_deg},
        {"delta_yaw_deg", snapshot.delta_yaw_deg},
        {"delta_pitch_deg", snapshot.delta_pitch_deg},
        {"no_send", snapshot.no_send}}},
      {"tracker_state", snapshot.tracker_state},
      {"tracked_armors", tracked_armors},
      {"targets", targets},
      {"runtime_debug", runtime_debug_json(runtime_debug)}};

    std::ofstream output_file(dir / "frame.json");
    if (!output_file.is_open()) {
      error = "failed to open frame.json";
      return false;
    }
    output_file << data.dump(2) << "\n";
    out_path = dir.string();
    return true;
  } catch (const std::exception & e) {
    error = e.what();
    return false;
  }
}

void print_tui(
  const UiState & ui, const io::GimbalState & gs, const Eigen::Vector3d & ypr_deg,
  const io::Command & command, size_t target_count, const std::string & tracker_state, double dt,
  bool no_send, double send_yaw_deg, double send_pitch_deg, double delta_yaw_deg,
  double delta_pitch_deg, const std::string & save_status)
{
  std::fputs("\033[2J\033[H", stdout);
  std::printf(
    "Auto Aim UI Test\n"
    "dt: %.1fms  tracking:%d  fric:%d  fire_mode:%u(%s)  pulse:%d  no_send:%d\n"
    "bullet_speed: %.2f (step %.2f)  offset_step: %.2fdeg\n"
    "offset (deg): yaw:%+.2f  pitch:%+.2f\n"
    "cmd   (deg): yaw:%+.2f  pitch:%+.2f  control:%d  targets:%zu  state:%s\n"
    "send  (deg): yaw:%+.2f  pitch:%+.2f  delta_to_fb: yaw:%+.2f  pitch:%+.2f\n"
    "fb    (deg): yaw:%+.2f  pitch:%+.2f  roll:%+.2f\n"
    "fb    (rad): yaw:%+.3f  pitch:%+.3f  roll:%+.3f  yaw_vel:%+.3f  pitch_vel:%+.3f\n"
    "Keys: q quit | w/s or Up/Down pitch_offset +/- | a/d or Left/Right yaw_offset -/+ | [/] step\n"
    "      z/x bullet_speed -/+ | ,/. speed_step | 0 reset_offset | p reset_speed | c tracking | r fric\n"
    "      1 off 2 ready 3 single 4 fire | f toggle fire | space single pulse | S snapshot\n",
    dt * 1e3, ui.tracking ? 1 : 0, ui.fric_on ? 1 : 0, ui.fire_mode,
    fire_mode_name(ui.fire_mode), ui.fire_pulse ? 1 : 0, no_send ? 1 : 0, ui.bullet_speed, ui.speed_step,
    ui.offset_step_deg, ui.yaw_offset_deg, ui.pitch_offset_deg, command.yaw * 57.3,
    command.pitch * 57.3, command.control ? 1 : 0, target_count, tracker_state.c_str(),
    send_yaw_deg, send_pitch_deg, delta_yaw_deg, delta_pitch_deg,
    ypr_deg[0], ypr_deg[1], ypr_deg[2], gs.yaw, gs.pitch, gs.roll, gs.yaw_vel, gs.pitch_vel);
  if (!save_status.empty()) std::printf("Snapshot: %s\n", save_status.c_str());
  std::fflush(stdout);
}

}  // namespace

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }"
  "{show s         | false  | 是否显示调试窗口}"
  "{no-send        | false  | 只计算目标角，不下发给云台}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  bool show = cli.get<bool>("show");
  bool no_send = cli.get<bool>("no-send");

  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  auto_aim::Runtime runtime(config_path, false);
  auto & solver = runtime.solver();
  auto & aimer = runtime.aimer();

  UiState ui;
  TerminalRawMode terminal;
  terminal.enable();

  bool use_gui = show;
  if (use_gui) {
    try {
      cv::namedWindow("Auto Aim UI Test", cv::WINDOW_NORMAL);
      cv::resizeWindow("Auto Aim UI Test", 1280, 720);
    } catch (const cv::Exception &) {
      use_gui = false;
    }
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  auto last_loop = std::chrono::steady_clock::now();
  int frame_index = 0;
  int snapshot_index = 0;
  std::string save_status;

  while (!exiter.exit()) {
    camera.read(img, t);
    if (img.empty()) continue;
    cv::Mat raw_img = img.clone();

    auto now = std::chrono::steady_clock::now();
    auto dt = tools::delta_time(now, last_loop);
    last_loop = now;

    auto gs = gimbal.state();
    auto q = gimbal.q(t - 1ms);
    auto output = runtime.step({img, t, q, ui.bullet_speed, frame_index});
    const auto & runtime_debug = runtime.debug();
    const auto & targets = output.targets;
    const auto & command = output.command;
    const auto & tracker_state = output.tracker_state;

    double yaw_offset = ui.yaw_offset_deg / 57.3;
    double pitch_offset = ui.pitch_offset_deg / 57.3;
    double send_yaw = command.yaw + yaw_offset;
    double send_pitch = command.pitch + pitch_offset;
    double send_yaw_deg = send_yaw * 57.3;
    double send_pitch_deg = send_pitch * 57.3;
    double delta_yaw_deg = send_yaw_deg - gs.yaw * 57.3;
    double delta_pitch_deg = send_pitch_deg - gs.pitch * 57.3;

    if (ui.fire_pulse && now >= ui.fire_pulse_until) ui.fire_pulse = false;

    io::VisionToGimbal plan{};
    plan.tracking = (ui.tracking && command.control) ? 1 : 0;
    plan.yaw = static_cast<float>(send_yaw);
    plan.pitch = static_cast<float>(send_pitch);
    uint8_t fire_cmd = ui.fire_mode;
    if (ui.fire_pulse) fire_cmd = static_cast<uint8_t>(FireMode::Single);
    if (!plan.tracking) fire_cmd = static_cast<uint8_t>(FireMode::Off);
    plan.fire = fire_cmd;
    plan.fric_on = ui.fric_on ? 1 : 0;
    if (!no_send) gimbal.send(plan);

    Eigen::Vector3d ypr_deg = tools::eulers(q, 2, 1, 0) * 57.3;
    print_tui(
      ui, gs, ypr_deg, command, targets.size(), tracker_state, dt, no_send, send_yaw_deg,
      send_pitch_deg, delta_yaw_deg, delta_pitch_deg, save_status);

    int key = -1;
    auto ev = read_key();
    if (ev.key == Key::Quit) break;
    if (ev.key == Key::Char) key = ev.ch;
    if (ev.key == Key::Left) key = 81;
    if (ev.key == Key::Right) key = 83;
    if (ev.key == Key::Up) key = 82;
    if (ev.key == Key::Down) key = 84;

    cv::Mat annotated_img = raw_img.clone();
    render_debug_overlay(
      annotated_img, targets, tracker_state, solver, aimer, ui, no_send, send_yaw_deg,
      send_pitch_deg, delta_yaw_deg, delta_pitch_deg);

    if (use_gui) {
      cv::Mat gui_img;
      cv::resize(annotated_img, gui_img, {}, 0.5, 0.5);
      cv::imshow("Auto Aim UI Test", gui_img);
      int gui_key = cv::waitKey(1);
      if (gui_key != -1) key = gui_key;
    }

    if (key == 'q') break;
    if (key == 'c') ui.tracking = !ui.tracking;
    if (key == 'r') ui.fric_on = !ui.fric_on;
    if (key == '1') ui.fire_mode = static_cast<uint8_t>(FireMode::Off);
    if (key == '2') ui.fire_mode = static_cast<uint8_t>(FireMode::Ready);
    if (key == '3') ui.fire_mode = static_cast<uint8_t>(FireMode::Single);
    if (key == '4') ui.fire_mode = static_cast<uint8_t>(FireMode::Fire);
    if (key == 'f') {
      ui.fire_mode =
        (ui.fire_mode == static_cast<uint8_t>(FireMode::Fire)) ?
        static_cast<uint8_t>(FireMode::Off) :
        static_cast<uint8_t>(FireMode::Fire);
    }
    if (key == ' ') {
      ui.fire_pulse = true;
      ui.fire_pulse_until = now + 120ms;
    }
    if (key == '0') {
      ui.yaw_offset_deg = 0.0;
      ui.pitch_offset_deg = 0.0;
    }
    if (key == 'p') ui.bullet_speed = 25.0;

    if (key == '[') ui.offset_step_deg = std::max(0.01, ui.offset_step_deg - 0.05);
    if (key == ']') ui.offset_step_deg = std::min(5.0, ui.offset_step_deg + 0.05);

    if (key == ',') ui.speed_step = std::max(0.1, ui.speed_step - 0.1);
    if (key == '.') ui.speed_step = std::min(10.0, ui.speed_step + 0.1);

    if (key == 'z') ui.bullet_speed = std::max(0.0, ui.bullet_speed - ui.speed_step);
    if (key == 'x') ui.bullet_speed += ui.speed_step;
    if (key == 'S') {
      SnapshotContext snapshot{};
      snapshot.snapshot_index = ++snapshot_index;
      snapshot.frame_index = frame_index;
      snapshot.no_send = no_send;
      snapshot.dt = dt;
      snapshot.send_yaw_deg = send_yaw_deg;
      snapshot.send_pitch_deg = send_pitch_deg;
      snapshot.delta_yaw_deg = delta_yaw_deg;
      snapshot.delta_pitch_deg = delta_pitch_deg;
      snapshot.q_gimbal2world = q;
      snapshot.gimbal_ypr_deg = ypr_deg;
      snapshot.gimbal_state = gs;
      snapshot.command = command;
      snapshot.tracker_state = tracker_state;

      std::string out_path;
      std::string error;
      bool ok = save_snapshot(
        raw_img, annotated_img, ui, snapshot, output, runtime_debug, out_path, error);
      save_status = ok ? fmt::format("saved: {}", out_path)
                       : fmt::format("save failed: {}", error.empty() ? "unknown" : error);
    }

    if (key == 'a' || key == 81) ui.yaw_offset_deg -= ui.offset_step_deg;
    if (key == 'd' || key == 83) ui.yaw_offset_deg += ui.offset_step_deg;
    if (key == 'w' || key == 82) ui.pitch_offset_deg += ui.offset_step_deg;
    if (key == 's' || key == 84) ui.pitch_offset_deg -= ui.offset_step_deg;

    std::this_thread::sleep_for(5ms);
    ++frame_index;
  }

  io::VisionToGimbal stop{};
  stop.tracking = 0;
  stop.yaw = 0;
  stop.pitch = 0;
  stop.fire = 0;
  stop.fric_on = 0;
  if (!no_send) gimbal.send(stop);

  return 0;
}
