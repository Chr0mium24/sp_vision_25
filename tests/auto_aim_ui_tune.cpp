#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

#include <Eigen/Geometry>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
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
  double yaw_offset_delta_deg = 0.0;
  double pitch_offset_delta_deg = 0.0;
  double offset_step_deg = 0.2;
};

struct ConfigState
{
  int min_detect_count = 5;
  int max_temp_lost_count = 15;
  int outpost_max_temp_lost_count = 75;

  double yaw_offset_deg = 0.0;
  double pitch_offset_deg = 0.0;
  double comming_angle_deg = 55.0;
  double leaving_angle_deg = 20.0;
  double decision_speed = 7.0;
  double high_speed_delay_time = 0.0;
  double low_speed_delay_time = 0.0;

  double first_tolerance_deg = 3.0;
  double second_tolerance_deg = 2.0;
  double judge_distance = 2.0;
  bool auto_fire = true;
};

enum class ParamType
{
  Int,
  Double,
  Bool
};

struct TuneParam
{
  std::string key;
  std::string label;
  ParamType type = ParamType::Double;
  double step = 0.1;
  double min_value = 0.0;
  double max_value = 0.0;
  int * int_value = nullptr;
  double * double_value = nullptr;
  bool * bool_value = nullptr;
};

struct LogState
{
  bool enabled = false;
  std::ofstream file;
  std::string path;
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

std::string trim_float(std::string value)
{
  auto pos = value.find('.');
  if (pos == std::string::npos) return value;
  while (!value.empty() && value.back() == '0') value.pop_back();
  if (!value.empty() && value.back() == '.') value.pop_back();
  return value;
}

std::string format_double(double value, int precision = 3)
{
  return trim_float(fmt::format("{:.{}}", value, precision));
}

bool replace_key_value(std::string & content, const std::string & key, const std::string & value)
{
  const std::regex re(
    "^([ \\t]*" + key + "[ \\t]*:[ \\t]*)([^#\\n]*)(.*)$",
    std::regex_constants::multiline);
  if (!std::regex_search(content, re)) return false;
  content = std::regex_replace(
    content, re, "$1" + value + "$3", std::regex_constants::format_first_only);
  return true;
}

ConfigState load_config_state(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  ConfigState state{};
  state.min_detect_count = yaml["min_detect_count"].as<int>();
  state.max_temp_lost_count = yaml["max_temp_lost_count"].as<int>();
  state.outpost_max_temp_lost_count = yaml["outpost_max_temp_lost_count"].as<int>();

  state.yaw_offset_deg = yaml["yaw_offset"].as<double>();
  state.pitch_offset_deg = yaml["pitch_offset"].as<double>();
  state.comming_angle_deg = yaml["comming_angle"].as<double>();
  state.leaving_angle_deg = yaml["leaving_angle"].as<double>();
  state.decision_speed = yaml["decision_speed"].as<double>();
  state.high_speed_delay_time = yaml["high_speed_delay_time"].as<double>();
  state.low_speed_delay_time = yaml["low_speed_delay_time"].as<double>();

  state.first_tolerance_deg = yaml["first_tolerance"].as<double>();
  state.second_tolerance_deg = yaml["second_tolerance"].as<double>();
  state.judge_distance = yaml["judge_distance"].as<double>();
  state.auto_fire = yaml["auto_fire"].as<bool>();

  return state;
}

std::vector<TuneParam> build_tune_params(ConfigState & state)
{
  std::vector<TuneParam> params;
  params.push_back(
    {"min_detect_count", "min_detect_count", ParamType::Int, 1, 1, 100, &state.min_detect_count});
  params.push_back(
    {"max_temp_lost_count", "max_temp_lost_count", ParamType::Int, 1, 1, 200,
     &state.max_temp_lost_count});
  params.push_back(
    {"outpost_max_temp_lost_count", "outpost_max_temp_lost_count", ParamType::Int, 1, 1, 500,
     &state.outpost_max_temp_lost_count});

  params.push_back(
    {"yaw_offset", "yaw_offset(deg)", ParamType::Double, 0.1, -20, 20, nullptr,
     &state.yaw_offset_deg});
  params.push_back(
    {"pitch_offset", "pitch_offset(deg)", ParamType::Double, 0.1, -20, 20, nullptr,
     &state.pitch_offset_deg});
  params.push_back(
    {"comming_angle", "comming_angle(deg)", ParamType::Double, 1.0, 0, 180, nullptr,
     &state.comming_angle_deg});
  params.push_back(
    {"leaving_angle", "leaving_angle(deg)", ParamType::Double, 1.0, 0, 180, nullptr,
     &state.leaving_angle_deg});
  params.push_back(
    {"decision_speed", "decision_speed", ParamType::Double, 0.1, 0, 50, nullptr,
     &state.decision_speed});
  params.push_back(
    {"high_speed_delay_time", "high_speed_delay_time", ParamType::Double, 0.005, 0, 1, nullptr,
     &state.high_speed_delay_time});
  params.push_back(
    {"low_speed_delay_time", "low_speed_delay_time", ParamType::Double, 0.005, 0, 1, nullptr,
     &state.low_speed_delay_time});

  params.push_back(
    {"first_tolerance", "first_tolerance(deg)", ParamType::Double, 0.1, 0, 10, nullptr,
     &state.first_tolerance_deg});
  params.push_back(
    {"second_tolerance", "second_tolerance(deg)", ParamType::Double, 0.1, 0, 10, nullptr,
     &state.second_tolerance_deg});
  params.push_back(
    {"judge_distance", "judge_distance", ParamType::Double, 0.1, 0, 10, nullptr,
     &state.judge_distance});
  params.push_back(
    {"auto_fire", "auto_fire", ParamType::Bool, 1, 0, 1, nullptr, nullptr, &state.auto_fire});
  return params;
}

void clamp_value(TuneParam & param)
{
  if (param.type == ParamType::Int && param.int_value) {
    int value = *param.int_value;
    value = std::max(static_cast<int>(param.min_value), value);
    if (param.max_value > param.min_value) {
      value = std::min(static_cast<int>(param.max_value), value);
    }
    *param.int_value = value;
    return;
  }
  if (param.type == ParamType::Double && param.double_value) {
    double value = *param.double_value;
    value = std::max(param.min_value, value);
    if (param.max_value > param.min_value) {
      value = std::min(param.max_value, value);
    }
    *param.double_value = value;
  }
}

void adjust_param(TuneParam & param, double direction)
{
  if (param.type == ParamType::Int && param.int_value) {
    int delta = static_cast<int>(param.step * direction);
    if (delta == 0) delta = direction > 0 ? 1 : -1;
    *param.int_value += delta;
    clamp_value(param);
    return;
  }
  if (param.type == ParamType::Double && param.double_value) {
    *param.double_value += param.step * direction;
    clamp_value(param);
    return;
  }
  if (param.type == ParamType::Bool && param.bool_value) {
    *param.bool_value = !(*param.bool_value);
  }
}

std::string format_param_value(const TuneParam & param)
{
  if (param.type == ParamType::Int && param.int_value) {
    return fmt::format("{}", *param.int_value);
  }
  if (param.type == ParamType::Double && param.double_value) {
    return format_double(*param.double_value, 3);
  }
  if (param.type == ParamType::Bool && param.bool_value) {
    return *param.bool_value ? "true" : "false";
  }
  return "-";
}

bool export_config(
  const std::string & config_path, const ConfigState & state, const UiState & ui,
  std::string & out_path, std::string & error)
{
  std::ifstream input(config_path);
  if (!input.is_open()) {
    error = "failed to open config";
    return false;
  }

  std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  input.close();

  double yaw_export = state.yaw_offset_deg + ui.yaw_offset_delta_deg;
  double pitch_export = state.pitch_offset_deg + ui.pitch_offset_delta_deg;

  std::vector<std::string> missing_keys;
  auto replace_or_mark = [&](const std::string & key, const std::string & value) {
    if (!replace_key_value(content, key, value)) {
      missing_keys.push_back(key);
    }
  };

  replace_or_mark("min_detect_count", fmt::format("{}", state.min_detect_count));
  replace_or_mark("max_temp_lost_count", fmt::format("{}", state.max_temp_lost_count));
  replace_or_mark(
    "outpost_max_temp_lost_count", fmt::format("{}", state.outpost_max_temp_lost_count));

  replace_or_mark("yaw_offset", format_double(yaw_export, 3));
  replace_or_mark("pitch_offset", format_double(pitch_export, 3));
  replace_or_mark("comming_angle", format_double(state.comming_angle_deg, 3));
  replace_or_mark("leaving_angle", format_double(state.leaving_angle_deg, 3));
  replace_or_mark("decision_speed", format_double(state.decision_speed, 3));
  replace_or_mark("high_speed_delay_time", format_double(state.high_speed_delay_time, 3));
  replace_or_mark("low_speed_delay_time", format_double(state.low_speed_delay_time, 3));

  replace_or_mark("first_tolerance", format_double(state.first_tolerance_deg, 3));
  replace_or_mark("second_tolerance", format_double(state.second_tolerance_deg, 3));
  replace_or_mark("judge_distance", format_double(state.judge_distance, 3));
  replace_or_mark("auto_fire", state.auto_fire ? "true" : "false");

  std::filesystem::path src_path(config_path);
  auto dir = src_path.parent_path();
  auto stem = src_path.stem().string();
  auto ext = src_path.extension().string();
  if (ext.empty()) ext = ".yaml";
  auto filename = fmt::format("{}_{}{}", stem, timestamp_string(), ext);
  auto output_path = dir / filename;

  std::ofstream output(output_path);
  if (!output.is_open()) {
    error = "failed to write config";
    return false;
  }
  output << content;
  output.close();

  out_path = output_path.string();
  if (!missing_keys.empty()) {
    std::string joined;
    for (size_t i = 0; i < missing_keys.size(); ++i) {
      if (i > 0) joined += ", ";
      joined += missing_keys[i];
    }
    error = fmt::format("missing keys: {}", joined);
    return false;
  }
  return true;
}

void toggle_log(LogState & log_state)
{
  if (log_state.enabled) {
    log_state.file.close();
    log_state.enabled = false;
    return;
  }

  std::filesystem::create_directories("logs");
  log_state.path = fmt::format("logs/auto_aim_ui_{}.jsonl", timestamp_string());
  log_state.file.open(log_state.path, std::ios::out);
  if (log_state.file.is_open()) {
    log_state.enabled = true;
  }
}

void print_tui(
  const UiState & ui, const ConfigState & cfg, const io::GimbalState & gs,
  const Eigen::Vector3d & ypr_deg, const io::Command & command, size_t target_count,
  const std::string & tracker_state, const std::vector<TuneParam> & tune_params,
  size_t selected_param, const std::string & save_status, bool log_on, double dt)
{
  std::fputs("\033[2J\033[H", stdout);
  std::printf(
    "Auto Aim UI Tune\n"
    "dt: %.1fms  tracking:%d  fric:%d  fire_mode:%u(%s)  pulse:%d  log:%d\n"
    "bullet_speed: %.2f (step %.2f)  offset_step: %.2fdeg\n"
    "offset_delta (deg): yaw:%+.2f  pitch:%+.2f\n"
    "offset_yaml  (deg): yaw:%+.2f  pitch:%+.2f  export: yaw:%+.2f pitch:%+.2f\n"
    "cmd   (deg): yaw:%+.2f  pitch:%+.2f  control:%d  targets:%zu  state:%s\n"
    "fb    (deg): yaw:%+.2f  pitch:%+.2f  roll:%+.2f\n"
    "fb    (rad): yaw:%+.3f  pitch:%+.3f  roll:%+.3f  yaw_vel:%+.3f  pitch_vel:%+.3f\n"
    "Realtime: bullet_speed, offset_delta, tracking/fric/fire\n"
    "Restart: tune params (j/k select, -/= adjust, u toggle), R export\n",
    dt * 1e3, ui.tracking ? 1 : 0, ui.fric_on ? 1 : 0, ui.fire_mode,
    fire_mode_name(ui.fire_mode), ui.fire_pulse ? 1 : 0, log_on ? 1 : 0, ui.bullet_speed,
    ui.speed_step, ui.offset_step_deg, ui.yaw_offset_delta_deg, ui.pitch_offset_delta_deg,
    cfg.yaw_offset_deg, cfg.pitch_offset_deg, cfg.yaw_offset_deg + ui.yaw_offset_delta_deg,
    cfg.pitch_offset_deg + ui.pitch_offset_delta_deg, command.yaw * 57.3, command.pitch * 57.3,
    command.control ? 1 : 0, target_count, tracker_state.c_str(), ypr_deg[0], ypr_deg[1],
    ypr_deg[2], gs.yaw, gs.pitch, gs.roll, gs.yaw_vel, gs.pitch_vel);

  for (size_t i = 0; i < tune_params.size(); ++i) {
    const auto & param = tune_params[i];
    const char * prefix = (i == selected_param) ? ">" : " ";
    auto value = format_param_value(param);
    std::printf("%s %-26s : %s\n", prefix, param.label.c_str(), value.c_str());
  }

  if (!save_status.empty()) {
    std::printf("Save: %s\n", save_status.c_str());
  }

  std::printf(
    "Keys: q quit | w/s or Up/Down pitch_delta +/- | a/d or Left/Right yaw_delta -/+ | [/] step\n"
    "      z/x bullet_speed -/+ | ,/. speed_step | 0 reset_offset | p reset_speed | c tracking\n"
    "      r fric | 1 off 2 ready 3 single 4 fire | f toggle fire | space single pulse\n"
    "      j/k select | -/= adjust | u toggle | R save | L log\n");
  std::fflush(stdout);
}

}  // namespace

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }"
  "{show s         | false  | 是否显示调试窗口}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  bool show = cli.get<bool>("show");

  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  UiState ui;
  ConfigState cfg = load_config_state(config_path);
  auto tune_params = build_tune_params(cfg);
  size_t selected_param = 0;

  TerminalRawMode terminal;
  terminal.enable();

  bool use_gui = show;
  if (use_gui) {
    try {
      cv::namedWindow("Auto Aim UI Tune", cv::WINDOW_NORMAL);
      cv::resizeWindow("Auto Aim UI Tune", 1280, 720);
    } catch (const cv::Exception &) {
      use_gui = false;
    }
  }

  LogState log_state;
  std::string save_status;

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  auto t0 = std::chrono::steady_clock::now();
  auto last_loop = t0;

  while (!exiter.exit()) {
    camera.read(img, t);
    if (img.empty()) continue;

    auto now = std::chrono::steady_clock::now();
    auto dt = tools::delta_time(now, last_loop);
    last_loop = now;

    auto gs = gimbal.state();
    auto q = gimbal.q(t - 1ms);
    solver.set_R_gimbal2world(q);

    auto armors = detector.detect(img);
    auto targets = tracker.track(armors, t);
    auto command = aimer.aim(targets, t, ui.bullet_speed);

    double yaw_offset = ui.yaw_offset_delta_deg / 57.3;
    double pitch_offset = ui.pitch_offset_delta_deg / 57.3;
    double send_yaw = command.yaw + yaw_offset;
    double send_pitch = command.pitch + pitch_offset;

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
    gimbal.send(plan);

    Eigen::Vector3d ypr_deg = tools::eulers(q, 2, 1, 0) * 57.3;
    print_tui(
      ui, cfg, gs, ypr_deg, command, targets.size(), tracker.state(), tune_params,
      selected_param, save_status, log_state.enabled, dt);

    if (log_state.enabled && log_state.file.is_open()) {
      nlohmann::json data;
      data["t"] = tools::delta_time(now, t0);
      data["dt"] = dt;
      data["targets"] = targets.size();
      data["tracker_state"] = tracker.state();
      data["command_yaw"] = command.yaw;
      data["command_pitch"] = command.pitch;
      data["send_yaw"] = send_yaw;
      data["send_pitch"] = send_pitch;
      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch_vel"] = gs.pitch_vel;
      data["yaw_err"] = send_yaw - gs.yaw;
      data["pitch_err"] = send_pitch - gs.pitch;
      data["bullet_speed"] = ui.bullet_speed;
      data["yaw_offset_base_deg"] = cfg.yaw_offset_deg;
      data["pitch_offset_base_deg"] = cfg.pitch_offset_deg;
      data["yaw_offset_delta_deg"] = ui.yaw_offset_delta_deg;
      data["pitch_offset_delta_deg"] = ui.pitch_offset_delta_deg;
      data["fire_mode"] = ui.fire_mode;
      data["tracking"] = ui.tracking;
      data["fric_on"] = ui.fric_on;
      data["auto_fire"] = cfg.auto_fire;
      if (!armors.empty()) {
        const auto & armor = armors.front();
        data["armor_center_norm_x"] = armor.center_norm.x;
        data["armor_center_norm_y"] = armor.center_norm.y;
        data["armor_confidence"] = armor.confidence;
      }
      log_state.file << data.dump() << "\n";
      log_state.file.flush();
    }

    int key = -1;
    auto ev = read_key();
    if (ev.key == Key::Quit) break;
    if (ev.key == Key::Char) key = ev.ch;
    if (ev.key == Key::Left) key = 81;
    if (ev.key == Key::Right) key = 83;
    if (ev.key == Key::Up) key = 82;
    if (ev.key == Key::Down) key = 84;

    if (use_gui) {
      if (!targets.empty()) {
        auto target = targets.front();
        tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

        for (const auto & xyza : target.armor_xyza_list()) {
          auto image_points =
            solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 255, 0});
        }

        auto aim_point = aimer.debug_aim_point;
        if (aim_point.valid) {
          auto image_points =
            solver.reproject_armor(
              aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 0, 255});
        }
      }

      if (!tune_params.empty()) {
        const auto & param = tune_params[selected_param];
        tools::draw_text(
          img, fmt::format("sel:{}={}", param.label, format_param_value(param)), {10, 60},
          {255, 255, 0}, 0.7, 2);
      }

      tools::draw_text(
        img,
        fmt::format(
          "spd:{:.2f} off_y:{:+.2f} off_p:{:+.2f} fire:{} log:{}",
          ui.bullet_speed, ui.yaw_offset_delta_deg, ui.pitch_offset_delta_deg,
          fire_mode_name(ui.fire_mode), log_state.enabled ? 1 : 0),
        {10, 90}, {255, 255, 0}, 0.7, 2);

      cv::resize(img, img, {}, 0.5, 0.5);
      cv::imshow("Auto Aim UI Tune", img);
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
      ui.yaw_offset_delta_deg = 0.0;
      ui.pitch_offset_delta_deg = 0.0;
    }
    if (key == 'p') ui.bullet_speed = 25.0;

    if (key == '[') ui.offset_step_deg = std::max(0.01, ui.offset_step_deg - 0.05);
    if (key == ']') ui.offset_step_deg = std::min(5.0, ui.offset_step_deg + 0.05);

    if (key == ',') ui.speed_step = std::max(0.1, ui.speed_step - 0.1);
    if (key == '.') ui.speed_step = std::min(10.0, ui.speed_step + 0.1);

    if (key == 'z') ui.bullet_speed = std::max(0.0, ui.bullet_speed - ui.speed_step);
    if (key == 'x') ui.bullet_speed += ui.speed_step;

    if (key == 'a' || key == 81) ui.yaw_offset_delta_deg -= ui.offset_step_deg;
    if (key == 'd' || key == 83) ui.yaw_offset_delta_deg += ui.offset_step_deg;
    if (key == 'w' || key == 82) ui.pitch_offset_delta_deg += ui.offset_step_deg;
    if (key == 's' || key == 84) ui.pitch_offset_delta_deg -= ui.offset_step_deg;

    if (key == 'j' && !tune_params.empty()) {
      selected_param = (selected_param + 1) % tune_params.size();
    }
    if (key == 'k' && !tune_params.empty()) {
      selected_param =
        (selected_param == 0) ? tune_params.size() - 1 : selected_param - 1;
    }
    if ((key == '-' || key == '_') && !tune_params.empty()) {
      adjust_param(tune_params[selected_param], -1.0);
    }
    if ((key == '=' || key == '+') && !tune_params.empty()) {
      adjust_param(tune_params[selected_param], 1.0);
    }
    if (key == 'u' && !tune_params.empty()) {
      if (tune_params[selected_param].type == ParamType::Bool) {
        adjust_param(tune_params[selected_param], 1.0);
      }
    }
    if (key == 'R') {
      std::string out_path;
      std::string error;
      bool ok = export_config(config_path, cfg, ui, out_path, error);
      if (ok) {
        save_status = fmt::format("saved: {}", out_path);
      } else {
        save_status = error.empty() ? "save failed" : fmt::format("save failed: {}", error);
      }
    }
    if (key == 'L') {
      toggle_log(log_state);
      save_status = log_state.enabled ? fmt::format("log: {}", log_state.path) : "log: off";
    }

    std::this_thread::sleep_for(5ms);
  }

  if (log_state.file.is_open()) log_state.file.close();

  io::VisionToGimbal stop{};
  stop.tracking = 0;
  stop.yaw = 0;
  stop.pitch = 0;
  stop.fire = 0;
  stop.fric_on = 0;
  gimbal.send(stop);

  return 0;
}
