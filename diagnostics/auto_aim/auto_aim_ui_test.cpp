#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
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

std::string timestamp_string()
{
  auto now = std::chrono::system_clock::now();
  auto tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &tt);
#else
  localtime_r(&tt, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
  return buf;
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

void print_tui(
  const UiState & ui, const io::GimbalState & gs, const Eigen::Vector3d & ypr_deg,
  const io::Command & command, size_t target_count, const std::string & tracker_state, double dt,
  bool no_send, double send_yaw_deg, double send_pitch_deg, double delta_yaw_deg,
  double delta_pitch_deg)
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
    "      1 off 2 ready 3 single 4 fire | f toggle fire | space single pulse\n",
    dt * 1e3, ui.tracking ? 1 : 0, ui.fric_on ? 1 : 0, ui.fire_mode,
    fire_mode_name(ui.fire_mode), ui.fire_pulse ? 1 : 0, no_send ? 1 : 0, ui.bullet_speed, ui.speed_step,
    ui.offset_step_deg, ui.yaw_offset_deg, ui.pitch_offset_deg, command.yaw * 57.3,
    command.pitch * 57.3, command.control ? 1 : 0, target_count, tracker_state.c_str(),
    send_yaw_deg, send_pitch_deg, delta_yaw_deg, delta_pitch_deg,
    ypr_deg[0], ypr_deg[1], ypr_deg[2], gs.yaw, gs.pitch, gs.roll, gs.yaw_vel, gs.pitch_vel);
  std::fflush(stdout);
}

}  // namespace

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }"
  "{show s         | false  | 是否显示调试窗口}"
  "{no-send        | false  | 只计算目标角，不下发给云台}"
  "{duration-ms    | 0      | 运行时长(ms), 0为无限}"
  "{log-jsonl      |        | 导出jsonl日志路径，留空则自动生成}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  bool show = cli.get<bool>("show");
  bool no_send = cli.get<bool>("no-send");
  int duration_ms = cli.get<int>("duration-ms");
  std::string log_jsonl = cli.get<std::string>("log-jsonl");

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
  auto start_t = std::chrono::steady_clock::now();
  auto last_loop = start_t;

  std::ofstream log_file;
  std::string log_path;
  if (!log_jsonl.empty() || duration_ms > 0) {
    std::filesystem::create_directories("logs");
    log_path =
      log_jsonl.empty() ? fmt::format("logs/auto_aim_ui_test_{}.jsonl", timestamp_string()) : log_jsonl;
    log_file.open(log_path, std::ios::out);
    if (!log_file.is_open()) {
      std::fprintf(stderr, "failed to open --log-jsonl: %s\n", log_path.c_str());
    } else {
      std::fprintf(stdout, "[auto_aim_ui_test] logging to %s\n", log_path.c_str());
    }
  }

  while (!exiter.exit()) {
    auto now = std::chrono::steady_clock::now();
    if (duration_ms > 0 && now - start_t >= std::chrono::milliseconds(duration_ms)) break;

    camera.read(img, t);
    if (img.empty()) continue;

    now = std::chrono::steady_clock::now();
    auto dt = tools::delta_time(now, last_loop);
    last_loop = now;

    auto gs = gimbal.state();
    auto q = gimbal.q(t - 1ms);
    auto output = runtime.step({img, t, q, ui.bullet_speed});
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

    if (log_file.is_open()) {
      nlohmann::json data;
      data["t"] = tools::delta_time(now, start_t);
      data["dt"] = dt;
      data["target_count"] = targets.size();
      data["tracker_state"] = tracker_state;
      data["command_control"] = command.control;
      data["command_shoot"] = command.shoot;
      data["command_yaw"] = command.yaw;
      data["command_pitch"] = command.pitch;
      data["ui_tracking"] = ui.tracking;
      data["ui_fric_on"] = ui.fric_on;
      data["ui_fire_mode"] = ui.fire_mode;
      data["ui_bullet_speed"] = ui.bullet_speed;
      data["ui_yaw_offset_deg"] = ui.yaw_offset_deg;
      data["ui_pitch_offset_deg"] = ui.pitch_offset_deg;
      data["send_tracking"] = plan.tracking;
      data["send_fire"] = plan.fire;
      data["send_fric_on"] = plan.fric_on;
      data["send_yaw"] = send_yaw;
      data["send_pitch"] = send_pitch;
      data["feedback_yaw"] = gs.yaw;
      data["feedback_pitch"] = gs.pitch;
      data["feedback_roll"] = gs.roll;
      data["feedback_yaw_vel"] = gs.yaw_vel;
      data["feedback_pitch_vel"] = gs.pitch_vel;
      data["feedback_bullet_speed"] = gs.bullet_speed;
      data["delta_yaw"] = send_yaw - gs.yaw;
      data["delta_pitch"] = send_pitch - gs.pitch;
      data["no_send"] = no_send;
      if (!output.armors.empty()) {
        const auto & armor = output.armors.front();
        data["armor_confidence"] = armor.confidence;
        data["armor_center_norm_x"] = armor.center_norm.x;
        data["armor_center_norm_y"] = armor.center_norm.y;
        data["armor_name"] = armor.name;
        data["armor_type"] = armor.type;
      }
      auto aim_point = aimer.debug_aim_point;
      data["aim_point_valid"] = aim_point.valid;
      if (aim_point.valid) {
        data["aim_x"] = aim_point.xyza[0];
        data["aim_y"] = aim_point.xyza[1];
        data["aim_z"] = aim_point.xyza[2];
        data["aim_a"] = aim_point.xyza[3];
      }
      log_file << data.dump() << "\n";
      log_file.flush();
    }

    Eigen::Vector3d ypr_deg = tools::eulers(q, 2, 1, 0) * 57.3;
    print_tui(
      ui, gs, ypr_deg, command, targets.size(), tracker_state, dt, no_send, send_yaw_deg,
      send_pitch_deg, delta_yaw_deg, delta_pitch_deg);

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
        tools::draw_text(img, fmt::format("[{}]", tracker_state), {10, 30}, {255, 255, 255});

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

      tools::draw_text(
        img,
        fmt::format(
          "spd:{:.2f} off_y:{:+.2f} off_p:{:+.2f} fire:{} no_send:{}",
          ui.bullet_speed, ui.yaw_offset_deg, ui.pitch_offset_deg, fire_mode_name(ui.fire_mode),
          no_send ? 1 : 0),
        {10, 60}, {255, 255, 0}, 0.8, 2);
      tools::draw_text(
        img,
        fmt::format(
          "send y:{:+.2f} p:{:+.2f} d_fb y:{:+.2f} p:{:+.2f}",
          send_yaw_deg, send_pitch_deg, delta_yaw_deg, delta_pitch_deg),
        {10, 90}, {0, 255, 255}, 0.8, 2);

      cv::resize(img, img, {}, 0.5, 0.5);
      cv::imshow("Auto Aim UI Test", img);
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

    if (key == 'a' || key == 81) ui.yaw_offset_deg -= ui.offset_step_deg;
    if (key == 'd' || key == 83) ui.yaw_offset_deg += ui.offset_step_deg;
    if (key == 'w' || key == 82) ui.pitch_offset_deg += ui.offset_step_deg;
    if (key == 's' || key == 84) ui.pitch_offset_deg -= ui.offset_step_deg;

    std::this_thread::sleep_for(5ms);
  }

  io::VisionToGimbal stop{};
  stop.tracking = 0;
  stop.yaw = 0;
  stop.pitch = 0;
  stop.fire = 0;
  stop.fric_on = 0;
  if (!no_send) gimbal.send(stop);
  if (log_file.is_open()) log_file.close();

  return 0;
}
