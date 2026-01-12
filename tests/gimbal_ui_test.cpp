#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <thread>
#include <unistd.h>
#include <termios.h>

#include <Eigen/Geometry>
#include <fmt/core.h>
#include <opencv2/opencv.hpp>

#include "io/gimbal/gimbal.hpp"
#include "tools/exiter.hpp"
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

  double cmd_yaw = 0.0;    // rad
  double cmd_pitch = 0.0;  // rad
  double step_deg = 5.0;
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
  int ch = 0;  // valid when key == Char
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
  if (c == 27) {  // ESC
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

void print_tui(const UiState & ui, const io::GimbalState & gs, const Eigen::Vector3d & ypr_deg, double dt)
{
  // Clear screen + move cursor home
  std::fputs("\033[2J\033[H", stdout);

  std::printf(
    "Gimbal UI Test (TUI fallback)\n"
    "dt: %.1fms  tracking:%d  fric:%d  fire_mode:%u(%s)  pulse:%d  step:%.2fdeg\n"
    "CMD (deg): yaw:%+.2f  pitch:%+.2f\n"
    "FB  (deg): yaw:%+.2f  pitch:%+.2f  roll:%+.2f   (from q(t))\n"
    "FB  (rad): yaw:%+.3f  pitch:%+.3f  roll:%+.3f  yaw_odom:%+.3f  pitch_odom:%+.3f\n"
    "VEL (rad/s): yaw_vel:%+.3f  pitch_vel:%+.3f  bullet_speed:%.2f  bullet_count:%u  robot_id:%d\n"
    "\n"
    "Keys: q quit | w/s or Up/Down pitch +/- | a/d or Left/Right yaw -/+ | [/] step | 0 reset | c tracking | r fric | 1 off 2 ready 3 single 4 fire | f toggle fire | space single pulse\n",
    dt * 1e3, ui.tracking ? 1 : 0, ui.fric_on ? 1 : 0, ui.fire_mode,
    fire_mode_name(ui.fire_mode), ui.fire_pulse ? 1 : 0, ui.step_deg, ui.cmd_yaw * 57.3,
    ui.cmd_pitch * 57.3,
    ypr_deg[0], ypr_deg[1], ypr_deg[2], gs.yaw, gs.pitch, gs.roll, gs.yaw_odom, gs.pitch_odom,
    gs.yaw_vel, gs.pitch_vel, gs.bullet_speed, gs.bullet_count, static_cast<int>(gs.robot_id));
  std::fflush(stdout);
}

void draw_gauges(
  cv::Mat & canvas, double yaw_deg, double pitch_deg, bool tracking, bool fric, bool fire)
{
  canvas.setTo(cv::Scalar(15, 15, 18));

  const cv::Scalar dim(70, 70, 75);
  const cv::Scalar bright(220, 220, 220);
  const cv::Scalar accent(0, 200, 255);
  const cv::Scalar ok(0, 220, 0);
  const cv::Scalar warn(0, 0, 220);

  auto draw_dial = [&](cv::Point center, int radius, double angle_deg, cv::Scalar color) {
    cv::circle(canvas, center, radius, dim, 2, cv::LINE_AA);
    cv::circle(canvas, center, 3, bright, -1, cv::LINE_AA);

    double angle_rad = angle_deg * M_PI / 180.0;
    cv::Point tip(
      center.x + static_cast<int>(std::cos(angle_rad) * radius * 0.85),
      center.y - static_cast<int>(std::sin(angle_rad) * radius * 0.85));
    cv::line(canvas, center, tip, color, 3, cv::LINE_AA);

    for (int a = 0; a < 360; a += 30) {
      double ar = a * M_PI / 180.0;
      cv::Point p1(
        center.x + static_cast<int>(std::cos(ar) * radius * 0.95),
        center.y - static_cast<int>(std::sin(ar) * radius * 0.95));
      cv::Point p2(
        center.x + static_cast<int>(std::cos(ar) * radius * 0.80),
        center.y - static_cast<int>(std::sin(ar) * radius * 0.80));
      cv::line(canvas, p1, p2, dim, 2, cv::LINE_AA);
    }
  };

  draw_dial({260, 280}, 160, yaw_deg, accent);
  draw_dial({720, 280}, 160, pitch_deg, accent);

  // Status LEDs (no text, just indicators)
  cv::circle(canvas, {80, 60}, 14, tracking ? ok : dim, -1, cv::LINE_AA);
  cv::circle(canvas, {120, 60}, 14, fric ? ok : dim, -1, cv::LINE_AA);
  cv::circle(canvas, {160, 60}, 14, fire ? warn : dim, -1, cv::LINE_AA);
}

}  // namespace

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   |      | 位置参数，yaml配置文件路径 }"
  "{nogui          | false| 强制不使用OpenCV窗口(TUI模式)}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  bool nogui = cli.get<bool>("nogui");
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;

  io::Gimbal gimbal(config_path);
  UiState ui;

  bool use_gui = !nogui;
  if (use_gui) {
    try {
      cv::namedWindow("Gimbal UI Test", cv::WINDOW_NORMAL);
      cv::resizeWindow("Gimbal UI Test", 980, 560);
    } catch (const cv::Exception &) {
      use_gui = false;
    }
  }

  TerminalRawMode terminal;
  terminal.enable();

  auto last_loop = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    auto now = std::chrono::steady_clock::now();
    auto dt = tools::delta_time(now, last_loop);
    last_loop = now;

    auto gs = gimbal.state();
    Eigen::Quaterniond q = gimbal.q(now);
    Eigen::Vector3d ypr = tools::eulers(q, 2, 1, 0) * 57.3;

    if (ui.fire_pulse && now >= ui.fire_pulse_until) ui.fire_pulse = false;

    io::VisionToGimbal plan{};
    plan.tracking = ui.tracking ? 1 : 0;
    plan.yaw = static_cast<float>(ui.cmd_yaw);
    plan.pitch = static_cast<float>(ui.cmd_pitch);
    uint8_t fire_cmd = ui.fire_mode;
    if (ui.fire_pulse) fire_cmd = static_cast<uint8_t>(FireMode::Single);
    plan.fire = fire_cmd;
    plan.fric_on = ui.fric_on ? 1 : 0;
    gimbal.send(plan);

    print_tui(ui, gs, ypr, dt);

    int key = -1;
    auto ev = read_key();
    if (ev.key == Key::Quit) break;
    if (ev.key == Key::Char) key = ev.ch;
    if (ev.key == Key::Left) key = 81;
    if (ev.key == Key::Right) key = 83;
    if (ev.key == Key::Up) key = 82;
    if (ev.key == Key::Down) key = 84;

    if (use_gui) {
      cv::Mat canvas(560, 980, CV_8UC3);
      bool fire_active = (fire_cmd == static_cast<uint8_t>(FireMode::Single)) ||
                         (fire_cmd == static_cast<uint8_t>(FireMode::Fire));
      draw_gauges(
        canvas, ypr[0], ypr[1], ui.tracking, ui.fric_on, fire_active);
      cv::imshow("Gimbal UI Test", canvas);
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
      ui.cmd_yaw = 0.0;
      ui.cmd_pitch = 0.0;
    }

    if (key == '[') ui.step_deg = std::max(0.01, ui.step_deg - 0.05);
    if (key == ']') ui.step_deg = std::min(5.0, ui.step_deg + 0.05);

    auto step_rad = ui.step_deg / 57.3;
    if (key == 'a' || key == 81) ui.cmd_yaw -= step_rad;   // left
    if (key == 'd' || key == 83) ui.cmd_yaw += step_rad;   // right
    if (key == 'w' || key == 82) ui.cmd_pitch += step_rad; // up
    if (key == 's' || key == 84) ui.cmd_pitch -= step_rad; // down

    std::this_thread::sleep_for(5ms);
  }

  io::VisionToGimbal stop{};
  stop.tracking = 0;
  stop.yaw = 0;
  stop.pitch = 0;
  stop.fire = 0;
  stop.fric_on = 0;
  gimbal.send(stop);

  return 0;
}
