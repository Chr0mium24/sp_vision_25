#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <termios.h>

#include <opencv2/opencv.hpp>

#include "io/gimbal/gimbal.hpp"
#include "tools/exiter.hpp"

using namespace std::chrono_literals;

namespace
{
struct UiState
{
  bool tracking = true;
  bool fric_on = true;
  uint8_t fire_mode = 0;

  double yaw_accum = 0.0;      // 本地累计，仅用于显示（rad）
  double pitch_state = 0.0;    // 发送时保持不变的 pitch（rad）
  double step_deg = 5.0;       // yaw 增量步进（deg）
  double pitch_step_deg = 1.0; // pitch 本地状态步进（deg）
};

class TerminalRawMode
{
public:
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
    if (enabled_) tcsetattr(STDIN_FILENO, TCSANOW, &orig_);
  }

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

const char * fire_mode_name(uint8_t mode)
{
  switch (mode) {
    case 0:
      return "off";
    case 1:
      return "ready";
    case 2:
      return "single";
    case 3:
      return "fire";
    default:
      return "unknown";
  }
}

void print_tui(
  const UiState & ui, const io::GimbalState & gs, const io::GimbalRxStats & rx, double dt_s,
  bool just_sent, float sent_yaw_delta)
{
  std::fputs("\033[2J\033[H", stdout);
  std::printf(
    "Gimbal Yaw-Delta Diagnose\n"
    "dt: %.1fms  tracking:%d  fric:%d  fire_mode:%u(%s)  step:%.2fdeg  pitch_step:%.2fdeg\n"
    "Local state: yaw_accum:%+.2fdeg  pitch_state:%+.2fdeg\n"
    "FB(deg): yaw:%+.2f  pitch:%+.2f  roll:%+.2f | FB(rad): yaw:%+.3f pitch:%+.3f roll:%+.3f\n"
    "Last send: %s yaw_delta:%+.3f rad (%+.2f deg)\n"
    "RX stats: good:%llu crc_fail:%llu short_read:%llu bad_header:%llu reconnect:%llu\n"
    "\n"
    "Keys: q quit | a/d or Left/Right => send yaw delta once | w/s => local pitch state +/-\n"
    "      c tracking | r fric | 1/2/3/4 fire mode | [/ ] yaw step | -/= pitch step\n",
    dt_s * 1e3, ui.tracking ? 1 : 0, ui.fric_on ? 1 : 0, ui.fire_mode, fire_mode_name(ui.fire_mode),
    ui.step_deg, ui.pitch_step_deg, ui.yaw_accum * 57.3, ui.pitch_state * 57.3, gs.yaw * 57.3,
    gs.pitch * 57.3, gs.roll * 57.3, gs.yaw, gs.pitch, gs.roll, just_sent ? "YES" : "NO",
    sent_yaw_delta, sent_yaw_delta * 57.3, static_cast<unsigned long long>(rx.good_frames),
    static_cast<unsigned long long>(rx.crc_fail), static_cast<unsigned long long>(rx.short_read),
    static_cast<unsigned long long>(rx.header_mismatch),
    static_cast<unsigned long long>(rx.reconnect_count));
  std::fflush(stdout);
}
}  // namespace

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   |      | 位置参数，yaml配置文件路径 }"
  "{loop-ms        | 5    | 每次循环sleep时长(ms) }"
  "{duration-ms    | 0    | 运行时长(ms), 0为无限 }"
  "{no-input       | false| 禁用键盘输入 }"
  "{tracking       | 1    | 初始tracking: 0/1 }"
  "{fric-on        | 1    | 初始fric_on: 0/1 }"
  "{fire-mode      | 0    | 初始fire_mode: 0/1/2/3 }"
  "{pitch-deg      | 0    | 初始pitch状态(度) }"
  "{step-deg       | 5    | yaw增量步进(度) }"
  "{pitch-step-deg | 1    | pitch本地状态步进(度) }"
  "{seed-yaw-from-feedback | false | 启动时将本地累计yaw对齐到反馈}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  const auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  const int loop_ms = std::max(1, cli.get<int>("loop-ms"));
  const int duration_ms = std::max(0, cli.get<int>("duration-ms"));
  const bool no_input = cli.get<bool>("no-input");

  tools::Exiter exiter;
  io::Gimbal gimbal(config_path, false);

  UiState ui;
  ui.tracking = cli.get<int>("tracking") != 0;
  ui.fric_on = cli.get<int>("fric-on") != 0;
  ui.fire_mode = static_cast<uint8_t>(std::clamp(cli.get<int>("fire-mode"), 0, 3));
  ui.pitch_state = cli.get<double>("pitch-deg") / 57.3;
  ui.step_deg = std::clamp(cli.get<double>("step-deg"), 0.01, 20.0);
  ui.pitch_step_deg = std::clamp(cli.get<double>("pitch-step-deg"), 0.01, 10.0);

  if (cli.get<bool>("seed-yaw-from-feedback")) {
    auto gs = gimbal.state();
    ui.yaw_accum = gs.yaw;
  }

  TerminalRawMode terminal;
  terminal.enable();

  const auto start_t = std::chrono::steady_clock::now();
  auto last_loop = start_t;
  float last_sent_yaw_delta = 0.0f;

  while (!exiter.exit()) {
    const auto now = std::chrono::steady_clock::now();
    if (duration_ms > 0 && now - start_t >= std::chrono::milliseconds(duration_ms)) break;

    const double dt_s = std::chrono::duration<double>(now - last_loop).count();
    last_loop = now;

    const auto gs = gimbal.state();
    const auto rx = gimbal.rx_stats();

    int key = -1;
    if (!no_input) {
      auto ev = read_key();
      if (ev.key == Key::Quit) break;
      if (ev.key == Key::Char) key = ev.ch;
      if (ev.key == Key::Left) key = 81;
      if (ev.key == Key::Right) key = 83;
      if (ev.key == Key::Up) key = 82;
      if (ev.key == Key::Down) key = 84;
    }

    if (key == 'q') break;
    if (key == 'c') ui.tracking = !ui.tracking;
    if (key == 'r') ui.fric_on = !ui.fric_on;
    if (key == '1') ui.fire_mode = 0;
    if (key == '2') ui.fire_mode = 1;
    if (key == '3') ui.fire_mode = 2;
    if (key == '4') ui.fire_mode = 3;
    if (key == '[') ui.step_deg = std::max(0.01, ui.step_deg - 0.05);
    if (key == ']') ui.step_deg = std::min(20.0, ui.step_deg + 0.05);
    if (key == '-') ui.pitch_step_deg = std::max(0.01, ui.pitch_step_deg - 0.05);
    if (key == '=') ui.pitch_step_deg = std::min(10.0, ui.pitch_step_deg + 0.05);

    const double pitch_step_rad = ui.pitch_step_deg / 57.3;
    if (key == 'w' || key == 82) ui.pitch_state += pitch_step_rad;
    if (key == 's' || key == 84) ui.pitch_state -= pitch_step_rad;

    bool sent_this_loop = false;
    float yaw_delta_to_send = 0.0f;
    const float yaw_step_rad = static_cast<float>(ui.step_deg / 57.3);
    if (key == 'a' || key == 81) yaw_delta_to_send = -yaw_step_rad;
    if (key == 'd' || key == 83) yaw_delta_to_send = yaw_step_rad;

    if (yaw_delta_to_send != 0.0f) {
      ui.yaw_accum += yaw_delta_to_send;

      io::VisionToGimbal plan{};
      plan.tracking = ui.tracking ? 1 : 0;
      plan.yaw = yaw_delta_to_send; // 增量发送
      plan.pitch = static_cast<float>(ui.pitch_state);
      plan.fire = ui.fire_mode;
      plan.fric_on = ui.fric_on ? 1 : 0;
      gimbal.send(plan);

      sent_this_loop = true;
      last_sent_yaw_delta = yaw_delta_to_send;
    }

    print_tui(ui, gs, rx, dt_s, sent_this_loop, last_sent_yaw_delta);
    std::this_thread::sleep_for(std::chrono::milliseconds(loop_ms));
  }

  return 0;
}
