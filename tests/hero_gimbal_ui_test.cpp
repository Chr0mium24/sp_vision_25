#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <thread>
#include <unistd.h>
#include <termios.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/opencv.hpp>

#include "serial/serial.h"
#include "tools/exiter.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono_literals;

namespace
{
constexpr uint8_t kFrameHeader = 0xAF;
constexpr uint8_t kFrameEnd = 0xFA;
constexpr size_t kVisionTxSize = 10;
constexpr size_t kVisionRxSize = 13;
constexpr double kPi = 3.14159265358979323846;

struct UiState
{
  double cmd_yaw_deg = 0.0;
  double cmd_pitch_deg = 0.0;
  double step_deg = 5.0;
  uint8_t shoot_mode = 0;
  uint16_t distance = 3000;
  bool single_pulse = false;
  std::chrono::steady_clock::time_point single_pulse_until{};
};

struct VisionRxFrame
{
  int16_t yaw_0p1deg = 0;
  int16_t pitch_0p1deg = 0;
  int16_t roll_0p1deg = 0;
  int16_t shoot_speed = 0;
  uint8_t color = 0;
  uint8_t mode = 0;
  uint8_t checksum = 0;
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

enum class ShootMode : uint8_t
{
  Off = 0,
  Ready = 1,
  Single = 2,
  Fire = 3
};

const char * shoot_mode_name(uint8_t mode)
{
  switch (static_cast<ShootMode>(mode)) {
    case ShootMode::Off:
      return "off";
    case ShootMode::Ready:
      return "ready";
    case ShootMode::Single:
      return "single";
    case ShootMode::Fire:
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

int16_t clamp_int16(double value)
{
  if (value > static_cast<double>(std::numeric_limits<int16_t>::max())) {
    return std::numeric_limits<int16_t>::max();
  }
  if (value < static_cast<double>(std::numeric_limits<int16_t>::min())) {
    return std::numeric_limits<int16_t>::min();
  }
  return static_cast<int16_t>(std::lround(value));
}

std::array<uint8_t, kVisionTxSize> build_tx_frame(
  double yaw_deg, double pitch_deg, uint8_t shoot_mode, uint16_t distance)
{
  std::array<uint8_t, kVisionTxSize> frame{};
  int16_t yaw = clamp_int16(yaw_deg * 10.0);
  int16_t pitch = clamp_int16(pitch_deg * 10.0);

  frame[0] = kFrameHeader;
  frame[1] = static_cast<uint8_t>((yaw >> 8) & 0xFF);
  frame[2] = static_cast<uint8_t>(yaw & 0xFF);
  frame[3] = static_cast<uint8_t>((pitch >> 8) & 0xFF);
  frame[4] = static_cast<uint8_t>(pitch & 0xFF);
  frame[5] = shoot_mode;
  frame[6] = static_cast<uint8_t>((distance >> 8) & 0xFF);
  frame[7] = static_cast<uint8_t>(distance & 0xFF);

  uint8_t xor_check = 0;
  for (size_t i = 1; i <= 7; ++i) {
    xor_check ^= frame[i];
  }
  frame[8] = xor_check;
  frame[9] = kFrameEnd;
  return frame;
}

int16_t unpack_int16(uint8_t high, uint8_t low)
{
  return static_cast<int16_t>((static_cast<uint16_t>(high) << 8) | low);
}

bool parse_rx_frame(std::vector<uint8_t> & buffer, VisionRxFrame & out)
{
  for (size_t i = 0; i + kVisionRxSize <= buffer.size(); ++i) {
    if (buffer[i] != kFrameHeader) continue;
    if (buffer[i + kVisionRxSize - 1] != kFrameEnd) continue;

    uint8_t xor_check = 0;
    for (size_t j = 1; j <= 10; ++j) {
      xor_check ^= buffer[i + j];
    }
    uint8_t checksum = buffer[i + 11];
    if (xor_check != checksum) continue;

    out.yaw_0p1deg = unpack_int16(buffer[i + 1], buffer[i + 2]);
    out.pitch_0p1deg = unpack_int16(buffer[i + 3], buffer[i + 4]);
    out.roll_0p1deg = unpack_int16(buffer[i + 5], buffer[i + 6]);
    out.shoot_speed = unpack_int16(buffer[i + 7], buffer[i + 8]);
    out.color = buffer[i + 9];
    out.mode = buffer[i + 10];
    out.checksum = checksum;

    buffer.erase(buffer.begin(), buffer.begin() + i + kVisionRxSize);
    return true;
  }

  if (buffer.size() > 64) {
    buffer.erase(buffer.begin(), buffer.end() - (kVisionRxSize - 1));
  }
  return false;
}

void print_tui(
  const UiState & ui, const VisionRxFrame & rx, bool rx_valid, double rx_age_ms, double dt)
{
  std::fputs("\033[2J\033[H", stdout);

  double rx_yaw = rx.yaw_0p1deg / 10.0;
  double rx_pitch = rx.pitch_0p1deg / 10.0;
  double rx_roll = rx.roll_0p1deg / 10.0;

  std::printf(
    "Hero Gimbal UI Test (Vision protocol)\n"
    "dt: %.1fms  shoot_mode:%u(%s)  distance:%u  step:%.2fdeg  rx_ok:%d age:%.0fms\n"
    "CMD (deg): yaw:%+.2f  pitch:%+.2f\n"
    "RX  (deg): yaw:%+.2f  pitch:%+.2f  roll:%+.2f  shoot_speed:%d  color:%u  mode:%u\n"
    "\n"
    "Keys: q quit | w/s or Up/Down pitch +/- | a/d or Left/Right yaw -/+ | [/] step | 0 reset\n"
    "      1 off 2 ready 3 single 4 fire | f toggle fire | space single pulse | ,/. distance -/+100\n",
    dt * 1e3, ui.shoot_mode, shoot_mode_name(ui.shoot_mode), ui.distance, ui.step_deg,
    rx_valid ? 1 : 0, rx_age_ms, ui.cmd_yaw_deg, ui.cmd_pitch_deg, rx_yaw, rx_pitch, rx_roll,
    static_cast<int>(rx.shoot_speed), rx.color, rx.mode);
  std::fflush(stdout);
}

void draw_gauges(
  cv::Mat & canvas, double yaw_deg, double pitch_deg, bool rx_ok, bool shoot_ready, bool shoot_fire)
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

    double angle_rad = angle_deg * kPi / 180.0;
    cv::Point tip(
      center.x + static_cast<int>(std::cos(angle_rad) * radius * 0.85),
      center.y - static_cast<int>(std::sin(angle_rad) * radius * 0.85));
    cv::line(canvas, center, tip, color, 3, cv::LINE_AA);

    for (int a = 0; a < 360; a += 30) {
      double ar = a * kPi / 180.0;
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

  cv::circle(canvas, {80, 60}, 14, rx_ok ? ok : dim, -1, cv::LINE_AA);
  cv::circle(canvas, {120, 60}, 14, shoot_ready ? ok : dim, -1, cv::LINE_AA);
  cv::circle(canvas, {160, 60}, 14, shoot_fire ? warn : dim, -1, cv::LINE_AA);
}

}  // namespace

const std::string keys =
  "{help h usage ? |      | show command line options}"
  "{@port          |      | serial port path }"
  "{baud           | 115200 | serial baud rate }"
  "{nogui          | false| force no OpenCV window (TUI mode)}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto port = cli.get<std::string>(0);
  int baud = cli.get<int>("baud");
  bool nogui = cli.get<bool>("nogui");
  if (cli.has("help") || port.empty()) {
    cli.printMessage();
    return 0;
  }

  serial::Serial serial;
  serial.setPort(port);
  serial.setBaudrate(static_cast<uint32_t>(baud));
  auto timeout = serial::Timeout::simpleTimeout(5);
  serial.setTimeout(timeout);
  serial.open();

  tools::Exiter exiter;
  UiState ui;

  bool use_gui = !nogui;
  if (use_gui) {
    try {
      cv::namedWindow("Hero Gimbal UI Test", cv::WINDOW_NORMAL);
      cv::resizeWindow("Hero Gimbal UI Test", 980, 560);
    } catch (const cv::Exception &) {
      use_gui = false;
    }
  }

  TerminalRawMode terminal;
  terminal.enable();

  std::vector<uint8_t> rx_buffer;
  VisionRxFrame rx{};
  bool has_rx = false;
  auto last_rx = std::chrono::steady_clock::now();
  auto last_loop = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    auto now = std::chrono::steady_clock::now();
    auto dt = tools::delta_time(now, last_loop);
    last_loop = now;

    if (ui.single_pulse && now >= ui.single_pulse_until) {
      ui.single_pulse = false;
    }

    size_t avail = serial.available();
    if (avail > 0) {
      std::vector<uint8_t> chunk(avail);
      size_t got = serial.read(chunk, avail);
      rx_buffer.insert(rx_buffer.end(), chunk.begin(), chunk.begin() + got);
    }

    while (parse_rx_frame(rx_buffer, rx)) {
      has_rx = true;
      last_rx = now;
    }

    uint8_t shoot_mode = ui.shoot_mode;
    if (ui.single_pulse) shoot_mode = static_cast<uint8_t>(ShootMode::Single);
    auto frame = build_tx_frame(ui.cmd_yaw_deg, ui.cmd_pitch_deg, shoot_mode, ui.distance);
    serial.write(frame.data(), frame.size());

    double rx_age_ms = has_rx ? tools::delta_time(now, last_rx) * 1e3 : -1.0;
    print_tui(ui, rx, has_rx, rx_age_ms, dt);

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
      double rx_yaw = has_rx ? (rx.yaw_0p1deg / 10.0) : 0.0;
      double rx_pitch = has_rx ? (rx.pitch_0p1deg / 10.0) : 0.0;
      bool rx_ok = has_rx && (tools::delta_time(now, last_rx) < 0.3);
      bool ready_on = (shoot_mode == static_cast<uint8_t>(ShootMode::Ready)) ||
                      (shoot_mode == static_cast<uint8_t>(ShootMode::Single)) ||
                      (shoot_mode == static_cast<uint8_t>(ShootMode::Fire));
      bool fire_on = shoot_mode == static_cast<uint8_t>(ShootMode::Fire);
      draw_gauges(canvas, rx_yaw, rx_pitch, rx_ok, ready_on, fire_on);
      cv::imshow("Hero Gimbal UI Test", canvas);
      int gui_key = cv::waitKey(1);
      if (gui_key != -1) key = gui_key;
    }

    if (key == 'q') break;
    if (key == '1') ui.shoot_mode = static_cast<uint8_t>(ShootMode::Off);
    if (key == '2') ui.shoot_mode = static_cast<uint8_t>(ShootMode::Ready);
    if (key == '3') ui.shoot_mode = static_cast<uint8_t>(ShootMode::Single);
    if (key == '4') ui.shoot_mode = static_cast<uint8_t>(ShootMode::Fire);
    if (key == 'f') {
      ui.shoot_mode =
        (ui.shoot_mode == static_cast<uint8_t>(ShootMode::Fire)) ?
        static_cast<uint8_t>(ShootMode::Off) :
        static_cast<uint8_t>(ShootMode::Fire);
    }
    if (key == ' ') {
      ui.single_pulse = true;
      ui.single_pulse_until = now + 120ms;
    }
    if (key == '0') {
      ui.cmd_yaw_deg = 0.0;
      ui.cmd_pitch_deg = 0.0;
    }

    if (key == '[') ui.step_deg = std::max(0.01, ui.step_deg - 0.05);
    if (key == ']') ui.step_deg = std::min(10.0, ui.step_deg + 0.05);

    if (key == ',') {
      ui.distance = static_cast<uint16_t>(std::max(0, static_cast<int>(ui.distance) - 100));
    }
    if (key == '.') {
      ui.distance = static_cast<uint16_t>(std::min(65535, static_cast<int>(ui.distance) + 100));
    }

    double step_deg = ui.step_deg;
    if (key == 'a' || key == 81) ui.cmd_yaw_deg -= step_deg;
    if (key == 'd' || key == 83) ui.cmd_yaw_deg += step_deg;
    if (key == 'w' || key == 82) ui.cmd_pitch_deg += step_deg;
    if (key == 's' || key == 84) ui.cmd_pitch_deg -= step_deg;

    std::this_thread::sleep_for(5ms);
  }

  auto stop = build_tx_frame(0.0, 0.0, static_cast<uint8_t>(ShootMode::Off), 0);
  serial.write(stop.data(), stop.size());
  return 0;
}
