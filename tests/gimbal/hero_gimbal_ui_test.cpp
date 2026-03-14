#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <thread>
#include <unistd.h>
#include <termios.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>

#include "serial/serial.h"
#include "tools/exiter.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono_literals;

namespace
{
constexpr uint8_t kUsbTxHeader = 0xA5;
constexpr uint8_t kUsbRxHeader = 0x5A;
constexpr size_t kUsbTxSize = 48;
constexpr size_t kUsbRxSize = 28;
constexpr double kPi = 3.14159265358979323846;
constexpr uint16_t kCrc16Init = 0xFFFF;
constexpr uint16_t kCrc16Poly = 0xA001;

struct UiState
{
  double cmd_yaw_deg = 0.0;
  double cmd_pitch_deg = 0.0;
  double step_deg = 5.0;
  float distance_m = 3.0f;
  bool tracking = true;
  uint8_t armor_id = 1;
  uint8_t armors_num = 4;
};

struct UsbRxFrame
{
  uint8_t detect_color = 0;
  uint8_t reset_tracker = 0;
  float roll = 0.0f;
  float pitch = 0.0f;
  float yaw = 0.0f;
  float aim_x = 0.0f;
  float aim_y = 0.0f;
  float aim_z = 0.0f;
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

uint16_t crc16_ibm(const uint8_t * data, size_t length)
{
  uint16_t crc = kCrc16Init;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      if (crc & 1) {
        crc = static_cast<uint16_t>((crc >> 1) ^ kCrc16Poly);
      } else {
        crc >>= 1;
      }
    }
  }
  return crc;
}

void pack_float(std::array<uint8_t, kUsbTxSize> & buffer, size_t & index, float value)
{
  std::memcpy(buffer.data() + index, &value, sizeof(float));
  index += sizeof(float);
}

float unpack_float(const uint8_t * data)
{
  float value = 0.0f;
  std::memcpy(&value, data, sizeof(float));
  return value;
}

std::array<uint8_t, kUsbTxSize> build_usb_tx_frame(const UiState & ui)
{
  std::array<uint8_t, kUsbTxSize> frame{};
  size_t index = 0;
  frame[index++] = kUsbTxHeader;

  uint8_t flags = 0;
  flags |= ui.tracking ? 0x01 : 0x00;
  flags |= static_cast<uint8_t>((ui.armor_id & 0x07) << 1);
  flags |= static_cast<uint8_t>((ui.armors_num & 0x07) << 4);
  frame[index++] = flags;

  double yaw_rad = ui.cmd_yaw_deg * kPi / 180.0;
  double pitch_rad = ui.cmd_pitch_deg * kPi / 180.0;
  float horizontal = ui.distance_m * static_cast<float>(std::cos(pitch_rad));
  float x = -horizontal * static_cast<float>(std::cos(yaw_rad));
  float y = -horizontal * static_cast<float>(std::sin(yaw_rad));
  float z = ui.distance_m * static_cast<float>(std::sin(pitch_rad));

  pack_float(frame, index, x);
  pack_float(frame, index, y);
  pack_float(frame, index, z);
  pack_float(frame, index, static_cast<float>(yaw_rad));
  pack_float(frame, index, 0.0f);
  pack_float(frame, index, 0.0f);
  pack_float(frame, index, 0.0f);
  pack_float(frame, index, 0.0f);
  pack_float(frame, index, 0.2f);
  pack_float(frame, index, 0.2f);
  pack_float(frame, index, 0.0f);

  uint16_t crc = crc16_ibm(frame.data(), kUsbTxSize - 2);
  frame[index++] = static_cast<uint8_t>(crc & 0xFF);
  frame[index++] = static_cast<uint8_t>((crc >> 8) & 0xFF);
  return frame;
}

bool parse_usb_rx_frame(
  std::vector<uint8_t> & buffer, UsbRxFrame & out, bool & checksum_ok)
{
  for (size_t i = 0; i + kUsbRxSize <= buffer.size(); ++i) {
    if (buffer[i] != kUsbRxHeader) continue;

    uint16_t crc_expected = crc16_ibm(buffer.data() + i, kUsbRxSize - 2);
    uint16_t crc_got = static_cast<uint16_t>(buffer[i + kUsbRxSize - 2]) |
                       static_cast<uint16_t>(buffer[i + kUsbRxSize - 1] << 8);
    checksum_ok = (crc_expected == crc_got);

    uint8_t flags = buffer[i + 1];
    out.detect_color = flags & 0x01;
    out.reset_tracker = (flags >> 1) & 0x01;
    out.roll = unpack_float(buffer.data() + i + 2);
    out.pitch = unpack_float(buffer.data() + i + 6);
    out.yaw = unpack_float(buffer.data() + i + 10);
    out.aim_x = unpack_float(buffer.data() + i + 14);
    out.aim_y = unpack_float(buffer.data() + i + 18);
    out.aim_z = unpack_float(buffer.data() + i + 22);

    buffer.erase(buffer.begin(), buffer.begin() + i + kUsbRxSize);
    return true;
  }

  if (buffer.size() > 64) {
    buffer.erase(buffer.begin(), buffer.end() - (kUsbRxSize - 1));
  }
  return false;
}

void print_tui(
  const UiState & ui, const UsbRxFrame & rx, bool rx_valid, bool checksum_ok, double rx_age_ms,
  double dt)
{
  std::fputs("\033[2J\033[H", stdout);

  double rx_yaw = rx.yaw * 180.0 / kPi;
  double rx_pitch = rx.pitch * 180.0 / kPi;
  double rx_roll = rx.roll * 180.0 / kPi;

  std::printf(
    "Hero Gimbal UI Test (USB auto-aim protocol)\n"
    "dt: %.1fms  tracking:%d  dist:%.2fm  step:%.2fdeg  rx_ok:%d crc:%d age:%.0fms\n"
    "CMD (deg): yaw:%+.2f  pitch:%+.2f\n"
    "RX  (deg): yaw:%+.2f  pitch:%+.2f  roll:%+.2f  color:%u reset:%u\n"
    "RX  (m):   aim_x:%+.2f  aim_y:%+.2f  aim_z:%+.2f\n"
    "\n"
    "Keys: q quit | w/s or Up/Down pitch +/- | a/d or Left/Right yaw -/+ | [/] step | 0 reset\n"
    "      c tracking | ,/. dist -/+0.1m\n",
    dt * 1e3, ui.tracking ? 1 : 0, ui.distance_m, ui.step_deg, rx_valid ? 1 : 0,
    checksum_ok ? 1 : 0, rx_age_ms, ui.cmd_yaw_deg, ui.cmd_pitch_deg, rx_yaw, rx_pitch, rx_roll,
    rx.detect_color, rx.reset_tracker, rx.aim_x, rx.aim_y, rx.aim_z);
  std::fflush(stdout);
}

void draw_gauges(
  cv::Mat & canvas, double yaw_deg, double pitch_deg, bool rx_ok, bool tracking)
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
  cv::circle(canvas, {120, 60}, 14, tracking ? ok : dim, -1, cv::LINE_AA);
  cv::circle(canvas, {160, 60}, 14, tracking ? warn : dim, -1, cv::LINE_AA);
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
  UsbRxFrame rx{};
  bool has_rx = false;
  bool checksum_ok = false;
  auto last_rx = std::chrono::steady_clock::now();
  auto last_loop = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    auto now = std::chrono::steady_clock::now();
    auto dt = tools::delta_time(now, last_loop);
    last_loop = now;

    size_t avail = serial.available();
    if (avail > 0) {
      std::vector<uint8_t> chunk(avail);
      size_t got = serial.read(chunk, avail);
      rx_buffer.insert(rx_buffer.end(), chunk.begin(), chunk.begin() + got);
    }

    while (parse_usb_rx_frame(rx_buffer, rx, checksum_ok)) {
      has_rx = true;
      last_rx = now;
    }

    auto frame = build_usb_tx_frame(ui);
    serial.write(frame.data(), frame.size());

    double rx_age_ms = has_rx ? tools::delta_time(now, last_rx) * 1e3 : -1.0;
    print_tui(ui, rx, has_rx, checksum_ok, rx_age_ms, dt);

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
      double rx_yaw = has_rx ? (rx.yaw * 180.0 / kPi) : 0.0;
      double rx_pitch = has_rx ? (rx.pitch * 180.0 / kPi) : 0.0;
      bool rx_ok = has_rx && (tools::delta_time(now, last_rx) < 0.3);
      draw_gauges(canvas, rx_yaw, rx_pitch, rx_ok, ui.tracking);
      cv::imshow("Hero Gimbal UI Test", canvas);
      int gui_key = cv::waitKey(1);
      if (gui_key != -1) key = gui_key;
    }

    if (key == 'q') break;
    if (key == 'c') ui.tracking = !ui.tracking;
    if (key == '0') {
      ui.cmd_yaw_deg = 0.0;
      ui.cmd_pitch_deg = 0.0;
    }

    if (key == '[') ui.step_deg = std::max(0.01, ui.step_deg - 0.05);
    if (key == ']') ui.step_deg = std::min(10.0, ui.step_deg + 0.05);

    if (key == ',') {
      ui.distance_m = std::max(0.1f, ui.distance_m - 0.1f);
    }
    if (key == '.') {
      ui.distance_m = std::min(50.0f, ui.distance_m + 0.1f);
    }

    double step_deg = ui.step_deg;
    if (key == 'a' || key == 81) ui.cmd_yaw_deg -= step_deg;
    if (key == 'd' || key == 83) ui.cmd_yaw_deg += step_deg;
    if (key == 'w' || key == 82) ui.cmd_pitch_deg += step_deg;
    if (key == 's' || key == 84) ui.cmd_pitch_deg -= step_deg;

    std::this_thread::sleep_for(5ms);
  }

  UiState stop_state{};
  stop_state.tracking = false;
  stop_state.distance_m = 0.0f;
  auto stop = build_usb_tx_frame(stop_state);
  serial.write(stop.data(), stop.size());
  return 0;
}
