#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "serial/serial.h"
#include "tools/crc.hpp"
#include "tools/exiter.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace
{
constexpr uint8_t kRxHeader = 0x5A;
constexpr size_t kExtendedFrameSize = 49;
constexpr double kRad2Deg = 57.29577951308232;

struct __attribute__((packed)) VisionToGimbal
{
  uint8_t header = 0xA5;
  uint8_t tracking = 0;
  float pitch = 0.0f;
  float yaw = 0.0f;
  uint8_t fire = 0;
  uint8_t fric_on = 0;
  uint16_t checksum = 0;
};

static_assert(sizeof(VisionToGimbal) == 14, "VisionToGimbal packet size mismatch.");

struct FrameSnapshot
{
  bool valid = false;
  uint8_t detect_color = 0;
  uint8_t reset_tracker = 0;
  uint8_t robot_id = 0;
  float yaw = 0.0f;
  float pitch = 0.0f;
  float roll = 0.0f;
  float yaw_odom = 0.0f;
  float pitch_odom = 0.0f;
  float yaw_vel = 0.0f;
  float pitch_vel = 0.0f;
  std::chrono::steady_clock::time_point t{};
};

struct DiagStats
{
  uint64_t tx_ok = 0;
  uint64_t tx_exc = 0;
  uint64_t read_calls = 0;
  uint64_t read_zero = 0;
  uint64_t read_exception = 0;
  uint64_t bytes = 0;
  uint64_t drop_bytes = 0;
  uint64_t headers_seen = 0;
  uint64_t extended_ok = 0;
  uint64_t extended_crc_fail = 0;
};

std::string trim_copy(const std::string & s)
{
  size_t l = 0;
  while (l < s.size() && std::isspace(static_cast<unsigned char>(s[l]))) ++l;
  size_t r = s.size();
  while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) --r;
  return s.substr(l, r - l);
}

std::vector<std::string> split_csv(const std::string & csv)
{
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= csv.size()) {
    size_t comma = csv.find(',', start);
    auto token =
      trim_copy(csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
    if (!token.empty()) out.push_back(token);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return out;
}

float unpack_float(const uint8_t * p)
{
  float value = 0.0f;
  std::memcpy(&value, p, sizeof(float));
  return value;
}

void parse_extended_frame(const uint8_t * frame, FrameSnapshot & out)
{
  out.valid = true;
  const uint8_t flags = frame[1];
  out.detect_color = flags & 0x01;
  out.reset_tracker = (flags >> 1) & 0x01;
  out.yaw = unpack_float(frame + 2);
  out.pitch = -unpack_float(frame + 6);
  out.roll = unpack_float(frame + 10);
  out.yaw_odom = unpack_float(frame + 14);
  out.pitch_odom = -unpack_float(frame + 18);
  out.yaw_vel = unpack_float(frame + 22);
  out.pitch_vel = -unpack_float(frame + 26);
  out.robot_id = frame[46];
  out.t = std::chrono::steady_clock::now();
}

void parse_stream(std::vector<uint8_t> & buffer, DiagStats & stats, FrameSnapshot & frame)
{
  while (true) {
    auto it = std::find(buffer.begin(), buffer.end(), kRxHeader);
    if (it == buffer.end()) {
      if (buffer.size() > kExtendedFrameSize - 1) {
        const size_t drop = buffer.size() - (kExtendedFrameSize - 1);
        stats.drop_bytes += drop;
        buffer.erase(buffer.begin(), buffer.begin() + static_cast<long>(drop));
      }
      return;
    }

    const size_t idx = static_cast<size_t>(it - buffer.begin());
    if (idx > 0) {
      stats.drop_bytes += idx;
      buffer.erase(buffer.begin(), it);
    }

    if (buffer.size() < kExtendedFrameSize) return;
    stats.headers_seen++;

    const bool extended_ok = tools::check_crc16(buffer.data(), kExtendedFrameSize);
    if (extended_ok) {
      stats.extended_ok++;
      parse_extended_frame(buffer.data(), frame);
      buffer.erase(buffer.begin(), buffer.begin() + static_cast<long>(kExtendedFrameSize));
      continue;
    }

    stats.extended_crc_fail++;
    buffer.erase(buffer.begin());
  }
}

std::vector<std::string> build_ports(const std::string & config_path, const std::string & ports_arg)
{
  std::vector<std::string> ports;
  if (!ports_arg.empty()) ports = split_csv(ports_arg);

  if (ports.empty()) {
    try {
      auto yaml = tools::load(config_path);
      auto from_config = tools::read<std::string>(yaml, "com_port");
      if (!from_config.empty()) ports.push_back(from_config);
    } catch (...) {
    }
  }

  if (ports.empty()) {
    ports = {"/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyS0"};
  }
  return ports;
}

bool open_first_available(
  serial::Serial & serial, const std::vector<std::string> & ports, int baud, std::string & opened_port,
  std::string & fail_reason)
{
  for (const auto & port : ports) {
    try {
      serial.setPort(port);
      serial.setBaudrate(static_cast<uint32_t>(baud));
      serial::Timeout timeout = serial::Timeout::simpleTimeout(20);
      serial.setTimeout(timeout);
      serial.open();
      opened_port = port;
      fail_reason.clear();
      return true;
    } catch (const std::exception & e) {
      fail_reason = e.what();
      std::fprintf(stderr, "[diag] open fail port=%s reason=%s\n", port.c_str(), e.what());
      try {
        serial.close();
      } catch (...) {
      }
    }
  }
  return false;
}

double age_ms(std::chrono::steady_clock::time_point t)
{
  if (t.time_since_epoch().count() == 0) return -1.0;
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

const std::string kKeys =
  "{help h usage ? |      | 输出命令行参数说明 }"
  "{@config-path   |      | 位置参数，yaml配置文件路径 }"
  "{ports          |      | 串口列表(逗号分隔)，为空则读config+默认列表 }"
  "{baud           | 115200 | 串口波特率 }"
  "{duration-ms    | 3000 | 运行时长(ms) }"
  "{summary-ms     | 1000 | 统计打印周期(ms) }"
  "{loop-ms        | 2 | 主循环sleep时长(ms) }"
  "{no-send        | false | 仅收包，不发控制包 }"
  "{tracking       | 1 | 下发tracking: 0/1 }"
  "{fric-on        | 1 | 下发fric_on: 0/1 }"
  "{fire-mode      | 0 | 下发fire: 0(off)/1(ready)/2(single)/3(fire) }"
  "{yaw-deg        | 0 | 下发yaw(度) }"
  "{pitch-deg      | 0 | 下发pitch(度) }"
  "{require-rx     | false | 若结束时仍无有效回传则返回非0 }";
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, kKeys);
  const auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  const int baud = std::max(1, cli.get<int>("baud"));
  const int duration_ms = std::max(1, cli.get<int>("duration-ms"));
  const int summary_ms = std::max(100, cli.get<int>("summary-ms"));
  const int loop_ms = std::max(1, cli.get<int>("loop-ms"));
  const bool no_send = cli.get<bool>("no-send");
  const bool require_rx = cli.get<bool>("require-rx");

  auto ports = build_ports(config_path, cli.get<std::string>("ports"));
  if (ports.empty()) {
    std::fprintf(stderr, "no serial ports available.\n");
    return 1;
  }

  VisionToGimbal tx{};
  tx.tracking = cli.get<int>("tracking") != 0 ? 1 : 0;
  tx.fric_on = cli.get<int>("fric-on") != 0 ? 1 : 0;
  tx.fire = static_cast<uint8_t>(std::clamp(cli.get<int>("fire-mode"), 0, 3));
  const auto cmd_yaw_deg = cli.get<double>("yaw-deg");
  const auto cmd_pitch_deg = cli.get<double>("pitch-deg");
  tx.yaw = static_cast<float>(cmd_yaw_deg / 57.3);
  // Protocol side uses pitch up negative, while CLI uses pitch up positive.
  tx.pitch = static_cast<float>(-cmd_pitch_deg / 57.3);
  tx.checksum = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx), sizeof(tx) - sizeof(tx.checksum));

  std::printf(
    "gimbal_link_diag_test: baud=%d duration=%dms summary=%dms loop=%dms send=%s cmd(track=%u fric=%u fire=%u yaw=%.3f pitch=%.3f) ports=",
    baud, duration_ms, summary_ms, loop_ms, no_send ? "off" : "on",
    static_cast<unsigned>(tx.tracking), static_cast<unsigned>(tx.fric_on),
    static_cast<unsigned>(tx.fire), cmd_yaw_deg / 57.3, cmd_pitch_deg / 57.3);
  for (size_t i = 0; i < ports.size(); ++i) {
    std::printf("%s%s", i ? "," : "", ports[i].c_str());
  }
  std::printf("\n");
  std::fflush(stdout);

  serial::Serial serial;
  std::string opened_port;
  std::string fail_reason;
  if (!open_first_available(serial, ports, baud, opened_port, fail_reason)) {
    std::fprintf(stderr, "failed to open serial ports: %s\n", fail_reason.c_str());
    return 1;
  }
  std::printf("[diag] opened %s\n", opened_port.c_str());
  std::fflush(stdout);

  tools::Exiter exiter;
  DiagStats stats{};
  DiagStats last_stats{};
  FrameSnapshot frame{};
  std::vector<uint8_t> buffer;
  buffer.reserve(2048);

  auto start_t = std::chrono::steady_clock::now();
  auto last_summary_t = start_t;

  while (!exiter.exit()) {
    auto now = std::chrono::steady_clock::now();
    if (now - start_t >= std::chrono::milliseconds(duration_ms)) break;

    if (!no_send) {
      try {
        serial.write(reinterpret_cast<const uint8_t *>(&tx), sizeof(tx));
        stats.tx_ok++;
      } catch (...) {
        stats.tx_exc++;
      }
    }

    try {
      size_t avail = serial.available();
      stats.read_calls++;
      if (avail == 0) {
        stats.read_zero++;
      } else {
        size_t to_read = std::min<size_t>(avail, 4096);
        std::vector<uint8_t> chunk;
        chunk.reserve(to_read);
        size_t got = serial.read(chunk, to_read);
        if (got > 0) {
          stats.bytes += got;
          buffer.insert(buffer.end(), chunk.begin(), chunk.end());
          parse_stream(buffer, stats, frame);
        } else {
          stats.read_zero++;
        }
      }
    } catch (...) {
      stats.read_exception++;
    }

    now = std::chrono::steady_clock::now();
    if (now - last_summary_t >= std::chrono::milliseconds(summary_ms)) {
      const auto d_tx = stats.tx_ok - last_stats.tx_ok;
      const auto d_bytes = stats.bytes - last_stats.bytes;
      const auto d_ok49 = stats.extended_ok - last_stats.extended_ok;
      const auto d_crc49 = stats.extended_crc_fail - last_stats.extended_crc_fail;

      std::printf(
        "[diag][%dms] port=%s tx=%llu(+%llu) bytes=%llu(+%llu) ok49=%llu(+%llu) crc49=%llu(+%llu) age=%.0fms\n",
        summary_ms, opened_port.c_str(), static_cast<unsigned long long>(stats.tx_ok),
        static_cast<unsigned long long>(d_tx), static_cast<unsigned long long>(stats.bytes),
        static_cast<unsigned long long>(d_bytes), static_cast<unsigned long long>(stats.extended_ok),
        static_cast<unsigned long long>(d_ok49),
        static_cast<unsigned long long>(stats.extended_crc_fail),
        static_cast<unsigned long long>(d_crc49), age_ms(frame.t));

      if (frame.valid) {
        std::printf(
          "[frame] proto=%s yaw=%.2fdeg pitch=%.2fdeg roll=%.2fdeg yaw_odom=%.3f pitch_odom=%.3f yaw_vel=%.3f pitch_vel=%.3f color=%u reset=%u robot_id=%u\n",
          "49B", frame.yaw * kRad2Deg, frame.pitch * kRad2Deg,
          frame.roll * kRad2Deg, frame.yaw_odom, frame.pitch_odom, frame.yaw_vel, frame.pitch_vel,
          frame.detect_color, frame.reset_tracker, frame.robot_id);
      }
      std::fflush(stdout);

      last_stats = stats;
      last_summary_t = now;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(loop_ms));
  }

  try {
    if (serial.isOpen()) serial.close();
  } catch (...) {
  }

  std::printf(
    "done: tx_ok=%llu tx_exc=%llu bytes=%llu ok49=%llu crc49=%llu read_exc=%llu\n",
    static_cast<unsigned long long>(stats.tx_ok), static_cast<unsigned long long>(stats.tx_exc),
    static_cast<unsigned long long>(stats.bytes),
    static_cast<unsigned long long>(stats.extended_ok),
    static_cast<unsigned long long>(stats.extended_crc_fail),
    static_cast<unsigned long long>(stats.read_exception));
  std::fflush(stdout);

  if (require_rx && stats.extended_ok == 0) {
    std::fprintf(stderr, "require-rx enabled but no valid rx frame observed.\n");
    return 2;
  }
  return 0;
}
