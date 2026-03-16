#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "serial/serial.h"
#include "tools/crc.hpp"
#include "tools/exiter.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace
{
constexpr uint8_t kHeader = 0x5A;
constexpr size_t kExtendedFrameSize = 49;
constexpr double kRad2Deg = 57.29577951308232;

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

struct ProbeStats
{
  uint64_t open_ok = 0;
  uint64_t open_fail = 0;
  uint64_t read_calls = 0;
  uint64_t read_zero = 0;
  uint64_t read_exception = 0;
  uint64_t bytes = 0;
  uint64_t drop_bytes = 0;
  uint64_t headers_seen = 0;
  uint64_t extended_ok = 0;
  uint64_t extended_crc_fail = 0;

  bool has_fail_sample = false;
  uint16_t last_extended_crc_rx = 0;
  uint16_t last_extended_crc_calc = 0;
  std::string last_fail_prefix;
};

float unpack_float(const uint8_t * p)
{
  float value = 0.0f;
  std::memcpy(&value, p, sizeof(float));
  return value;
}

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
    auto token = trim_copy(csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
    if (!token.empty()) out.push_back(token);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return out;
}

std::string hex_prefix(const uint8_t * data, size_t len, size_t max_len)
{
  std::string out;
  char tmp[4] = {};
  const size_t n = std::min(len, max_len);
  out.reserve(n * 3 + 4);
  for (size_t i = 0; i < n; ++i) {
    std::snprintf(tmp, sizeof(tmp), "%02X", data[i]);
    out += tmp;
    if (i + 1 < n) out.push_back(' ');
  }
  if (len > n) out += " ...";
  return out;
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
      try {
        serial.close();
      } catch (...) {
      }
    }
  }
  return false;
}

void parse_extended_frame(const uint8_t * frame, FrameSnapshot & out)
{
  out.valid = true;
  const uint8_t flags = frame[1];
  out.detect_color = flags & 0x01;
  out.reset_tracker = (flags >> 1) & 0x01;
  out.yaw = unpack_float(frame + 2);
  out.pitch = unpack_float(frame + 6);
  out.roll = unpack_float(frame + 10);
  out.yaw_odom = unpack_float(frame + 14);
  out.pitch_odom = unpack_float(frame + 18);
  out.yaw_vel = unpack_float(frame + 22);
  out.pitch_vel = unpack_float(frame + 26);
  out.robot_id = frame[46];
  out.t = std::chrono::steady_clock::now();
}

void parse_stream(
  std::vector<uint8_t> & buffer, ProbeStats & stats, FrameSnapshot & frame, size_t fail_hex_len)
{
  while (true) {
    auto it = std::find(buffer.begin(), buffer.end(), kHeader);
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
    stats.has_fail_sample = true;
    stats.last_extended_crc_calc = tools::get_crc16(buffer.data(), kExtendedFrameSize - 2);
    stats.last_extended_crc_rx = static_cast<uint16_t>(
      buffer[kExtendedFrameSize - 2] | (static_cast<uint16_t>(buffer[kExtendedFrameSize - 1]) << 8));
    stats.last_fail_prefix = hex_prefix(buffer.data(), kExtendedFrameSize, fail_hex_len);
    buffer.erase(buffer.begin());
  }
}

double age_ms(std::chrono::steady_clock::time_point t)
{
  if (t.time_since_epoch().count() == 0) return -1.0;
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

std::vector<std::string> build_ports(const std::string & config_path, const std::string & ports_arg)
{
  std::vector<std::string> ports;
  if (!ports_arg.empty()) ports = split_csv(ports_arg);

  if (ports.empty() && !config_path.empty()) {
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

}  // namespace

const std::string keys =
  "{help h usage ? |      | show command line options}"
  "{@config-path   |      | yaml config path (optional) }"
  "{ports          |      | comma separated ports, e.g. /dev/ttyACM0,/dev/ttyUSB0 }"
  "{baud           | 115200 | serial baud rate }"
  "{duration-ms    | 0 | run duration in ms, 0 means infinite }"
  "{summary-ms     | 1000 | summary print period in ms }"
  "{sleep-ms       | 2 | loop sleep in ms }"
  "{read-max       | 256 | max bytes to read per loop }"
  "{hex-len        | 24 | hex prefix length for raw/fail logs }"
  "{raw-log        | false | print each received chunk in hex }"
  "{reopen-ms      | 1000 | retry-open interval when disconnected }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  auto ports_arg = cli.get<std::string>("ports");
  int baud = std::max(1, cli.get<int>("baud"));
  int duration_ms = std::max(0, cli.get<int>("duration-ms"));
  int summary_ms = std::max(50, cli.get<int>("summary-ms"));
  int sleep_ms = std::max(0, cli.get<int>("sleep-ms"));
  int read_max = std::max(1, cli.get<int>("read-max"));
  int hex_len = std::max(4, cli.get<int>("hex-len"));
  int reopen_ms = std::max(10, cli.get<int>("reopen-ms"));
  bool raw_log = cli.get<bool>("raw-log");

  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto ports = build_ports(config_path, ports_arg);
  std::printf("gimbal_serial_probe: baud=%d ports=", baud);
  for (size_t i = 0; i < ports.size(); ++i) {
    std::printf("%s%s", ports[i].c_str(), (i + 1 < ports.size()) ? "," : "");
  }
  std::printf("\n");

  tools::Exiter exiter;
  serial::Serial serial;
  std::string opened_port;
  std::string fail_reason;
  ProbeStats stats{};
  ProbeStats last_stats{};
  std::vector<uint8_t> stream_buffer;
  FrameSnapshot frame{};

  auto t_start = std::chrono::steady_clock::now();
  auto t_last_summary = t_start;
  auto t_last_open_try = t_start - std::chrono::seconds(10);
  auto t_last_data = std::chrono::steady_clock::time_point{};
  std::string last_chunk_hex;

  while (!exiter.exit()) {
    auto now = std::chrono::steady_clock::now();
    if (duration_ms > 0) {
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count();
      if (elapsed_ms >= duration_ms) break;
    }

    if (!serial.isOpen()) {
      auto retry_age = std::chrono::duration_cast<std::chrono::milliseconds>(now - t_last_open_try).count();
      if (retry_age >= reopen_ms) {
        t_last_open_try = now;
        if (open_first_available(serial, ports, baud, opened_port, fail_reason)) {
          stats.open_ok++;
          std::printf("[probe] opened %s\n", opened_port.c_str());
        } else {
          stats.open_fail++;
          std::printf("[probe] open failed (%s)\n", fail_reason.c_str());
        }
      }
    } else {
      try {
        size_t avail = serial.available();
        size_t to_read = std::min(static_cast<size_t>(read_max), std::max<size_t>(1, avail));
        std::vector<uint8_t> chunk;
        chunk.reserve(to_read);
        size_t got = serial.read(chunk, to_read);
        stats.read_calls++;

        if (got == 0) {
          stats.read_zero++;
        } else {
          stats.bytes += got;
          t_last_data = now;
          stream_buffer.insert(stream_buffer.end(), chunk.begin(), chunk.end());
          last_chunk_hex = hex_prefix(chunk.data(), chunk.size(), static_cast<size_t>(hex_len));
          if (raw_log) {
            std::printf("[raw] n=%zu hex=%s\n", got, last_chunk_hex.c_str());
          }
        }
        parse_stream(stream_buffer, stats, frame, static_cast<size_t>(hex_len));
      } catch (const std::exception & e) {
        stats.read_exception++;
        std::printf("[probe] read exception on %s: %s\n", opened_port.c_str(), e.what());
        try {
          serial.close();
        } catch (...) {
        }
      }
    }

    now = std::chrono::steady_clock::now();
    auto summary_age = std::chrono::duration_cast<std::chrono::milliseconds>(now - t_last_summary).count();
    if (summary_age >= summary_ms) {
      auto d_bytes = stats.bytes - last_stats.bytes;
      auto d_reads = stats.read_calls - last_stats.read_calls;
      auto d_zero = stats.read_zero - last_stats.read_zero;
      auto d_drop = stats.drop_bytes - last_stats.drop_bytes;
      auto d_headers = stats.headers_seen - last_stats.headers_seen;
      auto d_ok49 = stats.extended_ok - last_stats.extended_ok;
      auto d_crc49 = stats.extended_crc_fail - last_stats.extended_crc_fail;

      std::printf(
        "[probe][%dms] port=%s bytes=%llu reads=%llu zero=%llu drop=%llu hdr=%llu ok49=%llu crc49=%llu frame_age=%.0fms data_age=%.0fms\n",
        summary_ms, serial.isOpen() ? opened_port.c_str() : "<closed>",
        static_cast<unsigned long long>(d_bytes), static_cast<unsigned long long>(d_reads),
        static_cast<unsigned long long>(d_zero), static_cast<unsigned long long>(d_drop),
        static_cast<unsigned long long>(d_headers), static_cast<unsigned long long>(d_ok49),
        static_cast<unsigned long long>(d_crc49), age_ms(frame.t), age_ms(t_last_data));

      if (frame.valid) {
        std::printf(
          "[frame] proto=%s yaw=%.2fdeg pitch=%.2fdeg roll=%.2fdeg yaw_odom=%.3f pitch_odom=%.3f yaw_vel=%.3f pitch_vel=%.3f color=%u reset=%u robot_id=%u\n",
          "49B", frame.yaw * kRad2Deg, frame.pitch * kRad2Deg,
          frame.roll * kRad2Deg, frame.yaw_odom, frame.pitch_odom, frame.yaw_vel, frame.pitch_vel,
          frame.detect_color, frame.reset_tracker, frame.robot_id);
      } else if (!last_chunk_hex.empty()) {
        std::printf("[frame] no valid frame yet, last_chunk=%s\n", last_chunk_hex.c_str());
      }

      if (stats.has_fail_sample) {
        std::printf(
          "[fail] crc49(rx/calc)=0x%04X/0x%04X prefix=%s\n", stats.last_extended_crc_rx,
          stats.last_extended_crc_calc, stats.last_fail_prefix.c_str());
      }

      last_stats = stats;
      t_last_summary = now;
      std::fflush(stdout);
    }

    if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  }

  if (serial.isOpen()) {
    try {
      serial.close();
    } catch (...) {
    }
  }

  std::printf(
    "done total: open_ok=%llu open_fail=%llu bytes=%llu reads=%llu zero=%llu drop=%llu ok49=%llu crc49=%llu read_exc=%llu\n",
    static_cast<unsigned long long>(stats.open_ok), static_cast<unsigned long long>(stats.open_fail),
    static_cast<unsigned long long>(stats.bytes), static_cast<unsigned long long>(stats.read_calls),
    static_cast<unsigned long long>(stats.read_zero), static_cast<unsigned long long>(stats.drop_bytes),
    static_cast<unsigned long long>(stats.extended_ok),
    static_cast<unsigned long long>(stats.extended_crc_fail),
    static_cast<unsigned long long>(stats.read_exception));
  return 0;
}
