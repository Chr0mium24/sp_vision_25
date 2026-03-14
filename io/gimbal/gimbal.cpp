#include "gimbal.hpp"

#include <array>
#include <cstring>
#include <sstream>

#include <fmt/core.h>

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
namespace
{
constexpr size_t kLegacyFrameSize = 28;
constexpr size_t kExtendedFrameSize = sizeof(GimbalToVision);

static_assert(kExtendedFrameSize == 49, "Unexpected GimbalToVision size.");

float unpack_float(const uint8_t * p)
{
  float value = 0.0f;
  std::memcpy(&value, p, sizeof(float));
  return value;
}

std::string hex_prefix(const uint8_t * data, size_t len, size_t max_len = 16)
{
  std::ostringstream oss;
  size_t n = std::min(len, max_len);
  for (size_t i = 0; i < n; ++i) {
    oss << fmt::format("{:02X}", data[i]);
    if (i + 1 < n) oss << ' ';
  }
  if (len > n) oss << " ...";
  return oss.str();
}
}  // namespace

Gimbal::Gimbal(const std::string & config_path, bool wait_for_first_q)
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");

  try {
    serial_.setPort(com_port);
    serial_.setBaudrate(115200);
    serial::Timeout timeout = serial::Timeout::simpleTimeout(100);
    serial_.setTimeout(timeout);
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  if (wait_for_first_q) {
    queue_.pop();
    tools::logger()->info("[Gimbal] First q received.");
  } else {
    tools::logger()->warn("[Gimbal] Skip waiting first q (debug mode).");
  }
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

GimbalRxStats Gimbal::rx_stats() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return rx_stats_;
}

bool Gimbal::has_valid_q() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return rx_stats_.good_frames > 0;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    if (t < t_a) return q_c;
    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  tx_data_.tracking = VisionToGimbal.tracking;
  tx_data_.pitch = VisionToGimbal.pitch;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.fire = VisionToGimbal.fire;
  tx_data_.fric_on = VisionToGimbal.fric_on;
  tx_data_.checksum = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.checksum));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    std::string hex;
    uint8_t * p = reinterpret_cast<uint8_t *>(&tx_data_);
    for (size_t i = 0; i < sizeof(tx_data_); ++i) {
      hex += fmt::format("{:02X} ", p[i]);
    }
    // tools::logger()->info("[Gimbal] TX: {}", hex);
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  tx_data_.tracking = control;
  tx_data_.yaw = yaw;
  tx_data_.pitch = pitch;
  tx_data_.fire = fire ? 1 : 0;
  tx_data_.fric_on = control ? 1 : 0;
  tx_data_.checksum = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.checksum));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    std::string hex;
    uint8_t * p = reinterpret_cast<uint8_t *>(&tx_data_);
    for (size_t i = 0; i < sizeof(tx_data_); ++i) {
      hex += fmt::format("{:02X} ", p[i]);
    }
    // tools::logger()->info("[Gimbal] TX: {}", hex);
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

bool Gimbal::read(uint8_t * buffer, size_t size)
{
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    // tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;
  }
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;
  auto last_stats_log = std::chrono::steady_clock::now();
  uint64_t prev_good = 0, prev_crc = 0, prev_short = 0, prev_header_mismatch = 0;
  uint64_t last_crc_sample = 0;

  while (!quit_) {
    if (error_count > 5000) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      {
        std::lock_guard<std::mutex> lock(mutex_);
        rx_stats_.reconnect_count++;
      }
      reconnect();
      continue;
    }

    if (!read(reinterpret_cast<uint8_t *>(&rx_data_.header), 1)) {
      error_count++;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        rx_stats_.short_read++;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    if (rx_data_.header != 0x5A) {
      std::lock_guard<std::mutex> lock(mutex_);
      rx_stats_.header_mismatch++;
      rx_stats_.last_header = rx_data_.header;
      continue;
    }

    auto t = std::chrono::steady_clock::now();
    std::array<uint8_t, kExtendedFrameSize> frame{};
    frame[0] = rx_data_.header;

    if (!read(frame.data() + 1, kLegacyFrameSize - 1)) {
      error_count++;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        rx_stats_.short_read++;
      }
      continue;
    }

    bool is_legacy = tools::check_crc16(frame.data(), kLegacyFrameSize);
    bool is_extended = false;

    if (!is_legacy) {
      if (!read(frame.data() + kLegacyFrameSize, kExtendedFrameSize - kLegacyFrameSize)) {
        error_count++;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          rx_stats_.short_read++;
        }
        continue;
      }
      is_extended = tools::check_crc16(frame.data(), kExtendedFrameSize);
    }

    if (!is_legacy && !is_extended) {
      error_count++;
      auto calc_crc_legacy = tools::get_crc16(frame.data(), kLegacyFrameSize - 2);
      auto rx_crc_legacy = static_cast<uint16_t>(
        frame[kLegacyFrameSize - 2] | (static_cast<uint16_t>(frame[kLegacyFrameSize - 1]) << 8));
      {
        std::lock_guard<std::mutex> lock(mutex_);
        rx_stats_.crc_fail++;
        rx_stats_.consecutive_crc_fail++;
        rx_stats_.last_rx_crc = rx_crc_legacy;
        rx_stats_.last_calc_crc = calc_crc_legacy;
      }
      auto snap = rx_stats();
      if (snap.crc_fail != last_crc_sample && snap.crc_fail % 200 == 1) {
        last_crc_sample = snap.crc_fail;
        tools::logger()->warn(
          "[Gimbal] CRC fail x{} (legacy_crc rx=0x{:04X} calc=0x{:04X}) frame_prefix=[{}]",
          snap.crc_fail, snap.last_rx_crc, snap.last_calc_crc,
          hex_prefix(frame.data(), kExtendedFrameSize));
      }
      continue;
    }

    error_count = 0;

    float yaw = 0.0f, pitch = 0.0f, roll = 0.0f;
    float yaw_odom = 0.0f, pitch_odom = 0.0f;
    float yaw_vel = 0.0f, pitch_vel = 0.0f;
    uint8_t robot_id = 0;

    if (is_extended) {
      std::memcpy(&rx_data_, frame.data(), sizeof(rx_data_));
      yaw = rx_data_.yaw;
      pitch = rx_data_.pitch;
      roll = rx_data_.roll;
      yaw_odom = rx_data_.yaw_odom;
      pitch_odom = rx_data_.pitch_odom;
      yaw_vel = rx_data_.yaw_vel;
      pitch_vel = rx_data_.pitch_vel;
      robot_id = rx_data_.robot_id;
    } else {
      const uint8_t flags = frame[1];
      (void)flags;
      roll = unpack_float(frame.data() + 2);
      pitch = unpack_float(frame.data() + 6);
      yaw = unpack_float(frame.data() + 10);
    }

    // Euler to Quaternion (Z-Y-X convolution: Yaw-Pitch-Roll)
    Eigen::Quaterniond q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                           Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

    queue_.push({q, t});

    {
      std::lock_guard<std::mutex> lock(mutex_);
      state_.yaw = yaw;
      state_.yaw_vel = yaw_vel;
      state_.pitch = pitch;
      state_.pitch_vel = pitch_vel;
      state_.roll = roll;
      state_.yaw_odom = yaw_odom;
      state_.pitch_odom = pitch_odom;
      state_.bullet_speed = 0;
      state_.bullet_count = 0;
      state_.robot_id = robot_id;
      mode_ = GimbalMode::AUTO_AIM;

      rx_stats_.good_frames++;
      rx_stats_.consecutive_crc_fail = 0;
      rx_stats_.last_good_frame_time = t;
      rx_stats_.last_header = 0x5A;
    }

    auto now = std::chrono::steady_clock::now();
    if (tools::delta_time(now, last_stats_log) >= 1.0) {
      auto snap = rx_stats();
      auto d_good = snap.good_frames - prev_good;
      auto d_crc = snap.crc_fail - prev_crc;
      auto d_short = snap.short_read - prev_short;
      auto d_header = snap.header_mismatch - prev_header_mismatch;
      if (d_crc + d_short + d_header > 0) {
        tools::logger()->warn(
          "[Gimbal][1s] good={} crc_fail={} short_read={} bad_header={} total(good={} crc={} short={} bad_header={})",
          d_good, d_crc, d_short, d_header, snap.good_frames, snap.crc_fail, snap.short_read,
          snap.header_mismatch);
      }
      prev_good = snap.good_frames;
      prev_crc = snap.crc_fail;
      prev_short = snap.short_read;
      prev_header_mismatch = snap.header_mismatch;
      last_stats_log = now;
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    try {
      serial_.close();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      serial_.open();  // 尝试重新打开
      queue_.clear();
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io
