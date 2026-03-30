#include "gimbal.hpp"

#include <array>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <sstream>

#include <fmt/core.h>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace io
{
namespace
{
constexpr size_t kExtendedFrameSize = sizeof(GimbalToVision);

static_assert(kExtendedFrameSize == 49, "Unexpected GimbalToVision size.");

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

std::string to_lower_copy(std::string value)
{
  for (char & c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

Eigen::Quaterniond quaternion_from_packet(const GimbalToVision & packet)
{
  return Eigen::AngleAxisd(packet.yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(packet.pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(packet.roll, Eigen::Vector3d::UnitX());
}

}  // namespace

Gimbal::Gimbal(const std::string & config_path, bool wait_for_first_q)
{
  auto yaml = tools::load(config_path);

  if (yaml["gimbal_to_vision_topic"]) {
    gimbal_to_vision_topic_ = yaml["gimbal_to_vision_topic"].as<std::string>();
  }
  if (yaml["vision_to_gimbal_topic"]) {
    vision_to_gimbal_topic_ = yaml["vision_to_gimbal_topic"].as<std::string>();
  }
  if (yaml["gimbal_ros2_node_name"]) {
    ros2_node_name_ = yaml["gimbal_ros2_node_name"].as<std::string>();
  }

  std::string transport = "serial";
  if (yaml["gimbal_transport"]) {
    transport = to_lower_copy(yaml["gimbal_transport"].as<std::string>());
  }
  use_ros2_transport_ = (transport == "ros2");

  if (use_ros2_transport_) {
#ifdef SP_HAS_ROS2_CORE
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
      owns_rclcpp_context_ = true;
    }

    ros2_node_ = std::make_shared<rclcpp::Node>(ros2_node_name_);
    ros2_tx_publisher_ =
      ros2_node_->create_publisher<std_msgs::msg::UInt8MultiArray>(vision_to_gimbal_topic_, 10);
    ros2_rx_subscription_ = ros2_node_->create_subscription<std_msgs::msg::UInt8MultiArray>(
      gimbal_to_vision_topic_, 10,
      [this](const std_msgs::msg::UInt8MultiArray::SharedPtr msg) {
        if (msg->data.size() != sizeof(GimbalToVision)) {
          RCLCPP_WARN(
            ros2_node_->get_logger(), "Ignore %s with invalid size %zu, expected %zu",
            gimbal_to_vision_topic_.c_str(), msg->data.size(), sizeof(GimbalToVision));
          return;
        }

        auto packet = from_bytes<GimbalToVision>(msg->data.data());
        if (packet.header != kGimbalToVisionHeader) {
          RCLCPP_WARN(
            ros2_node_->get_logger(), "Ignore %s with bad header 0x%02X",
            gimbal_to_vision_topic_.c_str(), packet.header);
          return;
        }
        if (!validate_crc16(packet)) {
          RCLCPP_WARN(
            ros2_node_->get_logger(), "Ignore %s with invalid CRC", gimbal_to_vision_topic_.c_str());
          return;
        }

        handle_rx_packet(packet, std::chrono::steady_clock::now());
      });

    ros2_spin_thread_ = std::thread([this]() {
      while (!quit_ && rclcpp::ok()) {
        rclcpp::spin_some(ros2_node_);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });

    tools::logger()->info(
      "[Gimbal] ROS2 transport enabled: rx={} tx={}", gimbal_to_vision_topic_,
      vision_to_gimbal_topic_);
#else
    tools::logger()->error(
      "[Gimbal] gimbal_transport=ros2 but ROS2 support was not compiled into this binary.");
    exit(1);
#endif
  } else {
    auto com_port = tools::read<std::string>(yaml, "com_port");

    try {
      serial_.setPort(com_port);
      serial_.setBaudrate(115200);
      serial::Timeout timeout = serial::Timeout::simpleTimeout(100);
      serial_.setTimeout(timeout);
      serial_.open();
      serial_open_ = true;
    } catch (const std::exception & e) {
      tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
      exit(1);
    }

    thread_ = std::thread(&Gimbal::read_thread, this);
  }

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
#ifdef SP_HAS_ROS2_CORE
  if (ros2_spin_thread_.joinable()) ros2_spin_thread_.join();
  if (owns_rclcpp_context_ && rclcpp::ok()) {
    rclcpp::shutdown();
  }
#endif
  if (serial_open_) {
    serial_.close();
    serial_open_ = false;
  }
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

void Gimbal::send(io::VisionToGimbal packet)
{
  packet.header = kVisionToGimbalHeader;
  refresh_crc16(packet);
  send_packet(packet);
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  (void)yaw_vel;
  (void)yaw_acc;
  (void)pitch_vel;
  (void)pitch_acc;

  VisionToGimbal packet{};
  packet.header = kVisionToGimbalHeader;
  packet.tracking = control ? 1 : 0;
  packet.yaw = yaw;
  packet.pitch = pitch;
  packet.fire = fire ? 1 : 0;
  packet.fric_on = control ? 1 : 0;
  refresh_crc16(packet);
  send_packet(packet);
}

void Gimbal::send_packet(const VisionToGimbal & packet)
{
  tx_data_ = packet;

  if (use_ros2_transport_) {
#ifdef SP_HAS_ROS2_CORE
    std_msgs::msg::UInt8MultiArray message;
    auto bytes = to_bytes(packet);
    message.data.assign(bytes.begin(), bytes.end());
    ros2_tx_publisher_->publish(message);
#endif
    return;
  }

  try {
    serial_.write(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

bool Gimbal::read(uint8_t * buffer, size_t size)
{
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    (void)e;
    return false;
  }
}

void Gimbal::handle_rx_packet(
  const GimbalToVision & packet, const std::chrono::steady_clock::time_point & timestamp)
{
  rx_data_ = packet;

  auto q = quaternion_from_packet(packet).normalized();
  queue_.push({q, timestamp});

  std::lock_guard<std::mutex> lock(mutex_);
  state_.yaw = packet.yaw;
  state_.yaw_vel = packet.yaw_vel;
  state_.pitch = packet.pitch;
  state_.pitch_vel = packet.pitch_vel;
  state_.roll = packet.roll;
  state_.yaw_odom = packet.yaw_odom;
  state_.pitch_odom = packet.pitch_odom;
  state_.bullet_speed = 0;
  state_.bullet_count = 0;
  state_.robot_id = packet.robot_id;
  mode_ = GimbalMode::AUTO_AIM;

  rx_stats_.good_frames++;
  rx_stats_.consecutive_crc_fail = 0;
  rx_stats_.last_good_frame_time = timestamp;
  rx_stats_.last_header = packet.header;
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

    if (rx_data_.header != kGimbalToVisionHeader) {
      std::lock_guard<std::mutex> lock(mutex_);
      rx_stats_.header_mismatch++;
      rx_stats_.last_header = rx_data_.header;
      continue;
    }

    auto t = std::chrono::steady_clock::now();
    std::array<uint8_t, kExtendedFrameSize> frame{};
    frame[0] = rx_data_.header;

    if (!read(frame.data() + 1, kExtendedFrameSize - 1)) {
      error_count++;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        rx_stats_.short_read++;
      }
      continue;
    }

    auto packet = from_bytes<GimbalToVision>(frame.data());
    if (!validate_crc16(packet)) {
      error_count++;
      auto calc_crc = compute_crc16(packet);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        rx_stats_.crc_fail++;
        rx_stats_.consecutive_crc_fail++;
        rx_stats_.last_rx_crc = packet.checksum;
        rx_stats_.last_calc_crc = calc_crc;
      }
      auto snap = rx_stats();
      if (snap.crc_fail != last_crc_sample && snap.crc_fail % 200 == 1) {
        last_crc_sample = snap.crc_fail;
        tools::logger()->warn(
          "[Gimbal] CRC fail x{} (crc49 rx=0x{:04X} calc=0x{:04X}) frame_prefix=[{}]",
          snap.crc_fail, snap.last_rx_crc, snap.last_calc_crc,
          hex_prefix(frame.data(), kExtendedFrameSize));
      }
      continue;
    }

    error_count = 0;
    handle_rx_packet(packet, t);

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
      serial_open_ = false;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      serial_.open();
      serial_open_ = true;
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
