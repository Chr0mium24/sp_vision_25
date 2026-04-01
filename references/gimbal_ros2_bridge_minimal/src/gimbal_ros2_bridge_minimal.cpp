#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include <yaml-cpp/yaml.h>

#include "bridge_packets.hpp"
#include "serial/serial.h"

namespace
{
using minimal_bridge::GimbalToVision;
using minimal_bridge::VisionToGimbal;

struct Options
{
  std::string config_path;
  std::vector<std::string> ports;
  int baud = 115200;
  int reopen_ms = 1000;
  int loop_sleep_ms = 2;
  std::string gimbal_to_vision_topic = "/gimbal_to_vision";
  std::string vision_to_gimbal_topic = "/vision_to_gimbal";
  std::string node_name = "sp_vision_gimbal_transport_bridge";
};

std::string trim_copy(const std::string & value)
{
  size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

std::vector<std::string> split_csv(const std::string & csv)
{
  std::vector<std::string> tokens;
  size_t start = 0;
  while (start <= csv.size()) {
    size_t comma = csv.find(',', start);
    auto token = trim_copy(csv.substr(
      start, comma == std::string::npos ? std::string::npos : comma - start));
    if (!token.empty()) {
      tokens.push_back(token);
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return tokens;
}

void print_usage()
{
  std::printf(
    "Usage: gimbal_ros2_bridge_minimal [config_path] [--ports=/dev/ttyACM0,/dev/ttyUSB0]\n"
    "                                  [--baud=115200] [--reopen-ms=1000]\n"
    "                                  [--loop-sleep-ms=2]\n");
}

Options parse_args(int argc, char * argv[])
{
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage();
      std::exit(0);
    }

    auto parse_value = [&](const std::string & prefix) -> std::string {
      if (arg.rfind(prefix, 0) == 0) {
        return arg.substr(prefix.size());
      }
      return {};
    };

    if (auto value = parse_value("--ports="); !value.empty()) {
      options.ports = split_csv(value);
    } else if (auto value = parse_value("--baud="); !value.empty()) {
      options.baud = std::max(1, std::stoi(value));
    } else if (auto value = parse_value("--reopen-ms="); !value.empty()) {
      options.reopen_ms = std::max(10, std::stoi(value));
    } else if (auto value = parse_value("--loop-sleep-ms="); !value.empty()) {
      options.loop_sleep_ms = std::max(0, std::stoi(value));
    } else if (!arg.empty() && arg[0] != '-' && options.config_path.empty()) {
      options.config_path = arg;
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  return options;
}

void load_config(Options & options)
{
  if (options.config_path.empty()) {
    return;
  }

  YAML::Node yaml = YAML::LoadFile(options.config_path);
  if (options.ports.empty() && yaml["com_port"]) {
    options.ports.push_back(yaml["com_port"].as<std::string>());
  }
  if (yaml["baud"]) {
    options.baud = yaml["baud"].as<int>();
  }
  if (yaml["gimbal_to_vision_topic"]) {
    options.gimbal_to_vision_topic = yaml["gimbal_to_vision_topic"].as<std::string>();
  }
  if (yaml["vision_to_gimbal_topic"]) {
    options.vision_to_gimbal_topic = yaml["vision_to_gimbal_topic"].as<std::string>();
  }
  if (yaml["gimbal_ros2_node_name"]) {
    options.node_name = yaml["gimbal_ros2_node_name"].as<std::string>() + "_bridge";
  }
  if (yaml["reopen_ms"]) {
    options.reopen_ms = yaml["reopen_ms"].as<int>();
  }
  if (yaml["loop_sleep_ms"]) {
    options.loop_sleep_ms = yaml["loop_sleep_ms"].as<int>();
  }
}

class BridgeNode : public rclcpp::Node
{
public:
  explicit BridgeNode(Options options)
  : Node(options.node_name), options_(std::move(options))
  {
    publisher_ =
      create_publisher<std_msgs::msg::UInt8MultiArray>(options_.gimbal_to_vision_topic, 10);
    subscription_ = create_subscription<std_msgs::msg::UInt8MultiArray>(
      options_.vision_to_gimbal_topic, 10,
      std::bind(&BridgeNode::handle_tx_message, this, std::placeholders::_1));

    if (options_.ports.empty()) {
      throw std::runtime_error("No serial ports configured.");
    }

    RCLCPP_INFO(
      get_logger(), "bridge ready: ports=%zu baud=%d rx=%s tx=%s", options_.ports.size(),
      options_.baud, options_.gimbal_to_vision_topic.c_str(),
      options_.vision_to_gimbal_topic.c_str());
  }

  void spin_loop()
  {
    while (rclcpp::ok()) {
      rclcpp::spin_some(shared_from_this());
      poll_serial_once();
      if (options_.loop_sleep_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(options_.loop_sleep_ms));
      }
    }
  }

private:
  void handle_tx_message(const std_msgs::msg::UInt8MultiArray::SharedPtr msg)
  {
    if (msg->data.size() != sizeof(VisionToGimbal)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "ignore tx payload size=%zu", msg->data.size());
      return;
    }

    const auto packet = minimal_bridge::from_bytes<VisionToGimbal>(msg->data.data());
    if (packet.header != minimal_bridge::kVisionToGimbalHeader) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "ignore tx with bad header");
      return;
    }
    if (!minimal_bridge::validate_crc16(packet)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "ignore tx with invalid CRC");
      return;
    }

    std::lock_guard<std::mutex> lock(serial_mutex_);
    if (!serial_.isOpen()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "serial is closed, drop tx");
      return;
    }

    try {
      serial_.write(msg->data.data(), msg->data.size());
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "serial write failed: %s", e.what());
      close_serial_unlocked();
    }
  }

  void parse_stream()
  {
    while (true) {
      auto it = std::find(
        stream_buffer_.begin(), stream_buffer_.end(), minimal_bridge::kGimbalToVisionHeader);
      if (it == stream_buffer_.end()) {
        if (stream_buffer_.size() > sizeof(GimbalToVision) - 1) {
          stream_buffer_.erase(
            stream_buffer_.begin(),
            stream_buffer_.begin() + static_cast<long>(stream_buffer_.size() - (sizeof(GimbalToVision) - 1)));
        }
        return;
      }

      if (it != stream_buffer_.begin()) {
        stream_buffer_.erase(stream_buffer_.begin(), it);
      }

      if (stream_buffer_.size() < sizeof(GimbalToVision)) {
        return;
      }

      const auto packet = minimal_bridge::from_bytes<GimbalToVision>(stream_buffer_.data());
      if (!minimal_bridge::validate_crc16(packet)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "drop rx with invalid CRC");
        stream_buffer_.erase(stream_buffer_.begin());
        continue;
      }

      std_msgs::msg::UInt8MultiArray message;
      message.data.assign(
        stream_buffer_.begin(),
        stream_buffer_.begin() + static_cast<long>(sizeof(GimbalToVision)));
      publisher_->publish(message);
      stream_buffer_.erase(
        stream_buffer_.begin(),
        stream_buffer_.begin() + static_cast<long>(sizeof(GimbalToVision)));
    }
  }

  void poll_serial_once()
  {
    ensure_serial_open();
    if (!serial_.isOpen()) {
      return;
    }

    std::vector<uint8_t> chunk;
    try {
      std::lock_guard<std::mutex> lock(serial_mutex_);
      if (!serial_.isOpen()) {
        return;
      }
      const size_t avail = serial_.available();
      if (avail == 0) {
        return;
      }
      const size_t to_read = std::min<size_t>(avail, 4096);
      chunk.reserve(to_read);
      const size_t got = serial_.read(chunk, to_read);
      if (got == 0) {
        return;
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "serial read failed: %s", e.what());
      close_serial();
      return;
    }

    stream_buffer_.insert(stream_buffer_.end(), chunk.begin(), chunk.end());
    parse_stream();
  }

  void ensure_serial_open()
  {
    if (serial_.isOpen()) {
      return;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto reopen_age =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - last_open_try_).count();
    if (reopen_age < options_.reopen_ms) {
      return;
    }
    last_open_try_ = now;

    for (const auto & port : options_.ports) {
      try {
        serial_.setPort(port);
        serial_.setBaudrate(static_cast<uint32_t>(options_.baud));
        serial::Timeout timeout = serial::Timeout::simpleTimeout(20);
        serial_.setTimeout(timeout);
        serial_.open();
        RCLCPP_INFO(get_logger(), "serial connected on %s", port.c_str());
        return;
      } catch (const std::exception & e) {
        close_serial();
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000, "failed to open %s: %s", port.c_str(), e.what());
      }
    }
  }

  void close_serial()
  {
    std::lock_guard<std::mutex> lock(serial_mutex_);
    close_serial_unlocked();
  }

  void close_serial_unlocked()
  {
    if (!serial_.isOpen()) {
      return;
    }
    try {
      serial_.close();
    } catch (...) {
    }
  }

private:
  Options options_;
  serial::Serial serial_;
  std::mutex serial_mutex_;
  std::chrono::steady_clock::time_point last_open_try_{};
  std::vector<uint8_t> stream_buffer_;
  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr publisher_;
  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr subscription_;
};

}  // namespace

int main(int argc, char * argv[])
{
  try {
    auto options = parse_args(argc, argv);
    load_config(options);
    if (options.ports.empty()) {
      options.ports.push_back("/dev/ttyACM0");
    }

    rclcpp::init(argc, argv);
    auto node = std::make_shared<BridgeNode>(options);
    node->spin_loop();
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception & e) {
    std::fprintf(stderr, "gimbal_ros2_bridge_minimal: %s\n", e.what());
    return 1;
  }
}
