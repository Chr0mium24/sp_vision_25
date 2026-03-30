#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>

#include "io/gimbal/gimbal.hpp"
#include "serial/serial.h"
#include "tools/yaml.hpp"

namespace
{
struct BridgeOptions
{
  std::string config_path;
  std::vector<std::string> ports;
  int baud = 115200;
  int reopen_ms = 1000;
  int loop_sleep_ms = 2;
  std::string gimbal_to_vision_topic = "/gimbal_to_vision";
  std::string vision_to_gimbal_topic = "/vision_to_gimbal";
  std::string node_name = "sp_vision_gimbal_bridge";
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
    "Usage: gimbal_ros2_bridge [config_path] [--ports=/dev/ttyACM0,/dev/ttyUSB0] [--baud=115200]\n"
    "                         [--gimbal-to-vision-topic=/gimbal_to_vision]\n"
    "                         [--vision-to-gimbal-topic=/vision_to_gimbal]\n"
    "                         [--node-name=sp_vision_gimbal_bridge] [--reopen-ms=1000]\n"
    "                         [--loop-sleep-ms=2]\n");
}

BridgeOptions parse_args(int argc, char * argv[])
{
  BridgeOptions options;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
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
    } else if (auto value = parse_value("--gimbal-to-vision-topic="); !value.empty()) {
      options.gimbal_to_vision_topic = value;
    } else if (auto value = parse_value("--vision-to-gimbal-topic="); !value.empty()) {
      options.vision_to_gimbal_topic = value;
    } else if (auto value = parse_value("--node-name="); !value.empty()) {
      options.node_name = value;
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

void load_config_overrides(BridgeOptions & options)
{
  if (options.config_path.empty()) {
    return;
  }

  auto yaml = tools::load(options.config_path);
  if (options.ports.empty() && yaml["com_port"]) {
    options.ports.push_back(yaml["com_port"].as<std::string>());
  }
  if (options.gimbal_to_vision_topic == "/gimbal_to_vision" && yaml["gimbal_to_vision_topic"]) {
    options.gimbal_to_vision_topic = yaml["gimbal_to_vision_topic"].as<std::string>();
  }
  if (options.vision_to_gimbal_topic == "/vision_to_gimbal" && yaml["vision_to_gimbal_topic"]) {
    options.vision_to_gimbal_topic = yaml["vision_to_gimbal_topic"].as<std::string>();
  }
  if (options.node_name == "sp_vision_gimbal_bridge" && yaml["gimbal_ros2_node_name"]) {
    options.node_name = yaml["gimbal_ros2_node_name"].as<std::string>() + "_bridge";
  }
}

class GimbalRos2Bridge : public rclcpp::Node
{
public:
  explicit GimbalRos2Bridge(BridgeOptions options)
  : Node(options.node_name), options_(std::move(options))
  {
    publisher_ = create_publisher<std_msgs::msg::UInt8MultiArray>(options_.gimbal_to_vision_topic, 10);
    subscription_ = create_subscription<std_msgs::msg::UInt8MultiArray>(
      options_.vision_to_gimbal_topic, 10,
      std::bind(&GimbalRos2Bridge::handle_tx_message, this, std::placeholders::_1));

    if (options_.ports.empty()) {
      throw std::runtime_error("No serial ports configured for gimbal_ros2_bridge.");
    }

    RCLCPP_INFO(
      get_logger(), "bridge ready: ports=%zu baud=%d rx=%s tx=%s", options_.ports.size(),
      options_.baud, options_.gimbal_to_vision_topic.c_str(),
      options_.vision_to_gimbal_topic.c_str());
  }

  ~GimbalRos2Bridge() override
  {
    close_serial();
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
    if (msg->data.size() != sizeof(io::VisionToGimbal)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "ignore %s with invalid payload size %zu",
        options_.vision_to_gimbal_topic.c_str(), msg->data.size());
      return;
    }

    const auto packet = io::from_bytes<io::VisionToGimbal>(msg->data.data());
    if (packet.header != io::kVisionToGimbalHeader) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "ignore %s with bad header 0x%02X",
        options_.vision_to_gimbal_topic.c_str(), packet.header);
      return;
    }
    if (!io::validate_crc16(packet)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "ignore %s with invalid CRC",
        options_.vision_to_gimbal_topic.c_str());
      return;
    }

    std::lock_guard<std::mutex> lock(serial_mutex_);
    if (!serial_.isOpen()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "serial is closed, drop %s command",
        options_.vision_to_gimbal_topic.c_str());
      return;
    }

    try {
      serial_.write(msg->data.data(), msg->data.size());
      ++tx_frames_;
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "serial write failed: %s", e.what());
      close_serial_unlocked();
    }
  }

  void poll_serial_once()
  {
    ensure_serial_open();
    if (!serial_.isOpen()) {
      return;
    }

    uint8_t header = 0;
    if (!read_exact(&header, 1)) {
      return;
    }

    if (header != io::kGimbalToVisionHeader) {
      ++bad_headers_;
      return;
    }

    std::array<uint8_t, sizeof(io::GimbalToVision)> frame{};
    frame[0] = header;
    if (!read_exact(frame.data() + 1, frame.size() - 1)) {
      ++short_reads_;
      return;
    }

    const auto packet = io::from_bytes<io::GimbalToVision>(frame.data());
    if (!io::validate_crc16(packet)) {
      ++crc_fail_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "drop serial frame with invalid CRC");
      return;
    }

    std_msgs::msg::UInt8MultiArray message;
    message.data.assign(frame.begin(), frame.end());
    publisher_->publish(message);
    ++rx_frames_;
  }

  bool read_exact(uint8_t * buffer, size_t size)
  {
    std::lock_guard<std::mutex> lock(serial_mutex_);
    if (!serial_.isOpen()) {
      return false;
    }

    try {
      return serial_.read(buffer, size) == size;
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "serial read failed: %s", e.what());
      close_serial_unlocked();
      return false;
    }
  }

  void ensure_serial_open()
  {
    if (serial_.isOpen()) {
      return;
    }

    auto now = std::chrono::steady_clock::now();
    auto reopen_age = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_open_try_).count();
    if (reopen_age < options_.reopen_ms) {
      return;
    }
    last_open_try_ = now;

    for (const auto & port : options_.ports) {
      try {
        serial_.setPort(port);
        serial_.setBaudrate(static_cast<uint32_t>(options_.baud));
        serial_.setTimeout(serial::Timeout::simpleTimeout(20));
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
  BridgeOptions options_;
  serial::Serial serial_;
  std::mutex serial_mutex_;
  std::chrono::steady_clock::time_point last_open_try_{};
  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr publisher_;
  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr subscription_;
  uint64_t rx_frames_ = 0;
  uint64_t tx_frames_ = 0;
  uint64_t crc_fail_ = 0;
  uint64_t short_reads_ = 0;
  uint64_t bad_headers_ = 0;
};

}  // namespace

int main(int argc, char * argv[])
{
  try {
    auto options = parse_args(argc, argv);
    load_config_overrides(options);

    if (options.ports.empty()) {
      options.ports.push_back("/dev/ttyACM0");
    }

    rclcpp::init(argc, argv);
    auto bridge = std::make_shared<GimbalRos2Bridge>(options);
    bridge->spin_loop();
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception & e) {
    std::fprintf(stderr, "gimbal_ros2_bridge: %s\n", e.what());
    return 1;
  }
}
