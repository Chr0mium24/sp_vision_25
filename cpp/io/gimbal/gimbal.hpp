#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "serial/serial.h"
#include "tools/concurrency/thread_safe_queue.hpp"

namespace io
{
struct __attribute__((packed)) GimbalToVision
{
  uint8_t header = 0x5A;
  uint8_t detect_color : 1;  // 0: 红, 1: 蓝
  uint8_t reset_tracker : 1;
  uint8_t reserved : 6;
  float yaw;    // rad
  float pitch;  // rad, pitch up is positive
  float roll;   // rad
  float yaw_odom;
  float pitch_odom;
  float yaw_vel;    // rad/s
  float pitch_vel;  // rad/s
  float roll_vel;   // rad/s
  float aim_x;
  float aim_y;
  float aim_z;
  uint8_t robot_id;
  uint16_t checksum;
};

static_assert(sizeof(GimbalToVision) <= 64);

struct __attribute__((packed)) VisionToGimbal
{
  uint8_t header = 0xA5;
  uint8_t tracking;
  float pitch;  // rad, pitch up is positive
  float yaw;
  uint8_t fire;
  uint8_t fric_on;
  uint16_t checksum;
};

static_assert(sizeof(VisionToGimbal) <= 64);

enum class GimbalMode
{
  IDLE,        // 空闲
  AUTO_AIM,    // 自瞄
  SMALL_BUFF,  // 小符
  BIG_BUFF     // 大符
};

struct GimbalState
{
  float yaw = 0.0f;
  float yaw_vel = 0.0f;
  float pitch = 0.0f;      // rad, pitch up is positive
  float pitch_vel = 0.0f;  // rad/s, pitch up is positive
  float roll = 0.0f;
  float yaw_odom = 0.0f;
  float pitch_odom = 0.0f;  // pitch up is positive
  float bullet_speed = 0.0f;
  uint16_t bullet_count = 0;
  uint8_t robot_id = 0;
};

struct GimbalRxStats
{
  uint64_t good_frames = 0;
  uint64_t crc_fail = 0;
  uint64_t short_read = 0;
  uint64_t header_mismatch = 0;
  uint64_t reconnect_count = 0;
  uint64_t consecutive_crc_fail = 0;
  uint8_t last_header = 0;
  uint16_t last_rx_crc = 0;
  uint16_t last_calc_crc = 0;
  std::chrono::steady_clock::time_point last_good_frame_time{};
};

class Gimbal
{
public:
  explicit Gimbal(const std::string & config_path, bool wait_for_first_q = true);

  ~Gimbal();

  GimbalMode mode() const;
  GimbalState state() const;
  GimbalRxStats rx_stats() const;
  bool has_valid_q() const;
  std::string str(GimbalMode mode) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);

  void send(io::VisionToGimbal VisionToGimbal);

private:
  struct SendTransform
  {
    float yaw_scale = 1.0f;
    float yaw_bias_rad = 0.0f;
    float pitch_scale = 1.0f;
    float pitch_bias_rad = 0.0f;
  };

  serial::Serial serial_;

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;

  GimbalToVision rx_data_{};
  VisionToGimbal tx_data_{};
  SendTransform send_transform_{};

  GimbalMode mode_ = GimbalMode::IDLE;
  GimbalState state_;
  GimbalRxStats rx_stats_;
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};

  void apply_send_transform(float & yaw, float & pitch) const;
  bool read(uint8_t * buffer, size_t size);
  void read_thread();
  void reconnect();
};

}  // namespace io

#endif  // IO__GIMBAL_HPP
