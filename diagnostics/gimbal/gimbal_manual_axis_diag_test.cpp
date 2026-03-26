#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "io/gimbal/gimbal.hpp"
#include "tools/exiter.hpp"

using namespace std::chrono_literals;

namespace
{
struct Sample
{
  double yaw = 0.0;
  double pitch = 0.0;
  double roll = 0.0;
  int count = 0;
};

struct Step
{
  std::string name;
  std::string prompt;
};

Sample average_state(io::Gimbal & gimbal, tools::Exiter & exiter, int duration_ms, int loop_ms)
{
  Sample sample;
  auto start = std::chrono::steady_clock::now();
  while (!exiter.exit()) {
    if (std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(duration_ms)) break;
    auto gs = gimbal.state();
    sample.yaw += gs.yaw;
    sample.pitch += gs.pitch;
    sample.roll += gs.roll;
    sample.count++;
    std::this_thread::sleep_for(std::chrono::milliseconds(loop_ms));
  }

  if (sample.count > 0) {
    sample.yaw /= sample.count;
    sample.pitch /= sample.count;
    sample.roll /= sample.count;
  }
  return sample;
}

bool wait_valid(io::Gimbal & gimbal, tools::Exiter & exiter, int wait_valid_ms)
{
  auto start = std::chrono::steady_clock::now();
  while (!exiter.exit()) {
    if (gimbal.has_valid_q()) return true;
    if (std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(wait_valid_ms)) {
      return false;
    }
    std::this_thread::sleep_for(5ms);
  }
  return false;
}

std::array<double, 3> delta_deg(const Sample & after, const Sample & before)
{
  return {
    (after.yaw - before.yaw) * 57.3,
    (after.pitch - before.pitch) * 57.3,
    (after.roll - before.roll) * 57.3,
  };
}

int dominant_axis(const std::array<double, 3> & delta)
{
  int axis = 0;
  for (int i = 1; i < 3; ++i) {
    if (std::abs(delta[i]) > std::abs(delta[axis])) axis = i;
  }
  return axis;
}

const char * axis_name(int axis)
{
  switch (axis) {
    case 0:
      return "yaw";
    case 1:
      return "pitch";
    case 2:
      return "roll";
    default:
      return "unknown";
  }
}
}  // namespace

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }"
  "{wait-valid-ms  | 1500 | 等待有效反馈超时(ms) }"
  "{sample-ms      | 700 | 每次保持动作时的采样时长(ms) }"
  "{loop-ms        | 5 | 采样循环sleep时长(ms) }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  const int wait_valid_ms = std::max(100, cli.get<int>("wait-valid-ms"));
  const int sample_ms = std::max(100, cli.get<int>("sample-ms"));
  const int loop_ms = std::max(1, cli.get<int>("loop-ms"));

  tools::Exiter exiter;
  io::Gimbal gimbal(config_path, false);
  if (!wait_valid(gimbal, exiter, wait_valid_ms)) {
    std::fprintf(stderr, "[manual-axis] failed to receive valid gimbal feedback within %d ms\n", wait_valid_ms);
    return 2;
  }

  const std::vector<Step> steps{
    {"up", "请手动向上抬枪口，保持住后按回车"},
    {"down", "请手动向下压枪口，保持住后按回车"},
    {"right", "请手动向右转云台，保持住后按回车"},
    {"left", "请手动向左转云台，保持住后按回车"},
  };

  std::puts("[manual-axis] 纯读取模式，不会下发控制。每一步先回到自然位置。");
  std::puts("[manual-axis] 采样前会先记录当前基准，再记录你保持动作时的反馈变化。");
  std::puts("");

  for (const auto & step : steps) {
    std::puts("------------------------------------------------------------");
    std::printf("[manual-axis] step=%s\n", step.name.c_str());
    std::puts("[manual-axis] 先回到自然位置，然后按回车记录 baseline");
    std::string line;
    std::getline(std::cin, line);
    const auto baseline = average_state(gimbal, exiter, sample_ms, loop_ms);

    std::printf("[manual-axis] baseline(deg): yaw=%+.2f pitch=%+.2f roll=%+.2f\n", baseline.yaw * 57.3,
      baseline.pitch * 57.3, baseline.roll * 57.3);
    std::puts(step.prompt.c_str());
    std::getline(std::cin, line);
    const auto moved = average_state(gimbal, exiter, sample_ms, loop_ms);

    const auto delta = delta_deg(moved, baseline);
    const int axis = dominant_axis(delta);
    std::printf(
      "[manual-axis] step=%s delta(deg): yaw=%+.2f pitch=%+.2f roll=%+.2f dominant=%s\n",
      step.name.c_str(), delta[0], delta[1], delta[2], axis_name(axis));
  }

  std::puts("------------------------------------------------------------");
  std::puts("[manual-axis] 解释：");
  std::puts("  up/down 主要应该对应 pitch");
  std::puts("  right/left 主要应该对应 yaw");
  std::puts("  如果 up/down 主要对应了 roll，说明 C 板安装轴和枪口轴存在 90 度映射问题");
  std::puts("  如果 up 主要对应 pitch 但符号和预期相反，更像是 pitch 符号反了");
  return 0;
}
