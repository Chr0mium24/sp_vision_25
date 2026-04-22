#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/auto_aim_runtime.hpp"
#include "tools/runtime/exiter.hpp"
#include "tools/runtime/logger.hpp"

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  auto_aim::Runtime runtime(config_path, false);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  auto mode = io::GimbalMode::IDLE;
  auto last_mode = io::GimbalMode::IDLE;

  while (!exiter.exit()) {
    camera.read(img, t);
    if (img.empty()) continue;

    auto q = gimbal.q(t - std::chrono::milliseconds(1));
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode;
    }

    auto output = runtime.step({img, t, q, 25.0});
    auto command = output.command;

    gimbal.send(
      command.control, command.shoot, static_cast<float>(command.yaw), 0, 0,
      static_cast<float>(command.pitch), 0, 0);
  }

  return 0;
}
