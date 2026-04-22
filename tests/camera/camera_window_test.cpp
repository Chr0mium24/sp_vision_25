#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tools/runtime/exiter.hpp"
#include "tools/runtime/logger.hpp"
#include "tools/math/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }"
  "{scale          | 0.5                    | 图像缩放比例}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>(0);
  auto scale = cli.get<double>("scale");
  if (scale <= 0.0) {
    scale = 1.0;
  }

  tools::Exiter exiter;
  io::Camera camera(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  auto last_stamp = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    if (img.empty()) {
      continue;
    }

    auto dt = tools::delta_time(timestamp, last_stamp);
    last_stamp = timestamp;
    tools::logger()->info("{:.2f} fps", 1.0 / dt);

    if (scale != 1.0) {
      cv::resize(img, img, {}, scale, scale);
    }
    cv::imshow("Minimum Vision Camera", img);
    if (cv::waitKey(1) == 'q') break;
  }

  return 0;
}
