#include <fmt/core.h>

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml   | yaml配置文件路径 }"
  "{output-folder o| assets/camera_captures   | 输出文件夹路径   }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>(0);
  auto output_folder = cli.get<std::string>("output-folder");
  std::filesystem::create_directories(output_folder);

  tools::Exiter exiter;
  io::Camera camera(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  int count = 0;

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    if (img.empty()) {
      continue;
    }

    cv::imshow("Camera Save Test (s:save, q:quit)", img);
    int key = cv::waitKey(1);
    if (key == 'q') {
      break;
    }
    if (key != 's') {
      continue;
    }

    count++;
    auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
    cv::imwrite(img_path, img);
    tools::logger()->info("[{}] Saved {}", count, img_path);
  }

  return 0;
}
