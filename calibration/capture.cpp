#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ?  |                          | 输出命令行参数说明}"
  "{@config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{output-folder o |      assets/img_with_q   | 输出文件夹路径   }"
  "{imu             | false                    | 启用IMU }";

void write_q(const std::string q_path, const Eigen::Quaterniond & q)
{
  std::ofstream q_file(q_path);
  Eigen::Vector4d xyzw = q.coeffs();
  // 输出顺序为wxyz
  q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  q_file.close();
}

void capture_loop(
  const std::string & config_path, const std::string & output_folder, bool use_imu,
  std::unique_ptr<io::Gimbal> gimbal)
{
  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  // 读取标定板配置
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  std::string pattern_type = "circle_grid";
  if (yaml["pattern_type"]) {
    pattern_type = yaml["pattern_type"].as<std::string>();
  }
  std::transform(pattern_type.begin(), pattern_type.end(), pattern_type.begin(), ::tolower);
  cv::Size pattern_size(pattern_cols, pattern_rows);

  int count = 0;
  while (true) {
    camera.read(img, timestamp);
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    if (gimbal) {
      q = gimbal->q(timestamp);
    }

    // 在图像上显示欧拉角，用来判断imuabs系的xyz正方向，同时判断imu是否存在零漂
    auto img_with_ypr = img.clone();
    if (gimbal) {
      Eigen::Vector3d zyx = tools::eulers(q, 2, 1, 0) * 57.3;  // degree
      tools::draw_text(img_with_ypr, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
      tools::draw_text(img_with_ypr, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
      tools::draw_text(img_with_ypr, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});
    }

    std::vector<cv::Point2f> centers_2d;
    bool success = false;
    if (pattern_type == "chessboard" || pattern_type == "checkerboard") {
      int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                  cv::CALIB_CB_FAST_CHECK;
      success = cv::findChessboardCorners(img, pattern_size, centers_2d, flags);
      if (success) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(
          gray, centers_2d, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
      }
    } else {
      success = cv::findCirclesGrid(img, pattern_size, centers_2d, cv::CALIB_CB_SYMMETRIC_GRID);
    }
    cv::drawChessboardCorners(img_with_ypr, pattern_size, centers_2d, success);  // 显示识别结果
    cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5);  // 显示时缩小图片尺寸

    // 按“s”保存图片和对应四元数，按“q”退出程序
    cv::imshow("Press s to save, q to quit", img_with_ypr);
    auto key = cv::waitKey(1);
    if (key == 'q')
      break;
    else if (key != 's')
      continue;

    // 保存图片和四元数
    count++;
    auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
    auto q_path = fmt::format("{}/{}.txt", output_folder, count);
    cv::imwrite(img_path, img);
    if (use_imu) write_q(q_path, q);
    tools::logger()->info("[{}] Saved in {}", count, output_folder);
  }

  // 离开该作用域时，camera和gimbal会自动关闭
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto output_folder = cli.get<std::string>("output-folder");
  auto use_imu = cli.get<bool>("imu");

  std::string img_folder = output_folder;
  if (!use_imu) {
    auto parent = std::filesystem::path(output_folder).parent_path();
    img_folder = (parent / "img").string();
  }

  // 新建输出文件夹
  std::filesystem::create_directory(img_folder);

  tools::logger()->info("默认标定板尺寸为10列7行");
  // 判断是否存在可用 IMU
  std::unique_ptr<io::Gimbal> gimbal;
  bool imu_available = false;
  try {
    auto yaml = YAML::LoadFile(config_path);
    if (yaml["com_port"]) {
      auto com_port = yaml["com_port"].as<std::string>();
      if (!com_port.empty() && std::filesystem::exists(com_port)) {
        imu_available = true;
      } else {
        tools::logger()->warn("[IMU] com_port missing or not found: {}", com_port);
      }
    } else {
      tools::logger()->warn("[IMU] com_port not found in config, fallback to no-imu.");
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("[IMU] Failed to read config for com_port: {}", e.what());
  }

  if (use_imu) {
    if (!imu_available) {
      tools::logger()->error("[IMU] --imu set but IMU unavailable, exiting.");
      return 1;
    }
    gimbal = std::make_unique<io::Gimbal>(config_path);
  } else {
    gimbal.reset();
  }

  // 主循环，保存图片和对应四元数
  capture_loop(config_path, img_folder, use_imu, std::move(gimbal));

  tools::logger()->warn("注意四元数输出顺序为wxyz");

  return 0;
}
