#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>
#include <Eigen/Geometry>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }"
  "{show s         | false  | 是否显示调试窗口}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  bool show = cli.get<bool>("show");

  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;

  // 使用库中的 IO 组件
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  // 初始化视觉任务组件
  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;
  nlohmann::json log_data;

  tools::logger()->info("Minimum Vision System Started.");

  while (!exiter.exit()) {
    // 1. 读取图像
    camera.read(img, t);
    if (img.empty()) {
      continue;
    }

    // 2. 获取对应时间戳的四元数 (从 io::Gimbal 获取)
    q = gimbal.q(t - 1ms);
    solver.set_R_gimbal2world(q);

    // 3. 识别装甲板
    auto armors = detector.detect(img);

    // 4. 目标追踪
    auto targets = tracker.track(armors, t);

    // 5. 弹道推算与瞄准
    // 注意：这里 bullet_speed 暂时设为固定值，或者从 gimbal.state() 获取获取其反馈（如果电控支持）
    auto command = aimer.aim(targets, t, 25.0);

    // 6. 发送控制指令
    gimbal.send(command.control, command.shoot, (float)command.yaw, 0, 0, (float)command.pitch, 0, 0);

    // --- 可视化与日志 ---
    log_data["fps"] = 1.0 / tools::delta_time(std::chrono::steady_clock::now(), t);
    
    if (show) {
      if (!targets.empty()) {
        auto target = targets.front();
        tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

        // 绘制预测位置
        for (const auto & xyza : target.armor_xyza_list()) {
          auto image_points = solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 255, 0});
        }

        // 绘制瞄准点
        auto aim_point = aimer.debug_aim_point;
        if (aim_point.valid) {
          auto image_points = solver.reproject_armor(aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 0, 255}); 
        }
      }

      cv::resize(img, img, {}, 0.5, 0.5); 
      cv::imshow("Minimum Vision System", img);
      if (cv::waitKey(1) == 'q') break;
    }
  }

  return 0;
}