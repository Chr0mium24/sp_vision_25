#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "io/gimbal/gimbal.hpp"
#include "tools/logger.hpp"

const std::string keys =
    "{help h usage ? |      | show help message }"
    "{yaw y          | 0.0  | target yaw (rad) }"
    "{pitch p        | 0.0  | target pitch (rad) }"
    "{fire f         | false| fire or not }"
    "{fric           | true | friction wheels on or off }"
    "{@config-path   |      | path to yaml config file }";

int main(int argc, char** argv) {
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>("@config-path");
    
    if (cli.has("help") || config_path.empty()) {
        cli.printMessage();
        return 0;
    }

    float target_yaw = cli.get<float>("yaw");
    float target_pitch = cli.get<float>("pitch");
    bool fire = cli.get<bool>("fire");
    bool fric = cli.get<bool>("fric");

    try {
        io::Gimbal gimbal(config_path);
        
        tools::logger()->info("Gimbal Ctrl Start: Yaw={}, Pitch={}, Fire={}, Fric={}", 
                             target_yaw, target_pitch, fire, fric);
        
        tools::logger()->info("Press Ctrl+C to stop. Sending commands and printing feedback...");

        while (true) {
            gimbal.send(true, fire, target_yaw, 0, 0, target_pitch, 0, 0);
            
            auto state = gimbal.state();
            // Just print one TX hex sample
            static bool printed_tx = false;
            if (!printed_tx) {
                uint8_t buffer[14];
                // Since tx_data_ is private, we can't easily see it here without a getter, 
                // but we can trust the log inside send() if we enable it.
                // Let's just print the values we sent.
                tools::logger()->info("TX Values: Yaw={:.3f}, Pitch={:.3f}, Tracking=1", target_yaw, target_pitch);
                printed_tx = true;
            }
            tools::logger()->info("RX: Yaw={:.3f}, Pitch={:.3f}, Roll={:.3f}, YawOdom={:.3f}, PitchOdom={:.3f}, ID={}", 
                                 state.yaw, state.pitch, state.roll, state.yaw_odom, state.pitch_odom, (int)state.robot_id);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
    } catch (const std::exception& e) {
        tools::logger()->error("Error: {}", e.what());
        return -1;
    }

    return 0;
}
