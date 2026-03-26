#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "io/gimbal/gimbal.hpp"
#include "tools/exiter.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono_literals;

namespace
{
struct AxisSample
{
  double yaw = 0.0;
  double pitch = 0.0;
  double roll = 0.0;
  int count = 0;
};

struct AxisResult
{
  const char * command_name = "";
  double step_deg = 0.0;
  std::array<double, 3> delta_deg{0.0, 0.0, 0.0};
  int dominant_axis = 0;
  bool sign_match = false;
};

struct Candidate
{
  Eigen::Matrix3d R_gimbal2imubody = Eigen::Matrix3d::Identity();
  std::array<int, 9> row_major{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::string label;
};

struct CandidateResult
{
  Candidate candidate;
  std::array<double, 3> yaw_delta_deg{0.0, 0.0, 0.0};
  std::array<double, 3> pitch_delta_deg{0.0, 0.0, 0.0};
  int yaw_dominant_axis = 0;
  int pitch_dominant_axis = 0;
  bool yaw_sign_match = false;
  bool pitch_sign_match = false;
  double score = -std::numeric_limits<double>::infinity();
};

AxisSample average_state(io::Gimbal & gimbal, tools::Exiter & exiter, int duration_ms, int loop_ms)
{
  AxisSample sample;
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

AxisSample average_transformed_state(
  io::Gimbal & gimbal, tools::Exiter & exiter, const Eigen::Matrix3d & R_gimbal2imubody,
  int duration_ms, int loop_ms)
{
  AxisSample sample;
  auto start = std::chrono::steady_clock::now();
  while (!exiter.exit()) {
    auto now = std::chrono::steady_clock::now();
    if (now - start >= std::chrono::milliseconds(duration_ms)) break;
    Eigen::Quaterniond q = gimbal.q(now);
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    Eigen::Matrix3d R_gimbal2world =
      R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;
    Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0);
    sample.yaw += ypr[0];
    sample.pitch += ypr[1];
    sample.roll += ypr[2];
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

void send_plan(io::Gimbal & gimbal, double yaw, double pitch)
{
  io::VisionToGimbal plan{};
  plan.tracking = 1;
  plan.fric_on = 0;
  plan.fire = 0;
  plan.yaw = static_cast<float>(yaw);
  plan.pitch = static_cast<float>(pitch);
  gimbal.send(plan);
}

AxisResult run_axis_step(
  io::Gimbal & gimbal, tools::Exiter & exiter, const char * command_name, double base_yaw,
  double base_pitch, double step_deg, bool command_yaw, int settle_ms, int sample_ms, int loop_ms)
{
  send_plan(gimbal, base_yaw, base_pitch);
  std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
  auto base = average_state(gimbal, exiter, sample_ms, loop_ms);

  double target_yaw = base_yaw;
  double target_pitch = base_pitch;
  if (command_yaw) {
    target_yaw += step_deg / 57.3;
  } else {
    target_pitch += step_deg / 57.3;
  }

  send_plan(gimbal, target_yaw, target_pitch);
  std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
  auto moved = average_state(gimbal, exiter, sample_ms, loop_ms);

  send_plan(gimbal, base_yaw, base_pitch);
  std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));

  AxisResult result;
  result.command_name = command_name;
  result.step_deg = step_deg;
  result.delta_deg = {
    (moved.yaw - base.yaw) * 57.3,
    (moved.pitch - base.pitch) * 57.3,
    (moved.roll - base.roll) * 57.3};

  for (int i = 1; i < 3; ++i) {
    if (std::abs(result.delta_deg[i]) > std::abs(result.delta_deg[result.dominant_axis])) {
      result.dominant_axis = i;
    }
  }

  const int expected_axis = command_yaw ? 0 : 1;
  result.sign_match =
    result.dominant_axis == expected_axis && ((step_deg > 0) == (result.delta_deg[expected_axis] > 0));
  return result;
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

std::vector<Candidate> build_candidates()
{
  std::vector<Candidate> candidates;
  const std::array<int, 6> perms{0, 1, 2, 3, 4, 5};
  const std::array<std::array<int, 3>, 6> perm_axes{
    std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 1}, std::array<int, 3>{1, 0, 2},
    std::array<int, 3>{1, 2, 0}, std::array<int, 3>{2, 0, 1}, std::array<int, 3>{2, 1, 0}};

  for (int perm_idx : perms) {
    const auto & axes = perm_axes[perm_idx];
    for (int sx : {-1, 1}) {
      for (int sy : {-1, 1}) {
        for (int sz : {-1, 1}) {
          Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
          R(axes[0], 0) = sx;
          R(axes[1], 1) = sy;
          R(axes[2], 2) = sz;
          if (std::round(R.determinant()) != 1.0) continue;

          Candidate candidate;
          candidate.R_gimbal2imubody = R;
          candidate.row_major = {
            static_cast<int>(R(0, 0)), static_cast<int>(R(0, 1)), static_cast<int>(R(0, 2)),
            static_cast<int>(R(1, 0)), static_cast<int>(R(1, 1)), static_cast<int>(R(1, 2)),
            static_cast<int>(R(2, 0)), static_cast<int>(R(2, 1)), static_cast<int>(R(2, 2))};
          candidate.label = cv::format(
            "[%d,%d,%d;%d,%d,%d;%d,%d,%d]", candidate.row_major[0], candidate.row_major[1],
            candidate.row_major[2], candidate.row_major[3], candidate.row_major[4],
            candidate.row_major[5], candidate.row_major[6], candidate.row_major[7],
            candidate.row_major[8]);
          candidates.push_back(candidate);
        }
      }
    }
  }
  return candidates;
}

int dominant_axis(const std::array<double, 3> & delta_deg)
{
  int axis = 0;
  for (int i = 1; i < 3; ++i) {
    if (std::abs(delta_deg[i]) > std::abs(delta_deg[axis])) axis = i;
  }
  return axis;
}

std::array<double, 3> delta_deg(const AxisSample & after, const AxisSample & before)
{
  return {(after.yaw - before.yaw) * 57.3, (after.pitch - before.pitch) * 57.3,
          (after.roll - before.roll) * 57.3};
}

double candidate_score(
  const std::array<double, 3> & yaw_delta_deg, const std::array<double, 3> & pitch_delta_deg)
{
  return std::abs(yaw_delta_deg[0]) * 2.0 + std::abs(pitch_delta_deg[1]) * 2.0 -
         std::abs(yaw_delta_deg[1]) - std::abs(yaw_delta_deg[2]) - std::abs(pitch_delta_deg[0]) -
         std::abs(pitch_delta_deg[2]);
}

std::vector<CandidateResult> evaluate_candidates(
  io::Gimbal & gimbal, tools::Exiter & exiter, const std::vector<Candidate> & candidates,
  int sample_ms, int loop_ms, double base_yaw, double base_pitch, double step_deg, int settle_ms)
{
  std::vector<CandidateResult> results;
  results.reserve(candidates.size());

  send_plan(gimbal, base_yaw, base_pitch);
  std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));

  std::vector<AxisSample> baseline_samples;
  baseline_samples.reserve(candidates.size());
  for (const auto & candidate : candidates) {
    baseline_samples.push_back(
      average_transformed_state(gimbal, exiter, candidate.R_gimbal2imubody, sample_ms, loop_ms));
  }

  send_plan(gimbal, base_yaw + step_deg / 57.3, base_pitch);
  std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
  std::vector<AxisSample> yaw_samples;
  yaw_samples.reserve(candidates.size());
  for (const auto & candidate : candidates) {
    yaw_samples.push_back(
      average_transformed_state(gimbal, exiter, candidate.R_gimbal2imubody, sample_ms, loop_ms));
  }

  send_plan(gimbal, base_yaw, base_pitch);
  std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));

  send_plan(gimbal, base_yaw, base_pitch + step_deg / 57.3);
  std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
  std::vector<AxisSample> pitch_samples;
  pitch_samples.reserve(candidates.size());
  for (const auto & candidate : candidates) {
    pitch_samples.push_back(
      average_transformed_state(gimbal, exiter, candidate.R_gimbal2imubody, sample_ms, loop_ms));
  }

  send_plan(gimbal, base_yaw, base_pitch);
  std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));

  for (size_t i = 0; i < candidates.size(); ++i) {
    CandidateResult result;
    result.candidate = candidates[i];
    result.yaw_delta_deg = delta_deg(yaw_samples[i], baseline_samples[i]);
    result.pitch_delta_deg = delta_deg(pitch_samples[i], baseline_samples[i]);
    result.yaw_dominant_axis = dominant_axis(result.yaw_delta_deg);
    result.pitch_dominant_axis = dominant_axis(result.pitch_delta_deg);
    result.yaw_sign_match =
      result.yaw_dominant_axis == 0 && result.yaw_delta_deg[0] > 0.0;
    result.pitch_sign_match =
      result.pitch_dominant_axis == 1 && result.pitch_delta_deg[1] > 0.0;
    result.score = candidate_score(result.yaw_delta_deg, result.pitch_delta_deg);
    if (!result.yaw_sign_match) result.score -= 6.0;
    if (!result.pitch_sign_match) result.score -= 6.0;
    if (result.yaw_dominant_axis != 0) result.score -= 4.0;
    if (result.pitch_dominant_axis != 1) result.score -= 4.0;
    results.push_back(result);
  }

  std::sort(results.begin(), results.end(), [](const CandidateResult & a, const CandidateResult & b) {
    return a.score > b.score;
  });
  return results;
}

void print_candidate(const CandidateResult & result, int rank)
{
  std::printf(
    "[candidate %d] score=%.2f R_gimbal2imubody=%s yaw(delta=%+.2f,%+.2f,%+.2f dominant=%s sign_ok=%d) pitch(delta=%+.2f,%+.2f,%+.2f dominant=%s sign_ok=%d)\n",
    rank, result.score, result.candidate.label.c_str(), result.yaw_delta_deg[0],
    result.yaw_delta_deg[1], result.yaw_delta_deg[2], axis_name(result.yaw_dominant_axis),
    result.yaw_sign_match ? 1 : 0, result.pitch_delta_deg[0], result.pitch_delta_deg[1],
    result.pitch_delta_deg[2], axis_name(result.pitch_dominant_axis),
    result.pitch_sign_match ? 1 : 0);
}

void print_result(const AxisResult & result)
{
  std::printf(
    "[axis] cmd=%s step=%+.2fdeg delta(yaw=%+.2f, pitch=%+.2f, roll=%+.2f) dominant=%s sign_ok=%d\n",
    result.command_name, result.step_deg, result.delta_deg[0], result.delta_deg[1],
    result.delta_deg[2], axis_name(result.dominant_axis), result.sign_match ? 1 : 0);
}

void print_conclusion(const AxisResult & yaw_result, const AxisResult & pitch_result)
{
  const bool yaw_axis_ok = yaw_result.dominant_axis == 0;
  const bool pitch_axis_ok = pitch_result.dominant_axis == 1;
  const bool axis_mapping_ok = yaw_axis_ok && pitch_axis_ok;
  const bool sign_ok = yaw_result.sign_match && pitch_result.sign_match;

  std::puts("");
  if (axis_mapping_ok && sign_ok) {
    std::puts(
      "[conclusion] yaw/pitch commands map to the expected feedback axes with the expected sign.");
    std::puts(
      "[conclusion] This looks like a small zero-offset problem. Check yaw_offset/pitch_offset first.");
    return;
  }

  if (!axis_mapping_ok) {
    std::puts("[conclusion] At least one commanded axis moves the wrong feedback axis.");
    std::puts(
      "[conclusion] This points to a C-board or IMU axis mapping problem. Check R_gimbal2imubody.");
  } else {
    std::puts("[conclusion] Axis mapping looks right, but at least one axis sign is opposite.");
    std::puts(
      "[conclusion] This points to an axis direction/sign problem in the IMU-to-gimbal mapping.");
  }

  if (yaw_result.dominant_axis == 2 || pitch_result.dominant_axis == 2) {
    std::puts(
      "[hint] Strong roll response to yaw/pitch commands often means the C-board is mounted 90 degrees off.");
  }
}
}  // namespace

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }"
  "{step-deg       | 5 | 测试步进角(度), 建议3~8 }"
  "{settle-ms      | 700 | 下发后等待稳定时长(ms) }"
  "{sample-ms      | 250 | 取平均采样时长(ms) }"
  "{wait-valid-ms  | 1500 | 等待有效反馈超时(ms) }"
  "{loop-ms        | 5 | 采样循环sleep时长(ms) }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  const double step_deg = std::clamp(cli.get<double>("step-deg"), 0.5, 15.0);
  const int settle_ms = std::max(100, cli.get<int>("settle-ms"));
  const int sample_ms = std::max(50, cli.get<int>("sample-ms"));
  const int wait_valid_ms = std::max(100, cli.get<int>("wait-valid-ms"));
  const int loop_ms = std::max(1, cli.get<int>("loop-ms"));

  tools::Exiter exiter;
  io::Gimbal gimbal(config_path, false);
  if (!wait_valid(gimbal, exiter, wait_valid_ms)) {
    std::fprintf(stderr, "[axis] failed to receive valid gimbal feedback within %d ms\n", wait_valid_ms);
    return 2;
  }

  auto base_state = gimbal.state();
  const double base_yaw = base_state.yaw;
  const double base_pitch = base_state.pitch;

  std::printf(
    "[axis] baseline command yaw=%+.2fdeg pitch=%+.2fdeg\n",
    base_yaw * 57.3, base_pitch * 57.3);

  send_plan(gimbal, base_yaw, base_pitch);
  std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));

  const auto yaw_result = run_axis_step(
    gimbal, exiter, "yaw+", base_yaw, base_pitch, step_deg, true, settle_ms, sample_ms, loop_ms);
  const auto pitch_result = run_axis_step(
    gimbal, exiter, "pitch+", base_yaw, base_pitch, step_deg, false, settle_ms, sample_ms, loop_ms);

  print_result(yaw_result);
  print_result(pitch_result);
  print_conclusion(yaw_result, pitch_result);

  std::puts("");
  std::puts("[sweep] ranking axis-aligned R_gimbal2imubody candidates:");
  const auto candidates = build_candidates();
  const auto candidate_results = evaluate_candidates(
    gimbal, exiter, candidates, sample_ms, loop_ms, base_yaw, base_pitch, step_deg, settle_ms);
  const int top_n = std::min<int>(5, candidate_results.size());
  for (int i = 0; i < top_n; ++i) {
    print_candidate(candidate_results[i], i + 1);
  }

  if (!candidate_results.empty()) {
    const auto & best = candidate_results.front();
    std::printf(
      "[recommend] try setting R_gimbal2imubody: [%d, %d, %d, %d, %d, %d, %d, %d, %d]\n",
      best.candidate.row_major[0], best.candidate.row_major[1], best.candidate.row_major[2],
      best.candidate.row_major[3], best.candidate.row_major[4], best.candidate.row_major[5],
      best.candidate.row_major[6], best.candidate.row_major[7], best.candidate.row_major[8]);
  }

  io::VisionToGimbal stop{};
  gimbal.send(stop);
  return 0;
}
