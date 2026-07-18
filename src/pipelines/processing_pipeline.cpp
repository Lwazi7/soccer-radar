#include "pipelines/processing_pipeline.hpp"
#include "utils/constants.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace soccer_radar {

void ProcessingPipeline::interpolate_ball_tracks(
    std::unordered_map<int, BBox>& ball_tracks,
    int total_frames) {

    std::cout << "[ProcessingPipeline] Interpolating ball tracks..." << std::endl;

    // Collect valid ball positions
    std::vector<std::pair<int, BBox>> valid_positions;
    for (int i = 0; i < total_frames; ++i) {
        auto it = ball_tracks.find(i);
        if (it != ball_tracks.end() && it->second.valid()) {
            valid_positions.emplace_back(i, it->second);
        }
    }

    if (valid_positions.size() < 2) return;

    // Constant-acceleration interpolation. Initial velocity is estimated from the
    // preceding observation; acceleration is then solved so the trajectory reaches
    // the next detection exactly. Box size is interpolated independently.
    for (size_t i = 0; i + 1 < valid_positions.size(); ++i) {
        const int frame1 = valid_positions[i].first;
        const int frame2 = valid_positions[i + 1].first;
        const BBox& b0 = valid_positions[i].second;
        const BBox& b1 = valid_positions[i + 1].second;
        const int gap = frame2 - frame1;
        if (gap <= 1 || gap > BALL_INTERPOLATION_LIMIT) continue;

        const float gap_f = static_cast<float>(gap);
        float vx = (b1.cx() - b0.cx()) / gap_f;
        float vy = (b1.cy() - b0.cy()) / gap_f;
        if (i > 0) {
            const int previous_gap = frame1 - valid_positions[i - 1].first;
            if (previous_gap > 0 && previous_gap <= BALL_INTERPOLATION_LIMIT) {
                const BBox& previous = valid_positions[i - 1].second;
                const float previous_gap_f = static_cast<float>(previous_gap);
                vx = (b0.cx() - previous.cx()) / previous_gap_f;
                vy = (b0.cy() - previous.cy()) / previous_gap_f;
            }
        }
        const float gap_sq = gap_f * gap_f;
        const float ax = 2.0f * (b1.cx() - b0.cx() - vx * gap_f) / gap_sq;
        const float ay = 2.0f * (b1.cy() - b0.cy() - vy * gap_f) / gap_sq;

        for (int f = frame1 + 1; f < frame2; ++f) {
            const float dt = static_cast<float>(f - frame1);
            const float t = dt / static_cast<float>(gap);
            const float cx = b0.cx() + vx * dt + 0.5f * ax * dt * dt;
            const float cy = b0.cy() + vy * dt + 0.5f * ay * dt * dt;
            const float w = b0.width() + t * (b1.width() - b0.width());
            const float h = b0.height() + t * (b1.height() - b0.height());

            BBox interp;
            interp.x1 = std::clamp(cx - 0.5f * w, 0.0f, static_cast<float>(INPUT_WIDTH));
            interp.y1 = std::clamp(cy - 0.5f * h, 0.0f, static_cast<float>(INPUT_HEIGHT));
            interp.x2 = std::clamp(cx + 0.5f * w, 0.0f, static_cast<float>(INPUT_WIDTH));
            interp.y2 = std::clamp(cy + 0.5f * h, 0.0f, static_cast<float>(INPUT_HEIGHT));
            interp.confidence = 0.35f;
            interp.class_id = static_cast<int>(ObjectClass::Ball);
            ball_tracks[f] = interp;
        }
    }
}

std::string ProcessingPipeline::generate_output_path(const std::string& input,
                                                      const std::string& suffix) {
    std::filesystem::path p(input);
    std::string stem = p.stem().string();
    std::string ext = p.extension().string();
    if (ext.empty()) ext = ".mp4";
    return (p.parent_path() / (stem + suffix + ext)).string();
}

} // namespace soccer_radar
