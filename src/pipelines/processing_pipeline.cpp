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

    // Linear interpolation between valid positions
    for (size_t i = 0; i < valid_positions.size() - 1; ++i) {
        int frame1 = valid_positions[i].first;
        int frame2 = valid_positions[i + 1].first;
        const BBox& bbox1 = valid_positions[i].second;
        const BBox& bbox2 = valid_positions[i + 1].second;

        int gap = frame2 - frame1;
        if (gap > BALL_INTERPOLATION_LIMIT) continue;

        for (int f = frame1 + 1; f < frame2; ++f) {
            float t = static_cast<float>(f - frame1) / static_cast<float>(gap);

            BBox interp;
            interp.x1 = bbox1.x1 + t * (bbox2.x1 - bbox1.x1);
            interp.y1 = bbox1.y1 + t * (bbox2.y1 - bbox1.y1);
            interp.x2 = bbox1.x2 + t * (bbox2.x2 - bbox1.x2);
            interp.y2 = bbox1.y2 + t * (bbox2.y2 - bbox1.y2);
            interp.confidence = 0.5f;
            interp.class_id = 1;

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
