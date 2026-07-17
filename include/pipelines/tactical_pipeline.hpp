#pragma once

#include "tactical_analysis/homography.hpp"
#include "utils/types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace soccer_radar {

// Tactical analysis: homography transformation and pitch visualization.
class TacticalPipeline {
public:
    TacticalPipeline();
    ~TacticalPipeline() = default;

    // Process detections and keypoints for tactical analysis
    // Returns tactical frame (pitch view) and metadata
    cv::Mat process_frame(const Detections& players,
                          const Detections& balls,
                          const Detections& referees,
                          const KeypointData& keypoints,
                          TacticalMetadata& metadata);

    // Create overlay: tactical view on top of original frame
    static cv::Mat create_overlay(const cv::Mat& original,
                                   const cv::Mat& tactical,
                                   int overlay_w = OVERLAY_WIDTH,
                                   int overlay_h = OVERLAY_HEIGHT);

private:
    HomographyTransformer homography_;
};

} // namespace soccer_radar
