#pragma once

#include "keypoint_detection/keypoint_detector.hpp"
#include "object_annotations/annotator.hpp"
#include "utils/types.hpp"
#include <opencv2/core.hpp>

namespace soccer_radar {

// Coordinates keypoint detection workflows.
class KeypointPipeline {
public:
    KeypointPipeline();
    ~KeypointPipeline() = default;

    bool initialize(const std::string& model_path);

    // Detect field keypoints in a single frame
    KeypointData detect_frame(const cv::Mat& frame);

    KeypointDetector& detector() { return detector_; }

private:
    KeypointDetector detector_;
    Annotator annotator_;
};

} // namespace soccer_radar
