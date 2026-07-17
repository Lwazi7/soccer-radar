#pragma once

#include "object_detection/object_detector.hpp"
#include "object_annotations/annotator.hpp"
#include "utils/types.hpp"
#include <opencv2/core.hpp>

namespace soccer_radar {

class DetectionPipeline {
public:
    DetectionPipeline();
    ~DetectionPipeline() = default;

    bool initialize(const std::string& model_path);

    void detect_frame(const cv::Mat& frame,
                      Detections& players,
                      Detections& balls,
                      Detections& referees,
                      DetectionTiming* timing = nullptr);

    ObjectDetector& detector() { return detector_; }

private:
    ObjectDetector detector_;
    Annotator annotator_;
};

} // namespace soccer_radar
