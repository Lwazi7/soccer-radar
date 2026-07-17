#include "pipelines/detection_pipeline.hpp"
#include <iostream>

namespace soccer_radar {

DetectionPipeline::DetectionPipeline() {}

bool DetectionPipeline::initialize(const std::string& model_path) {
    return detector_.load_model(model_path);
}

void DetectionPipeline::detect_frame(const cv::Mat& frame,
                                      Detections& players,
                                      Detections& balls,
                                      Detections& referees,
                                      DetectionTiming* timing) {
    detector_.detect(frame, players, balls, referees, timing);
}

} // namespace soccer_radar
