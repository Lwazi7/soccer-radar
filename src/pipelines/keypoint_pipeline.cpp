#include "pipelines/keypoint_pipeline.hpp"
#include <iostream>

namespace soccer_radar {

KeypointPipeline::KeypointPipeline() {}

bool KeypointPipeline::initialize(const std::string& model_path) {
    return detector_.load_model(model_path);
}

KeypointData KeypointPipeline::detect_frame(const cv::Mat& frame) {
    return detector_.detect(frame);
}

} // namespace soccer_radar
