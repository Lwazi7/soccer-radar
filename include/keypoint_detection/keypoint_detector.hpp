#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace soccer_radar {

class KeypointDetector {
public:
    KeypointDetector();
    ~KeypointDetector();

    KeypointDetector(const KeypointDetector&) = delete;
    KeypointDetector& operator=(const KeypointDetector&) = delete;

    bool load_model(const std::string& model_path);
    bool is_loaded() const { return session_ != nullptr; }

    // Detect keypoints in a frame (applies letterbox internally)
    // Returns keypoints with shape [NUM_KEYPOINTS][3] (x, y, confidence)
    KeypointData detect(const cv::Mat& frame);

private:
    void preprocess(const cv::Mat& frame, std::vector<float>& blob);
    void postprocess(const std::vector<float>& output, KeypointData& out);

    void* session_{nullptr};
    void* env_{nullptr};
    void* mem_info_{nullptr};

    // Pre-allocated buffers
    std::vector<float> input_blob_;
    std::vector<float> output_data_;
    cv::Mat letterboxed_;

    int input_height_{MODEL_HEIGHT};
    int input_width_{MODEL_WIDTH};
    std::string input_name_;
    std::vector<std::string> output_names_;
};

} // namespace soccer_radar
